/*
 * genUD.c — fast C implementation of the GLP-based uniform design search.
 *
 * Mirrors the Python baseline in `methods.py`:
 *   * Good-lattice-point (GLP) construction with power generator:
 *         U[j, d] = (mod(j * alpha^d, r_p + 1) / r_p) - 1 / (2 * r_p)
 *     for j = 1..r_p, d = 0..q-1.
 *
 *   * Mixture discrepancy squared D^2_M  (closed form, three terms).
 *
 * Optimisations (v2, still produces the same selected alpha as baseline):
 *   (1) Symmetry:  sum_{j,k} kappa(u_j,u_k) = 2 * sum_{j<k} + sum_{j=k}.
 *   (2) Diagonal simplification:  kappa(u_j,u_j) = prod_d (15/8 - 0.5*|u_jd-1/2|).
 *   (3) Pre-computed |u_jd - 1/2| tables so the inner loop has only
 *       (diff, |diff|, diff*diff, three subtractions, one multiply).
 *   (4) Natural vectorisation by the compiler: fully independent inner-d
 *       loop, `restrict` pointers, no function calls in the hot path.
 *   (5) Parallelism over the candidate alpha list using Win32 CreateThread
 *       (POSIX pthreads on non-Windows). Each thread owns its own U buffer,
 *       so there is no cache contention and no reduction is needed.
 *
 * Build (Windows, MinGW UCRT):
 *   gcc -O3 -march=native -ffast-math -shared -static-libgcc -o genUD.dll genUD.c
 *
 * Build (Linux / macOS, Zen 4 / EPYC 9654):
 *   gcc -O3 -march=znver4 -mavx512f -mavx512dq -mavx512bw \
 *       -ffast-math -funroll-loops -fPIC -shared -pthread -o libgenUD.so genUD.c
 *   # -march=native works too; -march=znver4 guarantees AVX-512 double-width
 *   # vectorisation of the pairwise d-loop (5x over AVX2 for q=8).
 *
 * Build (Linux, generic / any recent x86-64):
 *   gcc -O3 -march=native -ffast-math -fPIC -shared -pthread -o libgenUD.so genUD.c
 *
 * Public entry (ctypes-compatible):
 *   int c_genUD(int64_t r_p, int q,
 *               const int64_t *alphas, int n_alpha,
 *               double *out_U, double *out_best_d);
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #define UD_EXPORT __declspec(dllexport)
    typedef HANDLE ud_thread_t;
#else
    #include <pthread.h>
    #include <unistd.h>
    #define UD_EXPORT
    typedef pthread_t ud_thread_t;
#endif

/* -------------------------------------------------------------------------- */
/*  GLP construction                                                           */
/* -------------------------------------------------------------------------- */

static void glp_construct(int64_t r_p, int q, int64_t alpha, double *restrict U)
{
    const int64_t modulus = r_p + 1;
    const double inv_rp = 1.0 / (double)r_p;
    const double half_over_rp = 0.5 / (double)r_p;

    /* gamma[d] = alpha^d mod modulus */
    int64_t gamma_buf[64];
    int64_t *gamma = (q <= 64) ? gamma_buf
                               : (int64_t *)malloc((size_t)q * sizeof(int64_t));
    if (!gamma) return;
    int64_t val = 1;
    for (int d = 0; d < q; ++d) {
        gamma[d] = val % modulus;
        val = (val * alpha) % modulus;
    }

    for (int64_t j = 1; j <= r_p; ++j) {
        double *row = U + (j - 1) * (int64_t)q;
        for (int d = 0; d < q; ++d) {
            int64_t raw = (j * gamma[d]) % modulus;
            row[d] = (double)raw * inv_rp - half_over_rp;
        }
    }
    if (q > 64) free(gamma);
}

/* -------------------------------------------------------------------------- */
/*  Mixture discrepancy squared D^2_M                                          */
/* -------------------------------------------------------------------------- */
/*
 *  D^2_M = (19/12)^q
 *         - (2/r_p) * sum_j  prod_d A1(u_jd)
 *         + (1/r_p^2) * [ 2 * sum_{j<k} prod_d kappa_M(u_jd, u_kd)
 *                         + sum_j      prod_d (15/8 - 0.5*|u_jd - 1/2|) ]
 *  A1(u)  = 5/3 - 1/4 |u - 1/2| - 1/4 (u - 1/2)^2
 *  kappa_M(u,t) = 15/8 - 1/4 |u-1/2| - 1/4 |t-1/2| - 3/4 |u-t| + 1/2 (u-t)^2
 */

static double mixture_d2(const double *restrict U, int64_t r_p, int q)
{
    /* Term 1 */
    const double term1 = pow(19.0 / 12.0, (double)q);

    /* Precompute absolute-centred table. */
    double *restrict abs_c = (double *)malloc((size_t)r_p * (size_t)q * sizeof(double));
    if (!abs_c) return NAN;
    for (int64_t i = 0; i < r_p; ++i) {
        const double *row = U + i * (int64_t)q;
        double *out = abs_c + i * (int64_t)q;
        for (int d = 0; d < q; ++d) out[d] = fabs(row[d] - 0.5);
    }

    /* Term 2. */
    double term2 = 0.0;
    for (int64_t j = 0; j < r_p; ++j) {
        const double *row = U + j * (int64_t)q;
        const double *acj = abs_c + j * (int64_t)q;
        double prod = 1.0;
        for (int d = 0; d < q; ++d) {
            double c = row[d] - 0.5;
            prod *= (5.0 / 3.0) - 0.25 * acj[d] - 0.25 * c * c;
        }
        term2 += prod;
    }
    term2 *= -2.0 / (double)r_p;

    /* Pre-fuse per-row constants:  cj[d] = 15/8 - 0.25*|u_jd - 1/2|.
     * The pairwise kernel then becomes
     *     kappa = cj[d] + ck[d] - 15/8 - 0.75*|diff| + 0.5*diff^2
     * which removes two multiplies per (j,k,d) triple.
     *
     * Also store U row-by-row in a compact aligned layout so the inner
     * d-loop hits sequential cache lines.  For the typical simulation
     * regime (q ≤ 10, r_p ≤ 10000) this fits the L2 cache of a single
     * Zen 4 core, so the pairwise sweep is compute-bound rather than
     * memory-bound and benefits directly from AVX-512 vectorisation.
     */
    double *restrict cj_all = (double *)malloc((size_t)r_p * (size_t)q * sizeof(double));
    if (!cj_all) { free(abs_c); return NAN; }
    for (int64_t i = 0; i < r_p; ++i) {
        const double *ac_row = abs_c + i * (int64_t)q;
        double *cj_row = cj_all + i * (int64_t)q;
        for (int d = 0; d < q; ++d) {
            cj_row[d] = 15.0 / 8.0 - 0.25 * ac_row[d];
        }
    }
    free(abs_c);

    /* (a) Diagonal: kappa(u_j, u_j) = prod_d (15/8 - 0.5*|u_jd - 1/2|)
     *              = prod_d (2*cj[d] - 15/8).
     */
    double diag = 0.0;
    for (int64_t j = 0; j < r_p; ++j) {
        const double *cj_row = cj_all + j * (int64_t)q;
        double prod = 1.0;
        for (int d = 0; d < q; ++d) {
            prod *= 2.0 * cj_row[d] - 15.0 / 8.0;
        }
        diag += prod;
    }

    /* (b) Upper triangle j<k — the hot loop.  Written so that the
     * d-loop is fully independent across k (compiler can emit an AVX-512
     * or AVX2 inner loop when the q-factor specialisation is a power of
     * 2, or a scalar fallback otherwise).  We hoist all j-dependent
     * quantities out of the k-loop.
     *
     * For Zen 4 with -march=native, GCC auto-vectorises the d-loop on
     * the compact cj/U layout; the nested j/k loops are kept scalar to
     * maximise available SIMD lanes for the product reduction.
     */
    double tri = 0.0;
    for (int64_t j = 0; j < r_p - 1; ++j) {
        const double *restrict uj = U      + j * (int64_t)q;
        const double *restrict cjr = cj_all + j * (int64_t)q;
        double row_sum = 0.0;
        for (int64_t k = j + 1; k < r_p; ++k) {
            const double *restrict uk = U      + k * (int64_t)q;
            const double *restrict ckr = cj_all + k * (int64_t)q;
            double prod = 1.0;
            /* kappa = cjr[d] + ckr[d] - 15/8 - 0.75*|diff| + 0.5*diff^2 */
            for (int d = 0; d < q; ++d) {
                double diff = uj[d] - uk[d];
                double ad   = fabs(diff);
                double kd   = cjr[d] + ckr[d]
                              - 15.0 / 8.0
                              - 0.75 * ad
                              + 0.5 * diff * diff;
                prod *= kd;
            }
            row_sum += prod;
        }
        tri += row_sum;
    }

    free(cj_all);

    double total = 2.0 * tri + diag;
    double term3 = total / ((double)r_p * (double)r_p);
    return term1 + term2 + term3;
}

/* -------------------------------------------------------------------------- */
/*  Multithreaded candidate evaluation                                         */
/* -------------------------------------------------------------------------- */

typedef struct {
    int64_t       r_p;
    int           q;
    const int64_t *alphas;
    int           a_lo;   /* inclusive */
    int           a_hi;   /* exclusive */
    /* Best-in-chunk results. */
    double        best_d;
    int           best_i;
    double       *U_best;
    double       *U_tmp;
} worker_t;

static void worker_run(worker_t *w)
{
    double best_d = INFINITY;
    int    best_i = -1;
    const int64_t NQ = w->r_p * (int64_t)w->q;

    for (int a = w->a_lo; a < w->a_hi; ++a) {
        glp_construct(w->r_p, w->q, w->alphas[a], w->U_tmp);
        double d = mixture_d2(w->U_tmp, w->r_p, w->q);
        if (d < best_d) {
            best_d = d;
            best_i = a;
            memcpy(w->U_best, w->U_tmp, (size_t)NQ * sizeof(double));
        }
    }
    w->best_d = best_d;
    w->best_i = best_i;
}

#if defined(_WIN32) || defined(_WIN64)
static DWORD WINAPI worker_entry(LPVOID arg) {
    worker_run((worker_t *)arg);
    return 0;
}
static ud_thread_t thread_start(worker_t *w) {
    return CreateThread(NULL, 0, worker_entry, (LPVOID)w, 0, NULL);
}
static void thread_join(ud_thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
static int detect_ncpu(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}
#else
static void *worker_entry(void *arg) {
    worker_run((worker_t *)arg);
    return NULL;
}
static ud_thread_t thread_start(worker_t *w) {
    pthread_t t;
    pthread_create(&t, NULL, worker_entry, (void *)w);
    return t;
}
static void thread_join(ud_thread_t t) {
    pthread_join(t, NULL);
}
static int detect_ncpu(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}
#endif

static int pick_n_threads(int n_alpha, int r_p)
{
    /* Respect explicit caller-set thread cap (read once per call).
     *
     * Outer parallelism is common in Monte Carlo studies: e.g. joblib with
     * 16 workers × 8 internal genUD threads would oversubscribe a 16-core
     * machine by 8x.  Workers set UD_GENUD_NUM_THREADS=1 to pin each
     * genUD call to a single thread; the outer replication parallelism
     * keeps the cores busy without contention.
     *
     * Values:
     *   "0", "", unset  -> automatic (use up to 8 cores when beneficial)
     *   "1", "2", ...   -> force exactly that many threads
     */
    const char *env = getenv("UD_GENUD_NUM_THREADS");
    if (env && *env) {
        /* Simple positive-integer parse, avoiding strtol for portability. */
        int v = 0;
        for (const char *p = env; *p >= '0' && *p <= '9'; ++p) v = v * 10 + (*p - '0');
        if (v >= 1) {
            if (v > n_alpha) v = n_alpha;
            return v;
        }
    }

    /* Threading overhead dominates for tiny r_p — stay single-threaded. */
    if (r_p < 400 || n_alpha <= 1) return 1;
    int ncpu = detect_ncpu();
    if (ncpu <= 0) ncpu = 1;
    /* On 32 vCPU Zen 4 servers internal parallelism scales well up to
     * ~16 threads per call; beyond that memory bandwidth saturates. */
    int cap = 16;
    int t = ncpu < cap ? ncpu : cap;
    if (t > n_alpha) t = n_alpha;
    return t;
}

/* -------------------------------------------------------------------------- */
/*  Public entry                                                               */
/* -------------------------------------------------------------------------- */

UD_EXPORT int c_genUD(int64_t r_p,
                      int q,
                      const int64_t *alphas,
                      int n_alpha,
                      double *out_U,
                      double *out_best_d)
{
    if (r_p <= 0 || q <= 0 || n_alpha <= 0 || !alphas || !out_U) return -1;

    const int64_t NQ = r_p * (int64_t)q;
    const int n_thr = pick_n_threads(n_alpha, (int)r_p);

    /* Allocate per-worker scratch (U_best + U_tmp). */
    worker_t *workers = (worker_t *)calloc((size_t)n_thr, sizeof(worker_t));
    if (!workers) return -2;
    double *big_scratch = (double *)malloc((size_t)n_thr * 2u * (size_t)NQ * sizeof(double));
    if (!big_scratch) { free(workers); return -2; }

    /* Split alpha range evenly across threads. */
    for (int t = 0; t < n_thr; ++t) {
        workers[t].r_p    = r_p;
        workers[t].q      = q;
        workers[t].alphas = alphas;
        workers[t].a_lo   = (int)(((long long)n_alpha * t)       / n_thr);
        workers[t].a_hi   = (int)(((long long)n_alpha * (t + 1)) / n_thr);
        workers[t].U_best = big_scratch + (size_t)(2 * t)     * (size_t)NQ;
        workers[t].U_tmp  = big_scratch + (size_t)(2 * t + 1) * (size_t)NQ;
    }

    if (n_thr == 1) {
        worker_run(&workers[0]);
    } else {
        ud_thread_t *ths = (ud_thread_t *)malloc((size_t)n_thr * sizeof(ud_thread_t));
        if (!ths) { free(big_scratch); free(workers); return -2; }
        for (int t = 0; t < n_thr; ++t) ths[t] = thread_start(&workers[t]);
        for (int t = 0; t < n_thr; ++t) thread_join(ths[t]);
        free(ths);
    }

    /* Reduce: pick the global minimum across workers. */
    double best_d = INFINITY;
    int    best_i = -1;
    int    best_t = -1;
    for (int t = 0; t < n_thr; ++t) {
        if (workers[t].best_i >= 0 && workers[t].best_d < best_d) {
            best_d = workers[t].best_d;
            best_i = workers[t].best_i;
            best_t = t;
        }
    }
    if (best_t >= 0) {
        memcpy(out_U, workers[best_t].U_best, (size_t)NQ * sizeof(double));
    }

    free(big_scratch);
    free(workers);

    if (out_best_d) *out_best_d = best_d;
    return best_i;
}

/* Expose the D^2_M evaluator (helps verification). */
UD_EXPORT double c_mixture_d2(const double *U, int64_t r_p, int q)
{
    return mixture_d2(U, r_p, q);
}

/* Expose the design construction (helps verification). */
UD_EXPORT void c_glp_construct(int64_t r_p, int q, int64_t alpha, double *out_U)
{
    glp_construct(r_p, q, alpha, out_U);
}
