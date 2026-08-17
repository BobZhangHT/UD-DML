import unittest
from pathlib import Path

import numpy as np

import methods
import simulations
import evaluation


class UDWithoutReplacementTests(unittest.TestCase):
    def test_selected_residual_variance_matches_theory(self):
        Y = np.array([2.0, 0.0, 3.0, -1.0])
        W = np.array([1, 0, 1, 0])
        mu0 = np.zeros(4)
        mu1 = np.ones(4)
        e = np.full(4, 0.5)
        scores = methods._aipw_score(Y, W, mu0, mu1, e)

        estimate, lower, upper, standard_error, variance = (
            methods._selected_residual_inference(scores, Y, W, mu0, mu1, e)
        )

        residual_scores = W * (Y - mu1) / e - (1 - W) * (Y - mu0) / (1 - e)
        expected_variance = float(np.mean(residual_scores ** 2))
        self.assertAlmostEqual(estimate, float(np.mean(scores)))
        self.assertAlmostEqual(variance, expected_variance)
        self.assertAlmostEqual(standard_error, np.sqrt(expected_variance / 4))
        self.assertAlmostEqual(upper - estimate, estimate - lower)

    def test_sparse_assignment_is_one_to_one_under_collisions(self):
        points = np.arange(20, dtype=float)[:, None]
        targets = np.zeros((6, 1), dtype=float)
        tree = methods._build_kdtree(points)

        idx, distances, candidate_k = methods._match_without_replacement(
            tree,
            targets,
            initial_neighbors=1,
            max_neighbors=8,
        )

        self.assertEqual(idx.size, targets.shape[0])
        self.assertEqual(np.unique(idx).size, targets.shape[0])
        self.assertEqual(distances.shape, (targets.shape[0],))
        self.assertGreaterEqual(candidate_k, targets.shape[0])

    def test_ud_selection_has_exact_size_and_arm_balance(self):
        rng = np.random.default_rng(20260811)
        X = rng.normal(size=(600, 6))
        W = np.r_[np.zeros(300, dtype=int), np.ones(300, dtype=int)]
        diagnostics = {}

        idx = methods._select_ud_indices(
            X,
            W,
            80,
            np.random.default_rng(19),
            B_gamma=5,
            cache_seed=991,
            diagnostics=diagnostics,
        )

        self.assertEqual(idx.size, 80)
        self.assertEqual(np.unique(idx).size, 80)
        self.assertEqual(np.bincount(W[idx], minlength=2).tolist(), [40, 40])
        self.assertTrue(diagnostics["without_replacement"])
        self.assertGreaterEqual(diagnostics["matching_max_distance"], 0.0)
        self.assertIn("gefd_estimate", diagnostics)
        self.assertGreaterEqual(diagnostics["gefd_estimate"], 0.0)

    def test_ud_selection_is_deterministic_for_fixed_seed(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(500, 5))
        W = np.r_[np.zeros(250, dtype=int), np.ones(250, dtype=int)]

        idx1 = methods._select_ud_indices(
            X,
            W,
            60,
            np.random.default_rng(23),
            B_gamma=5,
            cache_seed=771,
            diagnostics={},
        )
        idx2 = methods._select_ud_indices(
            X,
            W,
            60,
            np.random.default_rng(23),
            B_gamma=5,
            cache_seed=771,
            diagnostics={},
        )
        np.testing.assert_array_equal(idx1, idx2)

    def test_odd_budget_is_rejected(self):
        X = np.zeros((20, 2))
        W = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
        with self.assertRaisesRegex(ValueError, "even r_total"):
            methods._select_ud_indices(
                X,
                W,
                5,
                np.random.default_rng(1),
                B_gamma=2,
                cache_seed=1,
            )

    def test_ud_budget_larger_than_population_is_rejected(self):
        X = np.arange(24, dtype=float).reshape(8, 3)
        W = np.array([0, 1] * 4)
        Y = np.zeros(8)

        with self.assertRaisesRegex(ValueError, "cannot exceed population size"):
            methods.run_ud(X, W, Y, 0.5, True, {"r_total": 10})

    def test_checkpoint_label_binds_budget_and_population(self):
        label = simulations._compose_variant_label(
            "UD",
            {"label": "baseline", "population_size": 5000},
            {"r_total": 500},
        )
        self.assertIn("r500", label)
        self.assertIn("n5000", label)
        self.assertIn(f"schema{simulations.RESULT_SCHEMA_VERSION}", label)

    def test_diagnostic_experiment_contains_complete_ablation(self):
        _, _, experiments = __import__("config").get_experiments()
        self.assertEqual(
            experiments["experiment_diagnostic_baselines"]["methods"],
            ["UNIF", "STRAT-UNIF", "SEP-UD", "UD"],
        )

    def test_smd_diagnostics_detect_balance(self):
        X = np.array([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [2.0, 3.0]])
        W = np.array([0, 0, 1, 1])
        diagnostics = methods._covariate_smd_diagnostics(X, W)
        self.assertAlmostEqual(diagnostics["smd_mean"], 0.0)
        self.assertAlmostEqual(diagnostics["smd_max"], 0.0)
        self.assertEqual(diagnostics["smd_count_above_0p1"], 0)

    def test_gefd_approximation_is_seed_reproducible(self):
        rng = np.random.default_rng(8)
        Z = rng.normal(size=(100, 2))
        Z_sorted = np.sort(Z, axis=0)
        U = rng.uniform(size=(20, 2))
        first = methods._approximate_gefd(
            Z, Z_sorted, U, seed=77, n_pairs=500
        )
        second = methods._approximate_gefd(
            Z, Z_sorted, U, seed=77, n_pairs=500
        )
        self.assertEqual(first, second)
        self.assertEqual(first["gefd_mc_pairs"], 500)

    def test_fast_demo_output_is_isolated(self):
        old_demo = simulations.FAST_DEMO_MODE
        old_tag = simulations.EXPERIMENT_OUTPUT_TAG
        try:
            simulations.FAST_DEMO_MODE = True
            simulations.EXPERIMENT_OUTPUT_TAG = None
            resolved = simulations._resolve_experiment_output_dir(
                Path("simulation_results") / "diagnostic_baselines"
            )
            self.assertEqual(
                resolved,
                Path("simulation_results") / "diagnostic_baselines_demo",
            )
        finally:
            simulations.FAST_DEMO_MODE = old_demo
            simulations.EXPERIMENT_OUTPUT_TAG = old_tag

    def test_family_overrides_bind_replications_population_and_budget(self):
        old_overrides = simulations.EXPERIMENT_FAMILY_OVERRIDES.copy()
        try:
            simulations.EXPERIMENT_FAMILY_OVERRIDES.update(
                {"n_replications": 20, "population_size": 500_000, "r_total": 5_000}
            )
            variant = simulations._apply_experiment_family_overrides(
                {"n_replications": 500, "population_size": 10_000, "r_total": 1_000}
            )
            self.assertEqual(variant["n_replications"], 20)
            self.assertEqual(variant["population_size"], 500_000)
            self.assertEqual(variant["r_total"], 5_000)
        finally:
            simulations.EXPERIMENT_FAMILY_OVERRIDES.clear()
            simulations.EXPERIMENT_FAMILY_OVERRIDES.update(old_overrides)

    def test_output_tag_rejects_path_characters(self):
        old_tag = simulations.EXPERIMENT_OUTPUT_TAG
        try:
            simulations.EXPERIMENT_OUTPUT_TAG = "../escape"
            with self.assertRaisesRegex(ValueError, "letters, numbers, and underscores"):
                simulations._resolve_experiment_output_dir("simulation_results/base")
        finally:
            simulations.EXPERIMENT_OUTPUT_TAG = old_tag

    def test_tagged_results_get_tagged_analysis_directory(self):
        report = evaluation.generate_reports(
            "experiment_diagnostic_baselines",
            [],
            Path("simulation_results") / "diagnostic_baselines_gate_B20",
        )
        self.assertEqual(
            report["tables_dir"],
            Path("analysis_results") / "diagnostic_baselines_gate_B20" / "tables",
        )


if __name__ == "__main__":
    unittest.main()
