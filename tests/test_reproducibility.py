"""CPU checks that do not train models or require cohort data."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj


budget = module("budget", "scripts/figures/fig_budget.py")
runner = module("runner", "scripts/run_pipeline.py")
prep = module("prep", "scripts/prepare_data.py")


class ReproducibilityTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(fit=seed, predictor="totalVI", map="shared", K=k,
            rep=rep, protein=p, family="lineage", offset_r2=r2,
            below_85=.1, n_calls=nc, n_wrong=nw, n_decisions=100)
            for seed in [2026, 2027] for k in [0, 1] for rep in [0, 1]
            for p, r2, nc, nw in [("A", .2, 10, 5), ("B", .8, 90, 0)]])

    def test_aggregate_counts_not_mean_of_error_rates(self):
        result = budget.summarize(self.fixture())
        np.testing.assert_allclose(result.error_among_calls, .05)
        np.testing.assert_allclose(result.offset_r2, .5)

    def test_only_defined_equivalent_budgets_are_copied(self):
        result = budget.summarize(self.fixture())
        self.assertEqual(set(result["map"]), {"shared", "multi"})
        for k in [0, 1]:
            self.assertEqual(len(result[result.K == k]), 2)

    def test_missing_columns_fail(self):
        with self.assertRaises(ValueError): budget.summarize(self.fixture().drop(columns="rep"))

    def test_duplicate_results_fail(self):
        f = self.fixture()
        with self.assertRaises(ValueError): budget.summarize(pd.concat([f, f.iloc[:1]]))

    def test_incomplete_seed_set_fails(self):
        f = self.fixture()
        with self.assertRaises(ValueError): budget.summarize(f[~((f.fit == 2027) & (f.K == 1))])

    def test_no_calls_is_undefined_not_zero(self):
        f = self.fixture(); f["n_calls"] = 0; f["n_wrong"] = 0
        self.assertTrue(budget.summarize(f).error_among_calls.isna().all())

    def test_pipeline_creates_anchors_before_using_them(self):
        for cohort in ["stephenson", "combat"]:
            scripts = [s for s, _ in runner.plan(cohort, [2026, 2027])]
            first = scripts.index("analysis/baselines_calib.py")
            consumer = scripts.index("analysis/ridge_sentinels.py")
            self.assertLess(first, consumer)
            self.assertIn("analysis/baselines_calib.py", scripts[consumer + 1:])
            self.assertTrue(all((ROOT / "scripts" / s).is_file() for s in scripts))

    def test_counts_reject_normalized_and_nonfinite_data(self):
        prep.check_counts(np.array([[0, 1], [2, 4.0]]), "counts")
        for invalid in [np.array([.3, 1.1]), np.array([-1, 0]), np.array([np.nan])]:
            with self.assertRaises(ValueError): prep.check_counts(invalid, "counts")


if __name__ == "__main__":
    unittest.main()
