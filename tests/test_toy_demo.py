"""Meaningful checks for the submission demo's train/test boundary."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("toy", ROOT / "examples/run_toy.py")
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


class ToyTests(unittest.TestCase):
    def test_included_data_has_disjoint_participant_roles(self):
        frame = pd.read_csv(ROOT / "examples/toy_data/synthetic_counts.csv")
        toy.validate(frame)
        self.assertEqual(frame.patient.nunique(), 53)
        self.assertEqual(len(frame), 1272)

    def test_participant_role_leakage_is_rejected(self):
        frame = pd.read_csv(ROOT / "examples/toy_data/synthetic_counts.csv")
        frame.loc[0, "role"] = "design"
        with self.assertRaisesRegex(ValueError, "leakage"):
            toy.validate(frame)

    def test_training_scaler_is_not_refit_on_test_batch(self):
        x = np.arange(20, dtype=float).reshape(10, 2)
        y = np.c_[x[:, 0] + 2, x[:, 1] * .5]
        model = toy.ridge_fit(x, y)
        one = toy.ridge_predict(model, x[:1])
        batch = toy.ridge_predict(model, np.r_[x[:1], [[1e8, -1e8]]])
        np.testing.assert_allclose(one, batch[:1])


if __name__ == "__main__":
    unittest.main()
