"""CPU regression tests for the calibration upgrades.

These check the GUARANTEES, not just that the code runs: the rank rule, the panel-tolerance
statement, the conformal-risk-control floor, and the loss monotonicity CRC requires. Keep them in
CI so a future change to proteoguard_upgrades.py cannot silently break validity.

    python -m unittest discover -s tests -v
"""
import unittest
import numpy as np

import proteoguard_upgrades as pg


def synthetic(n_pat=60, m=200, J=25, seed=0, factor_sd=0.35):
    """Participants exchangeable by construction, with a rank-1 participant factor."""
    rng = np.random.default_rng(seed)
    pats = np.array([f"P{i:03d}" for i in range(n_pat)])
    patient = np.repeat(pats, m)
    b = rng.normal(0, factor_sd, n_pat)
    beta = np.abs(rng.normal(1.0, 0.15, J))
    mean = rng.normal(2.0, 0.5, (n_pat * m, J))
    sd = rng.uniform(0.4, 1.2, J)[None, :] * np.ones((n_pat * m, 1))
    y = mean + beta[None, :] * np.repeat(b, m)[:, None] + rng.normal(0, 1, (n_pat * m, J)) * sd
    lo, hi = mean - 1.645 * sd, mean + 1.645 * sd
    return y.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32), patient, pats


class TestContiguity(unittest.TestCase):
    def test_interleaved_fails(self):
        # A participant appearing in two blocks must fail; comparing the distinct labels that
        # start each run does NOT detect this and passed everything.
        self.assertFalse(pg.check_contiguous(np.array(["A", "A", "B", "B", "A", "A"])))

    def test_contiguous_passes(self):
        self.assertTrue(pg.check_contiguous(np.array(["A", "A", "A", "B", "B", "C"])))

    def test_single_participant(self):
        self.assertTrue(pg.check_contiguous(np.array(["A", "A", "A"])))


class TestBlockRule(unittest.TestCase):
    def test_rank_and_bound(self):
        R = np.sort(np.random.default_rng(1).normal(size=(15, 4)), axis=0)
        lam, k, bound = pg.block_lambda(R, delta=0.20)
        self.assertEqual(k, 13)                       # ceil((15+1)*0.8)
        self.assertAlmostEqual(bound, 13 / 16)
        np.testing.assert_allclose(lam, R[12])        # 13th smallest, 1-based

    def test_infinite_when_rank_exceeds_n(self):
        # n_g = 3 at delta = 0.2 needs k = 4 > 3: the quantile is undefined and must be +inf,
        # never silently capped. This is why Mondrian needs n_g >= 4.
        lam, k, _ = pg.block_lambda(np.zeros((3, 2)), delta=0.20)
        self.assertEqual(k, 4)
        self.assertTrue(np.isinf(lam).all())

    def test_minimum_group_size_is_four(self):
        ok = [n for n in range(1, 30) if int(np.ceil((n + 1) * 0.8)) <= n]
        self.assertEqual(min(ok), 4)

    def test_empirical_coverage_meets_bound(self):
        y, lo, hi, patient, pats = synthetic(seed=2)
        _, R = pg.participant_score_matrix(y, lo, hi, patient, pats=pats)
        rng = np.random.default_rng(3)
        hits = []
        for _ in range(200):
            perm = rng.permutation(len(pats))
            lam = pg.block_lambda(R[perm[:15]], delta=0.20)[0]
            hits.append((R[perm[15:]] <= lam[None, :]).mean())
        self.assertGreaterEqual(float(np.mean(hits)), 0.75)   # bound 0.8125, slack for 200 draws


class TestPanelTolerance(unittest.TestCase):
    def test_simultaneous_beats_per_protein_on_whole_panel(self):
        y, lo, hi, patient, pats = synthetic(J=30, seed=5)
        _, R = pg.participant_score_matrix(y, lo, hi, patient, pats=pats)
        design, cal, test = R[:20], R[20:35], R[35:]
        c = pg.block_lambda(design, 0.20)[0]
        per_protein = pg.block_lambda(cal, 0.20)[0]
        lam_panel, _, _, _ = pg.panel_lambda(cal, c, 0.20, tau=0.0)
        whole_pp = (test <= per_protein[None, :]).all(axis=1).mean()
        whole_panel = pg.panel_attainment(test, lam_panel, tau=0.0).mean()
        self.assertGreater(whole_panel, whole_pp)

    def test_tau_monotone_in_width(self):
        y, lo, hi, patient, pats = synthetic(J=30, seed=6)
        _, R = pg.participant_score_matrix(y, lo, hi, patient, pats=pats)
        c = pg.block_lambda(R[:20], 0.20)[0]
        w = [np.median(pg.panel_lambda(R[20:35], c, 0.20, tau=t)[0]) for t in (0.0, 0.05, 0.20)]
        self.assertTrue(w[0] >= w[1] >= w[2])          # allowing more failures buys narrower


class TestRiskControl(unittest.TestCase):
    def test_floor_is_one_over_n_plus_one(self):
        # Even with ZERO observed loss the bound is 1/(n+1); no smaller alpha is certifiable.
        grid = np.linspace(0, 1, 11)
        zero = np.zeros((15, len(grid)))
        self.assertIsNone(pg.crc_threshold(grid, zero, alpha=0.05)[0])
        self.assertIsNotNone(pg.crc_threshold(grid, zero, alpha=1 / 16 + 1e-9)[0])

    def test_loss_must_be_monotone(self):
        # wrong-calls-per-decision is non-increasing in the margin; error-among-calls is not,
        # which is why the former is the controlled quantity.
        rng = np.random.default_rng(7)
        n, J = 12, 40
        part = np.repeat([f"P{i}" for i in range(n)], J)
        pred = rng.normal(0.5, 0.2, n * J)
        true = pred + rng.normal(0, 0.15, n * J)
        thr = np.full(n * J, 0.5)
        grid = np.linspace(0, 0.3, 16)
        _, loss, rate, _ = pg.decision_losses(pred, true, thr, part, grid)
        self.assertTrue((np.diff(loss, axis=1) <= 1e-12).all())
        self.assertTrue((np.diff(rate, axis=1) <= 1e-12).all())


class TestDiagnostic(unittest.TestCase):
    def test_normalization_diagnostic_runs(self):
        y, lo, hi, patient, pats = synthetic(J=10, seed=8)
        mean = (lo + hi) / 2
        d = pg.normalization_diagnostic(y, mean, lo, hi, patient, pats)
        self.assertEqual(len(d["rho_per_protein"]), 10)
        self.assertTrue(-1.0 <= d["median_rho"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
