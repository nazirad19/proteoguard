"""Simulation: cell-wise vs donor-block calibration when cells are clustered within donors.

Model: residual r_ij = b_i + e_ij, donor effect b_i ~ N(0, tau^2), cell noise e_ij ~ N(0, 1).
Target: 90% intervals [pred - lam, pred + lam] for cells of a NEW donor.

Methods
  raw        : lam = 1.645 (model ignores donor effects, like a per-cell predictive interval)
  cellwise   : split conformal pooling all calibration cells as if independent
  donor_crc  : conformal risk control over donors; loss = a donor's miscoverage fraction.
               lam = smallest value with (sum_i L_i(lam) + 1) / (K + 1) <= alpha
               -> E[miscoverage of a new donor] <= alpha when donors are exchangeable.
  donor_quantile : each calibration donor's score = the half-width it needs for 90% of its own cells;
               lam = conformal quantile of those K scores at level 1 - DELTA
               -> P(a new donor gets >= 90% coverage) >= 1 - DELTA (needs K >= 1/DELTA - 1).
Reported per setting (over reps): mean coverage of the new donor, and the probability that the new
donor's coverage is below 85%.
"""
import numpy as np, pandas as pd
from scipy.stats import norm

ALPHA, DELTA, REPS = 0.10, 0.20, 1000
GRID = np.linspace(0, 8, 801)

def new_donor_cov(lam, tau, rng):
    b = rng.normal(0, tau)
    return norm.cdf(lam - b) - norm.cdf(-lam - b)

def run(K, n, tau, rng):
    out = {"raw": [], "cellwise": [], "donor_crc": [], "donor_quantile": []}; width = {k: [] for k in out}
    for _ in range(REPS):
        r = np.abs(rng.normal(0, tau, (K, 1)) + rng.normal(0, 1, (K, n)))
        flat = np.sort(r.ravel()); N = flat.size
        lam_cell = flat[min(int(np.ceil((N + 1) * (1 - ALPHA))), N) - 1]
        rs = np.sort(r, axis=1)
        L = 1 - np.stack([np.searchsorted(rs[i], GRID, side="right") for i in range(K)]) / n   # K x grid
        ok = (L.sum(0) + 1) / (K + 1) <= ALPHA
        lam_crc = GRID[np.argmax(ok)] if ok.any() else np.inf
        d = np.sort(rs[:, int(np.ceil(0.9 * n)) - 1]); k = int(np.ceil((K + 1) * (1 - DELTA)))
        lam_dq = d[k - 1] if k <= K else np.inf
        for name, lam in [("raw", 1.645), ("cellwise", lam_cell), ("donor_crc", lam_crc), ("donor_quantile", lam_dq)]:
            out[name].append(new_donor_cov(lam, tau, rng) if np.isfinite(lam) else 1.0)
            width[name].append(2 * lam)
    return out, width

rng = np.random.default_rng(0)
rows = []
for tau in [0.0, 0.5, 1.0]:
    for K in [3, 5, 10, 30]:
        for n in [50, 500, 5000]:
            res, wid = run(K, n, tau, rng)
            for name, cov in res.items():
                cov = np.asarray(cov)
                rows.append(dict(tau=tau, K_donors=K, cells_per_donor=n, method=name,
                                 mean_cov=cov.mean(), p_donor_below_90=(cov < 0.90).mean(),
                                 p_donor_below_85=(cov < 0.85).mean(), median_width=np.median(wid[name])))
df = pd.DataFrame(rows)
df.to_csv("sim_donor_blocks.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot_table(index=["tau", "K_donors", "cells_per_donor"], columns="method",
                     values=["mean_cov", "p_donor_below_85", "median_width"]).round(3).to_string())
