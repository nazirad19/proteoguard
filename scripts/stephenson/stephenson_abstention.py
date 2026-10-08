# %% Cell A1 - abstention: does rejecting high-uncertainty cells remove the remaining coverage failures?
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
K_SITE = int(os.environ.get("STEPH_KSITE", "10"))
ALPHA, DELTA = 0.10, 0.20
KEEP = [1.0, 0.9, 0.8]

def tables_from(D, keep):
    """Per-donor tables for the cells in `keep` (cells stay grouped by donor)."""
    idx = np.where(keep)[0]
    y, lo, hi, mu = D["y"][idx], D["lo"][idx], D["hi"][idx], D["mean"][idx]
    S = np.maximum(lo - y, y - hi)
    donors, start = np.unique(D["patient"][idx], return_index=True)
    o = np.argsort(start); donors, start = donors[o], start[o]
    stop = np.r_[start[1:], len(y)]; nD, nP = len(donors), y.shape[1]
    GRID = np.sort(np.vstack([np.quantile(S, np.linspace(0, 1, 1601), axis=0), np.zeros((1, nP))]), axis=0)
    F = np.zeros((nD, GRID.shape[0], nP), np.float32)
    for i, (s0, e0) in enumerate(zip(start, stop)):
        for j in range(nP):
            F[i, :, j] = np.searchsorted(np.sort(S[s0:e0, j]), GRID[:, j], side="right") / (e0 - s0)
    DQ = np.stack([np.quantile(S[s0:e0], 1 - ALPHA, axis=0, method="higher") for s0, e0 in zip(start, stop)])
    MEDW = np.stack([np.median((hi - lo)[s0:e0], 0) for s0, e0 in zip(start, stop)])
    rmsle = np.array([np.sqrt(((mu[s0:e0] - y[s0:e0]) ** 2).mean()) for s0, e0 in zip(start, stop)])
    site = pd.Series(D["site"][idx]).groupby(D["patient"][idx]).first().reindex(donors).to_numpy()
    return dict(F=F, GRID=GRID, DQ=DQ, MEDW=MEDW, rmsle=rmsle, site=site, donors=donors, nP=nP)

def cov_donor_quantile(T, cal, test):
    K = len(cal); k = int(np.ceil((K + 1) * (1 - DELTA))); cols = np.arange(T["nP"])
    lam = np.sort(T["DQ"][cal], 0)[k - 1]
    g = np.array([max(np.searchsorted(T["GRID"][:, j], lam[j], side="right") - 1, 0) for j in cols])
    return T["F"][test][:, g, cols], np.median(T["MEDW"][test] + 2 * lam)

rows = []
for seed in FITS:
    D = np.load(WORK / PRED.format(seed=seed), allow_pickle=True)
    u = D["sd"].mean(1)                                       # per-cell uncertainty: mean posterior-predictive SD
    site_cell = D["site"]
    for keep_frac in KEEP:
        for score in (["none"] if keep_frac == 1.0 else ["posterior_sd", "random"]):
            keep = np.ones(len(u), bool)
            if score != "none":
                s = u if score == "posterior_sd" else np.random.default_rng(seed).random(len(u))
                for st in np.unique(site_cell):              # label-free threshold per site
                    m = site_cell == st
                    keep[m] = s[m] <= np.quantile(s[m], keep_frac)
            T = tables_from(D, keep)
            rng = np.random.default_rng(2)
            for rep in range(300):
                for site in ["Ncl", "Cambridge"]:
                    pool = rng.permutation(np.where(T["site"] == site)[0])
                    cal, test = pool[:K_SITE], pool[K_SITE:]
                    cov, width = cov_donor_quantile(T, cal, test)
                    rows.append(dict(fit=seed, site=site, keep=keep_frac, score=score,
                                     below_85=(cov < 0.85).mean(), coverage=cov.mean(),
                                     width=width, rmsle=T["rmsle"][test].mean()))
    print("fit", seed, "done", flush=True)

A = pd.DataFrame(rows)
A.to_csv(WORK / "abstention_results.csv", index=False)
pd.set_option("display.width", 200)
print(A.groupby(["site", "keep", "score", "fit"])[["below_85", "coverage", "width", "rmsle"]].mean().round(3))
