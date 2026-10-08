# %% Cell S1 - scaled intervals: widen each cell's interval in proportion to totalVI's own uncertainty
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
K_SITE = int(os.environ.get("STEPH_KSITE", "10"))
ALPHA, DELTA = 0.10, 0.20

def tables(D, scale):
    """scale = per cell x protein divisor of the score (ones = additive widening)."""
    y, lo, hi = D["y"], D["lo"], D["hi"]
    S = np.maximum(lo - y, y - hi) / scale
    donors, start = np.unique(D["patient"], return_index=True)
    o = np.argsort(start); donors, start = donors[o], start[o]
    stop = np.r_[start[1:], len(y)]; ncell = stop - start; nD, nP = len(donors), y.shape[1]
    GRID = np.sort(np.vstack([np.quantile(S, np.linspace(0, 1, 1601), axis=0), np.zeros((1, nP))]), axis=0)
    F = np.zeros((nD, GRID.shape[0], nP), np.float32)
    for i, (s0, e0) in enumerate(zip(start, stop)):
        for j in range(nP):
            F[i, :, j] = np.searchsorted(np.sort(S[s0:e0, j]), GRID[:, j], side="right") / (e0 - s0)
    DQ = np.stack([np.quantile(S[s0:e0], 1 - ALPHA, axis=0, method="higher") for s0, e0 in zip(start, stop)])
    W0 = np.stack([(hi - lo)[s0:e0].mean(0) for s0, e0 in zip(start, stop)])        # mean raw width
    SC = np.stack([scale[s0:e0].mean(0) for s0, e0 in zip(start, stop)])            # mean scale
    site = pd.Series(D["site"]).groupby(D["patient"]).first().reindex(donors).to_numpy()
    return dict(F=F, GRID=GRID, DQ=DQ, W0=W0, SC=SC, ncell=ncell, site=site, nD=nD, nP=nP)

def lam_for(T, method, cal):
    F, GRID, nP = T["F"], T["GRID"], T["nP"]; cols = np.arange(nP)
    if method == "cellwise":
        n = T["ncell"][cal]; pooled = (F[cal] * n[:, None, None]).sum(0) / n.sum()
        level = min(np.ceil((n.sum() + 1) * (1 - ALPHA)) / n.sum(), 1.0)
        return GRID[np.argmax(pooled >= level, 0), cols]
    K = len(cal); k = int(np.ceil((K + 1) * (1 - DELTA)))
    return np.sort(T["DQ"][cal], 0)[k - 1]

def evaluate(T, lam, test):
    cols = np.arange(T["nP"])
    g = np.array([max(np.searchsorted(T["GRID"][:, j], lam[j], side="right") - 1, 0) for j in cols])
    cov = T["F"][test][:, g, cols]
    width = (T["W0"][test] + 2 * lam * T["SC"][test]).mean()      # mean interval width (log1p counts)
    return cov, width

rows = []
for seed in FITS:
    D = np.load(WORK / PRED.format(seed=seed), allow_pickle=True)
    sd = D["sd"].astype(np.float64)
    floor = np.quantile(sd, 0.10, axis=0, keepdims=True) + 1e-3     # avoid dividing by ~0 for flat cells
    variants = {"additive": np.ones_like(sd), "scaled": np.maximum(sd, floor)}
    for vname, scale in variants.items():
        T = tables(D, scale); rng = np.random.default_rng(2)
        for rep in range(300):
            for site in ["Ncl", "Cambridge"]:
                pool = rng.permutation(np.where(T["site"] == site)[0])
                cal_site, test = pool[:K_SITE], pool[K_SITE:]
                cal_pooled = rng.choice(np.setdiff1d(np.arange(T["nD"]), test), K_SITE, replace=False)
                for meth, scheme, cal in [("cellwise", "pooled", cal_pooled), ("donor_quantile", "same_site", cal_site)]:
                    cov, width = evaluate(T, lam_for(T, meth, cal), test)
                    rows.append(dict(fit=seed, site=site, intervals=vname, method=f"{meth} | {scheme}",
                                     below_85=(cov < 0.85).mean(), coverage=cov.mean(), mean_width=width))
    print("fit", seed, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(WORK / "scaled_results.csv", index=False)
pd.set_option("display.width", 200)
print(R.groupby(["site", "method", "intervals", "fit"])[["below_85", "coverage", "mean_width"]].mean().round(3))
