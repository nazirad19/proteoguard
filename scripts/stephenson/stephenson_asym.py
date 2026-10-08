# %% Cell B1 - asymmetric donor-level calibration (separate lower/upper widening, same site)
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
K_SITE = int(os.environ.get("STEPH_KSITE", "10"))
N_REPS = int(os.environ.get("STEPH_REPS", "300"))
ALPHA, DELTA = 0.10, 0.20

rows, side_rows = [], []
for seed in FITS:
    D = np.load(WORK / PRED.format(seed=seed), allow_pickle=True)
    y, lo, hi = D["y"], D["lo"], D["hi"]
    SL, SH = lo - y, y - hi                                   # >0: below the interval / above the interval
    donors, start = np.unique(D["patient"], return_index=True)
    o = np.argsort(start); donors, start = donors[o], start[o]
    stop = np.r_[start[1:], len(y)]
    site = pd.Series(D["site"]).groupby(D["patient"]).first().reindex(donors).to_numpy()
    sl = [slice(a, b) for a, b in zip(start, stop)]
    # per-donor quantiles of each score
    q_sym = np.stack([np.quantile(np.maximum(SL[s], SH[s]), 1 - ALPHA, axis=0, method="higher") for s in sl])
    q_lo = np.stack([np.quantile(SL[s], 1 - ALPHA / 2, axis=0, method="higher") for s in sl])
    q_hi = np.stack([np.quantile(SH[s], 1 - ALPHA / 2, axis=0, method="higher") for s in sl])
    W0 = np.stack([(hi - lo)[s].mean(0) for s in sl])

    # which side do raw misses fall on? (per site)
    for st in ["Ncl", "Cambridge", "Sanger"]:
        idx = np.concatenate([np.arange(len(y))[sl[i]] for i in np.where(site == st)[0]])
        side_rows.append(dict(fit=seed, site=st, miss_below=(SL[idx] > 0).mean(), miss_above=(SH[idx] > 0).mean()))

    def cover(test, lam_lo, lam_hi):
        cov = np.stack([((y[sl[t]] >= lo[sl[t]] - lam_lo) & (y[sl[t]] <= hi[sl[t]] + lam_hi)).mean(0) for t in test])
        return cov, (W0[test] + lam_lo + lam_hi).mean()

    rng = np.random.default_rng(2)
    for rep in range(N_REPS):
        for st in ["Ncl", "Cambridge"]:
            pool = rng.permutation(np.where(site == st)[0]); cal, test = pool[:K_SITE], pool[K_SITE:]
            K = len(cal)
            k1 = int(np.ceil((K + 1) * (1 - DELTA)))             # symmetric: one score
            k2 = int(np.ceil((K + 1) * (1 - DELTA / 2)))         # asymmetric: two scores, union bound
            lam = np.sort(q_sym[cal], 0)[k1 - 1]
            for name, (a, b) in {"symmetric": (lam, lam),
                                 "asymmetric": (np.sort(q_lo[cal], 0)[min(k2, K) - 1],
                                                np.sort(q_hi[cal], 0)[min(k2, K) - 1])}.items():
                cov, width = cover(test, a, b)
                rows.append(dict(fit=seed, site=st, method=f"donor_quantile | same_site | {name}",
                                 below_85=(cov < 0.85).mean(), coverage=cov.mean(), mean_width=width))
    print("fit", seed, "done", flush=True)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
print("Raw totalVI misses by side (fraction of cell x protein values):")
print(pd.DataFrame(side_rows).groupby(["site", "fit"]).mean().round(3))
R = pd.DataFrame(rows); R.to_csv(WORK / "asym_results.csv", index=False)
print("\nSame-site donor-level calibration, symmetric vs asymmetric:")
print(R.groupby(["site", "method", "fit"])[["below_85", "coverage", "mean_width"]].mean().round(3))
