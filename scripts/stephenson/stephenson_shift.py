# %% Cell C1 - re-centre intervals by the site-level offset learned from calibration donors, then calibrate
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
K_SITE = int(os.environ.get("STEPH_KSITE", "10"))
N_REPS = int(os.environ.get("STEPH_REPS", "300"))
ALPHA, DELTA = 0.10, 0.20

rows = []
for seed in FITS:
    D = np.load(WORK / PRED.format(seed=seed), allow_pickle=True)
    y, lo, hi = D["y"], D["lo"], D["hi"]
    mid = (lo + hi) / 2
    donors, start = np.unique(D["patient"], return_index=True)
    o = np.argsort(start); donors, start = donors[o], start[o]
    stop = np.r_[start[1:], len(y)]
    site = pd.Series(D["site"]).groupby(D["patient"]).first().reindex(donors).to_numpy()
    sl = [slice(a, b) for a, b in zip(start, stop)]
    off = np.stack([np.median(y[s] - mid[s], 0) for s in sl])          # per-donor signed offset (donor x protein)
    W0 = np.stack([(hi - lo)[s].mean(0) for s in sl])

    def donor_q(i, b):                                                 # donor i's 90% score after shifting by b
        s = sl[i]; return np.quantile(np.maximum(lo[s] + b - y[s], y[s] - hi[s] - b), 1 - ALPHA, axis=0, method="higher")

    rng = np.random.default_rng(2)
    for rep in range(N_REPS):
        for st in ["Ncl", "Cambridge"]:
            pool = rng.permutation(np.where(site == st)[0]); cal, test = pool[:K_SITE], pool[K_SITE:]
            k = int(np.ceil((len(cal) + 1) * (1 - DELTA)))
            for name in ["symmetric", "recentred"]:
                b = np.median(off[cal], 0) if name == "recentred" else np.zeros(y.shape[1])
                lam = np.sort(np.stack([donor_q(i, b) for i in cal]), 0)[k - 1]
                cov = np.stack([((y[sl[t]] >= lo[sl[t]] + b - lam) & (y[sl[t]] <= hi[sl[t]] + b + lam)).mean(0) for t in test])
                rows.append(dict(fit=seed, site=st, method=name, below_85=(cov < 0.85).mean(),
                                 coverage=cov.mean(), mean_width=(W0[test] + 2 * lam).mean()))
    print("fit", seed, "done", flush=True)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
R = pd.DataFrame(rows); R.to_csv(WORK / "recentred_results.csv", index=False)
print(R.groupby(["site", "method", "fit"])[["below_85", "coverage", "mean_width"]].mean().round(3))
