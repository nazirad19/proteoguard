# %% Cell E1 - anchor-panel calibration: a few measured proteins on the new donor re-centre the rest
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
K_SITE = int(os.environ.get("STEPH_KSITE", "10"))
N_REPS = int(os.environ.get("STEPH_REPS", "200"))
ANCHORS = [int(a) for a in os.environ.get("STEPH_ANCHORS", "5,10").split(",")]
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
    off = np.stack([np.median(y[s] - mid[s], 0) for s in sl])          # donor x protein offset (needs labels)
    nP = y.shape[1]

    def fit_predict(train, x_train, x_new, cols):
        X = np.c_[np.ones(len(train)), x_train]
        coef, *_ = np.linalg.lstsq(X, off[np.ix_(train, cols)], rcond=None)
        return coef[0] + coef[1] * x_new                                 # predicted offset for the other proteins

    def donor_score(i, shift, cols):                                     # half-width donor i needs for 90% coverage
        s = sl[i]
        return np.quantile(np.maximum(lo[s][:, cols] + shift - y[s][:, cols], y[s][:, cols] - hi[s][:, cols] - shift),
                           1 - ALPHA, axis=0, method="higher")

    def coverage(i, shift, lam, cols):
        s = sl[i]
        return ((y[s][:, cols] >= lo[s][:, cols] + shift - lam) & (y[s][:, cols] <= hi[s][:, cols] + shift + lam)).mean(0)

    rng = np.random.default_rng(3)
    for n_anchor in ANCHORS:
        for rep in range(N_REPS):
            anchors = rng.choice(nP, n_anchor, replace=False); cols = np.setdiff1d(np.arange(nP), anchors)
            x = off[:, anchors].mean(1)                                   # each donor's measured anchor offset
            for st in ["Ncl", "Cambridge"]:
                pool = rng.permutation(np.where(site == st)[0]); cal, test = pool[:K_SITE], pool[K_SITE:]
                K = len(cal); k = int(np.ceil((K + 1) * (1 - DELTA)))
                width0 = np.stack([(hi - lo)[sl[t]][:, cols].mean(0) for t in test])
                # (a) no anchors: same-site symmetric donor-level calibration on the same proteins
                lam0 = np.sort(np.stack([donor_score(i, 0.0, cols) for i in cal]), 0)[k - 1]
                cov0 = np.stack([coverage(t, 0.0, lam0, cols) for t in test])
                # (b) anchors: shift each donor by its predicted offset; calibration donors use leave-one-out fits
                sh_cal = [fit_predict(np.setdiff1d(cal, [i]), x[np.setdiff1d(cal, [i])], x[i], cols) for i in cal]
                lam1 = np.sort(np.stack([donor_score(i, sh, cols) for i, sh in zip(cal, sh_cal)]), 0)[k - 1]
                cov1 = np.stack([coverage(t, fit_predict(cal, x[cal], x[t], cols), lam1, cols) for t in test])
                for name, cov, lam in [("no anchors", cov0, lam0), (f"{n_anchor} anchors", cov1, lam1)]:
                    rows.append(dict(fit=seed, site=st, n_anchor=n_anchor, method=name,
                                     below_85=(cov < 0.85).mean(), coverage=cov.mean(),
                                     mean_width=(width0 + 2 * lam).mean()))
    print("fit", seed, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(WORK / "anchor_results.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
print(R.groupby(["site", "n_anchor", "method", "fit"])[["below_85", "coverage", "mean_width"]].mean().round(3))
