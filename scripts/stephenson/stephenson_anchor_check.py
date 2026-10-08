# %% Cell D1 - can a few measured "anchor" proteins reveal a new donor's offset on the other proteins?
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")

rows = []
for seed in FITS:
    D = np.load(WORK / PRED.format(seed=seed), allow_pickle=True)
    y, mid = D["y"], (D["lo"] + D["hi"]) / 2
    donors, start = np.unique(D["patient"], return_index=True)
    o = np.argsort(start); donors, start = donors[o], start[o]
    stop = np.r_[start[1:], len(y)]
    site = pd.Series(D["site"]).groupby(D["patient"]).first().reindex(donors).to_numpy()
    off = np.stack([np.median(y[a:b] - mid[a:b], 0) for a, b in zip(start, stop)])   # donor x protein offset
    for st in np.unique(site):                                                          # remove the site-level part
        m = site == st; off[m] -= off[m].mean(0)
    nD, nP = off.shape
    rng = np.random.default_rng(0)
    for n_anchor in [5, 10, 20]:
        for rep in range(50):
            anchors = rng.choice(nP, n_anchor, replace=False); others = np.setdiff1d(np.arange(nP), anchors)
            x = off[:, anchors].mean(1)                                                  # donor's mean anchor offset
            pred = np.zeros((nD, len(others)))
            for i in range(nD):                                                          # leave-one-donor-out fit
                tr = np.arange(nD) != i
                X = np.c_[np.ones(tr.sum()), x[tr]]
                coef, *_ = np.linalg.lstsq(X, off[tr][:, others], rcond=None)
                pred[i] = coef[0] + coef[1] * x[i]
            resid = off[:, others] - pred
            r2 = 1 - (resid ** 2).sum(0) / ((off[:, others] - off[:, others].mean(0)) ** 2).sum(0)
            rows.append(dict(fit=seed, n_anchor=n_anchor, median_r2=np.median(r2),
                             frac_proteins_r2_above_0_3=(r2 > 0.3).mean(),
                             sd_reduction=1 - np.sqrt((resid ** 2).mean()) / off[:, others].std()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Within-site donor offsets explained by the donor's mean offset on a few anchor proteins (leave-one-donor-out):")
print(R.groupby(["n_anchor", "fit"]).mean().drop(columns="rep", errors="ignore").round(3))
