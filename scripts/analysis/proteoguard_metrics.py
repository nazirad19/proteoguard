# %% Cell P2 - full calibration metrics per test patient + patient-level bootstrap CIs
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"]}
N_ANCHOR, N_CAL, N_REPS, ALPHA, DELTA = 5, int(os.environ.get("STEPH_NCAL", "15")), int(os.environ.get("STEPH_REPS", "100")), 0.10, 0.20
EXCLUDE = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69", "CD274", "PD-L1", "CTLA4"]

def run(seed, name, f):
    D = np.load(WORK / f, allow_pickle=True)
    design = pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    y, lo, hi = D["y"].astype(np.float32), D["lo"].astype(np.float32), D["hi"].astype(np.float32)
    mid = (lo + hi) / 2; prot = D["protein"].astype(str); nP = len(prot)
    pats, start = np.unique(D["patient"].astype(str), return_index=True)
    o = np.argsort(start); pats, start = pats[o], start[o]
    stop = np.r_[start[1:], len(y)]; sl = [slice(a, b) for a, b in zip(start, stop)]
    site = pd.Series(D["site"].astype(str)).groupby(D["patient"].astype(str)).first().reindex(pats).to_numpy()
    off = np.stack([np.median(y[s] - mid[s], 0) for s in sl])
    dz = np.where(np.isin(pats, design))[0]; rest = np.where(~np.isin(pats, design) & (site != "Sanger"))[0]
    dcells = np.concatenate([np.arange(len(y))[sl[i]] for i in dz])
    ok = ((y[dcells] > 0).mean(0) >= 0.5) & np.array([not any(m in p.upper() for m in EXCLUDE) and "ISO" not in p.upper()
                                                     and "IGG" not in p.upper() for p in prot])
    def fit_map(A, idx):
        x = off[np.ix_(idx, A)].mean(1); return np.linalg.lstsq(np.c_[np.ones(len(idx)), x], off[idx], rcond=None)[0]
    def loo_r2(A):
        O = np.setdiff1d(np.arange(nP), A); err = base = 0.0
        for i in dz:
            tr = dz[dz != i]; B = fit_map(A, tr); pred = B[0, O] + B[1, O] * off[i, A].mean()
            err += ((off[i, O] - pred) ** 2).sum(); base += ((off[i, O] - off[tr][:, O].mean(0)) ** 2).sum()
        return 1 - err / base
    cand = np.where(ok)[0]; A = []
    for _ in range(N_ANCHOR):
        A.append(max((c for c in cand if c not in A), key=lambda c: loo_r2(np.array(A + [c]))))
    A = np.array(A); O = np.setdiff1d(np.arange(nP), A)
    B = fit_map(A, dz); SH = B[0] + np.outer(off[:, A].mean(1), B[1]); Z = np.zeros_like(SH)

    def pscore(i, sh):
        s = sl[i]; return np.quantile(np.maximum(lo[s][:, O] + sh[O] - y[s][:, O], y[s][:, O] - hi[s][:, O] - sh[O]), 1 - ALPHA, axis=0, method="higher")
    def metrics(i, sh, lam):
        s = sl[i]; L = lo[s][:, O] + sh[O] - lam; H = hi[s][:, O] + sh[O] + lam; Y = y[s][:, O]
        cov = ((Y >= L) & (Y <= H)).mean(0)
        isc = ((H - L) + (2 / ALPHA) * (np.maximum(L - Y, 0) + np.maximum(Y - H, 0))).mean(0)
        return dict(abs_cov_error=np.abs(cov - 0.9).mean(), below_85=(cov < 0.85).mean(), above_95=(cov > 0.95).mean(),
                    interval_score=isc.mean(), median_width=np.median(H - L), coverage=cov.mean())
    rows = []; rng = np.random.default_rng(7)
    for rep in range(N_REPS):
        perm = rng.permutation(rest); cal, test = perm[:N_CAL], perm[N_CAL:]
        k = int(np.ceil((len(cal) + 1) * (1 - DELTA)))
        sc = np.concatenate([np.maximum(lo[sl[i]][:, O] - y[sl[i]][:, O], y[sl[i]][:, O] - hi[sl[i]][:, O]) for i in cal]); n = len(sc)
        lams = {"cell-wise conformal": (Z, np.quantile(sc, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1), axis=0, method="higher")),
                "patient-block, no anchors": (Z, np.sort(np.stack([pscore(i, Z[i]) for i in cal]), 0)[k - 1]),
                "ProteoGuard": (SH, np.sort(np.stack([pscore(i, SH[i]) for i in cal]), 0)[k - 1])}
        for meth, (sh, lam) in lams.items():
            for t in test:
                rows.append(dict(fit=seed, predictor=name, method=meth, rep=rep, patient=pats[t], **metrics(t, sh[t], lam)))
    return pd.DataFrame(rows)

R = pd.concat([run(s, n, f.format(seed=s)) for s in FITS for n, f in PREDS.items() if (WORK / f.format(seed=s)).exists()])
R.to_csv(WORK / "proteoguard_patient_metrics.csv", index=False)
cols = ["coverage", "abs_cov_error", "below_85", "above_95", "interval_score", "median_width"]
pm = R.groupby(["fit", "predictor", "method", "patient"])[cols].mean().reset_index()   # one row per test patient

def boot(g, B=2000, rng=np.random.default_rng(0)):          # patient-level bootstrap of the mean
    v = g.to_numpy(); idx = rng.integers(0, len(v), (B, len(v))); m = v[idx].mean(1)
    return f"{v.mean():.3f} [{np.quantile(m, .025):.3f}, {np.quantile(m, .975):.3f}]"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print("Mean over test patients [95% patient-bootstrap CI]:")
print(pm.groupby(["predictor", "fit", "method"])[cols].agg(boot).to_string())

# paired differences vs standard calibration (same patients)
w = pm.pivot_table(index=["fit", "predictor", "patient"], columns="method", values=["below_85", "interval_score", "median_width"])
print("\nProteoGuard minus cell-wise conformal, paired by patient [95% bootstrap CI]:")
for (fit, pred), g in w.groupby(level=["fit", "predictor"]):
    out = {m: boot(g[(m, "ProteoGuard")] - g[(m, "cell-wise conformal")]) for m in ["below_85", "interval_score", "median_width"]}
    print(fit, pred, out)
