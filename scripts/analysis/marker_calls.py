# %% Cell M1 - patient-level biomarker calls: naive point prediction vs ProteoGuard (calibrated, with abstention)
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"]}
# marker -> (protein name token, cell type it is read out in)
MARKERS = {"HLA-DR": ("HLA-DR", "CD14"), "CD38": ("CD38", "CD8"), "PD-1": ("PDCD1", "CD8"),
           "TIGIT": ("TIGIT", "CD8"), "CD69": ("CD69", "CD4"), "CD25": ("IL2RA", "CD4")}
if os.environ.get("STEPH_MARKERS"):                                    # test hook: "name:token:celltype,..."
    MARKERS = {m.split(":")[0]: (m.split(":")[1], m.split(":")[2]) for m in os.environ["STEPH_MARKERS"].split(",")}
N_ANCHOR, N_CAL, N_REPS, ALPHA = 5, 15, 200, 0.10
EXCLUDE = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69", "CD274", "PD-L1", "CTLA4"]
MIN_CELLS = 20

def run(seed, name, f):
    D = np.load(WORK / f, allow_pickle=True)
    design = pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    y, lo, hi = D["y"], D["lo"], D["hi"]; mid = (lo + hi) / 2; prot = D["protein"].astype(str); nP = len(prot)
    ct = D["celltype"].astype(str)
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
    B = fit_map(np.array(A), dz); shift = B[0] + np.outer(off[:, A].mean(1), B[1])     # per patient x protein

    rows = []
    for mk, (tok, cell) in MARKERS.items():
        j = [k for k, p in enumerate(prot) if tok.upper() in p.upper()]
        if not j: print(f"  {mk}: protein not found"); continue
        j = j[0]
        # patient-level truth and predictions: median over that patient's cells of the given type
        true_m, naive_m, pg_m, valid = np.full(len(pats), np.nan), np.full(len(pats), np.nan), np.full(len(pats), np.nan), np.zeros(len(pats), bool)
        for i, s in enumerate(sl):
            c = ct[s] == cell
            if c.sum() < MIN_CELLS: continue
            valid[i] = True
            true_m[i] = np.median(y[s][c, j]); naive_m[i] = np.median(mid[s][c, j]); pg_m[i] = naive_m[i] + shift[i, j]
        tau = np.nanmedian(true_m[dz][valid[dz]])                       # "high" threshold from design patients only
        pool = rest[valid[rest]]
        rng = np.random.default_rng(11)
        for rep in range(N_REPS):
            perm = rng.permutation(pool); cal, test = perm[:N_CAL], perm[N_CAL:]
            k = min(int(np.ceil((len(cal) + 1) * (1 - ALPHA))), len(cal))
            q_pb = np.sort(np.abs(true_m[cal] - naive_m[cal]))[k - 1]  # patient-level conformal, no anchors
            q_pg = np.sort(np.abs(true_m[cal] - pg_m[cal]))[k - 1]     # patient-level conformal, anchor-corrected
            truth = true_m[test] > tau
            for meth, est, q in [("naive point prediction", naive_m, 0.0),
                                 ("patient-block, no anchors", naive_m, q_pb),
                                 ("ProteoGuard", pg_m, q_pg)]:
                e = est[test]
                call = np.where(e - q > tau, 1, np.where(e + q < tau, 0, -1))   # 1 high, 0 low, -1 ambiguous
                conf = call >= 0
                rows.append(dict(fit=seed, predictor=name, marker=mk, method=meth,
                                 wrong_call_rate=((call != truth.astype(int)) & conf).mean(),
                                 ambiguous_rate=(~conf).mean(),
                                 error_among_confident=((call != truth.astype(int)) & conf).sum() / max(conf.sum(), 1),
                                 interval_covers=(np.abs(true_m[test] - e) <= q).mean() if q > 0 else np.nan))
    return pd.DataFrame(rows)

out = []
for seed in FITS:
    for name, f in PREDS.items():
        f = f.format(seed=seed)
        if (WORK / f).exists(): print("running", name, seed, flush=True); out.append(run(seed, name, f))
M = pd.concat(out); M.to_csv(WORK / "marker_calls.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
print("\nAll markers pooled (mean over resamples, fits):")
print(M.groupby(["predictor", "method"])[["wrong_call_rate", "ambiguous_rate", "error_among_confident", "interval_covers"]].mean().round(3))
print("\nPer marker (totalVI):")
print(M[M.predictor == "totalVI"].groupby(["marker", "method"])[["wrong_call_rate", "ambiguous_rate", "error_among_confident"]].mean().round(3))
