# %% Cell M2 - systematic biomarker calls: every protein x major cell type, GMM-defined positivity
import os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"]}
CELLTYPES = os.environ.get("STEPH_CTS", "CD14,CD4,CD8,NK_16hi,B_cell").split(",")
N_ANCHOR, N_CAL, N_REPS, ALPHA = 5, 15, int(os.environ.get("STEPH_REPS", "100")), 0.10
MIN_CELLS = int(os.environ.get("STEPH_MINCELLS", "20"))
EXCLUDE = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69", "CD274", "PD-L1", "CTLA4"]

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
    B = fit_map(np.array(A), dz); shift = B[0] + np.outer(off[:, A].mean(1), B[1])
    targets = np.setdiff1d(np.arange(nP), A)                            # every non-anchor protein

    # positivity boundary per protein: 2-component GMM on measured log1p ADT of design cells
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(dcells, min(20000, len(dcells)), replace=False)
    cut = np.full(nP, np.nan)
    for j in targets:
        v = y[sub, j].reshape(-1, 1)
        if v.std() < 1e-3: continue
        g = GaussianMixture(2, random_state=0).fit(v)
        m = np.sort(g.means_.ravel())
        if m[1] - m[0] < 0.5: continue                                  # not bimodal enough -> no positivity call
        grid = np.linspace(m[0], m[1], 200).reshape(-1, 1)
        hi_comp = np.argmax(g.means_.ravel())
        cut[j] = grid[np.argmax(g.predict_proba(grid)[:, hi_comp] > 0.5), 0]
    use = targets[np.isfinite(cut[targets])]
    print(f"[{name} {seed}] proteins with a bimodal positivity boundary: {len(use)} of {len(targets)}")

    rows = []
    for cell in CELLTYPES:
        # patient-level fraction positive: truth (measured), naive (prediction), ProteoGuard (shift-corrected)
        T = np.full((len(pats), nP), np.nan); N = T.copy(); P = T.copy()
        for i, s in enumerate(sl):
            c = ct[s] == cell
            if c.sum() < MIN_CELLS: continue
            T[i, use] = (y[s][c][:, use] > cut[use]).mean(0)
            N[i, use] = (mid[s][c][:, use] > cut[use]).mean(0)
            P[i, use] = (mid[s][c][:, use] + shift[i, use] > cut[use]).mean(0)
        valid = ~np.isnan(T[:, use[0]]) if len(use) else np.zeros(len(pats), bool)
        pool = rest[valid[rest]]
        if len(pool) < N_CAL + 5 or valid[dz].sum() < 5: continue
        tau = np.median(T[dz][valid[dz]], 0)                            # "high" = above design patients' median fraction
        informative = use[(T[dz][valid[dz]][:, use] > 0.02).mean(0) > 0.5]   # skip proteins ~never positive in this cell type
        rng = np.random.default_rng(11)
        for rep in range(N_REPS):
            perm = rng.permutation(pool); cal, test = perm[:N_CAL], perm[N_CAL:]
            k = min(int(np.ceil((len(cal) + 1) * (1 - ALPHA))), len(cal))
            for meth, E in [("naive point prediction", N), ("patient-block, no anchors", N), ("ProteoGuard", P)]:
                q = np.zeros(nP) if meth.startswith("naive") else np.sort(np.abs(T[cal] - E[cal]), 0)[k - 1]
                e, t = E[test][:, informative], T[test][:, informative] > tau[informative]
                qq, tt = q[informative], tau[informative]
                call = np.where(e - qq > tt, 1, np.where(e + qq < tt, 0, -1)); conf = call >= 0
                wrong = (call != t.astype(int)) & conf
                rows.append(dict(fit=seed, predictor=name, celltype=cell, method=meth, n_proteins=len(informative),
                                 wrong_call_rate=wrong.mean(), ambiguous_rate=(~conf).mean(),
                                 error_among_confident=wrong.sum() / max(conf.sum(), 1)))
    return pd.DataFrame(rows)

out = []
for seed in FITS:
    for name, f in PREDS.items():
        f = f.format(seed=seed)
        if (WORK / f).exists(): out.append(run(seed, name, f))
M = pd.concat(out); M.to_csv(WORK / "marker_calls_all.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
print("\nAll proteins x cell types (mean over resamples and fits):")
print(M.groupby(["predictor", "method"])[["wrong_call_rate", "ambiguous_rate", "error_among_confident"]].mean().round(3))
print("\nBy cell type (totalVI):")
print(M[M.predictor == "totalVI"].groupby(["celltype", "method"])[["n_proteins", "wrong_call_rate", "ambiguous_rate", "error_among_confident"]].mean().round(3))
