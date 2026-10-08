# %% Cell M3b - risk-coverage for participant-level biomarker calls, incl. RNA + sentinels ridge
# Same sentinels as B1 (sentinels_fit{seed}.json) and one common set of target proteins for every predictor.
import os, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "combat"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz", "ridge + sentinels": "pred_ridgeanchor_fit{seed}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"]}
CELLTYPES = os.environ.get("STEPH_CTS", "cMono,CD4,CD8,NK,B").split(",")
N_ANCHOR, N_CAL, N_REPS, ALPHA = 5, int(os.environ.get("STEPH_NCAL", "15")), int(os.environ.get("STEPH_REPS", "30")), 0.10
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
    S = json.load(open(WORK / f"sentinels_fit{seed}.json")); pl = list(prot)
    A = [pl.index(p) for p in S["totalVI" if name == "totalVI" else "ridge"]]
    iso = [j for j, p in enumerate(prot) if re.search(r"(?i)(isotyp|igg1|igg2|igg_|iggfc|mouse_?ig|rat_?ig|hamster)", p)]
    B = fit_map(np.array(A), dz); shift = B[0] + np.outer(off[:, A].mean(1), B[1])
    drop = set(iso) | {pl.index(p) for v in S.values() for p in v}
    targets = np.array(sorted(set(range(nP)) - drop))                  # common targets for all predictors

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
        T = np.full((len(pats), nP), np.nan); N = T.copy(); P = T.copy()
        for i, s in enumerate(sl):
            c = ct[s] == cell
            if c.sum() < MIN_CELLS: continue
            T[i, use] = (y[s][c][:, use] > cut[use]).mean(0)
            N[i, use] = (mid[s][c][:, use] > cut[use]).mean(0)
            P[i, use] = (mid[s][c][:, use] + shift[i, use] > cut[use]).mean(0)
        valid = ~np.isnan(T[:, use[0]]); pool = rest[valid[rest]]
        if len(pool) < N_CAL + 5 or valid[dz].sum() < 5: continue
        tau = np.median(T[dz][valid[dz]], 0)
        inf_ = use[(T[dz][valid[dz]][:, use] > 0.02).mean(0) > 0.5]
        rng = np.random.default_rng(11)
        for rep in range(N_REPS):
            perm = rng.permutation(pool); cal, test = perm[:N_CAL], perm[N_CAL:]
            k = min(int(np.ceil((len(cal) + 1) * (1 - ALPHA))), len(cal))
            q_pb = np.sort(np.abs(T[cal] - N[cal]), 0)[k - 1][inf_]; q_pg = np.sort(np.abs(T[cal] - P[cal]), 0)[k - 1][inf_]
            t = (T[test][:, inf_] > tau[inf_]).ravel(); n_ = N[test][:, inf_]; p_ = P[test][:, inf_]; tt = tau[inf_]
            scores = {"naive (distance to threshold)": (np.abs(n_ - tt), n_ > tt),
                      "random abstention": (rng.random(n_.shape), n_ > tt),
                      "patient-block, no anchors": (np.abs(n_ - tt) - q_pb, n_ > tt),
                      "anchor-corrected point (no interval)": (np.abs(p_ - tt), p_ > tt),
                      "ProteoGuard": (np.abs(p_ - tt) - q_pg, p_ > tt)}
            for meth, (sc, call) in scores.items():
                rows.append(pd.DataFrame(dict(fit=seed, predictor=name, method=meth, score=sc.ravel(),
                                              wrong=(call.ravel() != t))))
        print(f"  {name} {seed} {cell} done", flush=True)
    return pd.concat(rows) if rows else pd.DataFrame()

out = []
for seed in FITS:
    for name, f in PREDS.items():
        f = f.format(seed=seed)
        if (WORK / f).exists(): out.append(run(seed, name, f))
C = pd.concat(out, ignore_index=True)
RATES = [0.1, 0.2, 0.27, 0.4, 0.6, 0.8, 1.0]
rows = []
for (pred, meth), g in C.groupby(["predictor", "method"]):
    o = np.argsort(-g["score"].to_numpy()); w = g["wrong"].to_numpy()[o]
    for r in RATES:
        n = max(1, int(round(r * len(w)))); rows.append(dict(predictor=pred, method=meth, call_rate=r, selective_error=w[:n].mean()))
    if meth in ("ProteoGuard", "patient-block, no anchors"):            # the method's own calls: margin > 0
        own = g["score"].to_numpy() > 0
        rows.append(dict(predictor=pred, method=meth + " (own calls)", call_rate=round(own.mean(), 3),
                         selective_error=g["wrong"].to_numpy()[own].mean() if own.any() else np.nan))
RC = pd.DataFrame(rows); RC.to_csv(WORK / "risk_coverage_b.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
print(RC[~RC.method.str.contains("own")].pivot_table(index=["predictor", "method"], columns="call_rate", values="selective_error").round(3))
print("\nEach calibrated method's own calls (margin > 0):")
print(RC[RC.method.str.contains("own")].round(3).to_string(index=False))
