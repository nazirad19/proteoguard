# %% Cell AB - anchor budget: 0, 1, 3, 5, 10, 20 measured sentinel proteins
# For each budget: offset recovery, participant-level calibration, biomarker calls, overall and by protein family.
# Two maps from anchors to the rest: "shared" (one global factor, = ProteoGuard) and "multi" (ridge on all anchors,
# can learn protein-specific structure). Evaluation always uses the same proteins: those never chosen as an anchor.
import os, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"]}
CELLTYPES = os.environ.get("STEPH_CTS", "CD14,CD4,CD8,NK_16hi,B_cell").split(",")
BUDGETS = [0, 1, 3, 5, 10, 20]
N_CAL, N_REPS, ALPHA, DELTA = int(os.environ.get("STEPH_NCAL", "15")), int(os.environ.get("STEPH_REPS", "30")), 0.10, 0.20
MIN_CELLS = 20
EXCLUDE = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69", "CD274", "PD-L1", "CTLA4"]
FAMILIES = {
    "activation": ["CD38", "HLA-DR", "CD69", "CD25", "IL2RA", "CD71", "CD137", "TNFRSF9", "CD154", "CD40LG", "CD134", "OX40",
                   "ICOS", "CD278", "CD86", "CD80", "CD83"],
    "exhaustion/inhibitory": ["TIGIT", "PD-1", "PDCD1", "CD279", "LAG3", "LAG-3", "CD223", "TIM-3", "HAVCR2", "CD366",
                              "CTLA4", "CD152", "CD39", "ENTPD1", "CD244", "KLRG1", "CD160", "CD274", "PD-L1"],
    "lineage": ["CD3", "CD4", "CD8", "CD8A", "CD8B", "CD19", "CD20", "MS4A1", "CD14", "CD16", "FCGR3A", "CD56", "NCAM1",
                "CD11C", "ITGAX", "CD123", "IL3RA", "CD1C", "CD141", "CD33", "CD11B", "CD7", "CD2", "CD5", "TCRAB", "TCRVD2"],
    "trafficking": ["CCR7", "CD197", "CD62L", "SELL", "CXCR3", "CD183", "CXCR4", "CD184", "CXCR5", "CD185", "CCR4", "CD194",
                    "CCR6", "CD196", "CCR5", "CD195", "CCR2", "CD192", "CD49D", "ITGA4", "ITGB7", "CLA", "CD103", "ITGAE",
                    "CX3CR1", "CD62P"]}

def tokens(p):
    core = re.sub(r"_A\d+$", "", re.sub(r"^(AB_)?(ADT_)?", "", p.upper()))
    return {t for t in re.split(r"[()/ ]", core) if t} | {core}
def family_of(p):
    t = tokens(p)
    for fam, al in FAMILIES.items():
        if t & set(al): return fam
    return "other"

def run(seed, name, f):
    D = np.load(WORK / f, allow_pickle=True)
    design = pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    y, lo, hi = D["y"].astype(np.float32), D["lo"].astype(np.float32), D["hi"].astype(np.float32)
    mid = (lo + hi) / 2; prot = D["protein"].astype(str); nP = len(prot); ct = D["celltype"].astype(str)
    pats, start = np.unique(D["patient"].astype(str), return_index=True)
    o = np.argsort(start); pats, start = pats[o], start[o]
    stop = np.r_[start[1:], len(y)]; sl = [slice(a, b) for a, b in zip(start, stop)]
    site = pd.Series(D["site"].astype(str)).groupby(D["patient"].astype(str)).first().reindex(pats).to_numpy()
    off = np.stack([np.median(y[s] - mid[s], 0) for s in sl])
    dz = np.where(np.isin(pats, design))[0]; rest = np.where(~np.isin(pats, design) & (site != "Sanger"))[0]
    dcells = np.concatenate([np.arange(len(y))[sl[i]] for i in dz])
    ok = ((y[dcells] > 0).mean(0) >= 0.5) & np.array([not any(m in p.upper() for m in EXCLUDE) and "ISO" not in p.upper()
                                                     and "IGG" not in p.upper() for p in prot])

    # ---- greedy anchor order (same criterion as ProteoGuard), extended to the largest budget
    def fit_shared(A, idx):
        x = off[np.ix_(idx, A)].mean(1); return np.linalg.lstsq(np.c_[np.ones(len(idx)), x], off[idx], rcond=None)[0]
    def loo_r2(A):
        O = np.setdiff1d(np.arange(nP), A); err = base = 0.0
        for i in dz:
            tr = dz[dz != i]; B = fit_shared(A, tr); pred = B[0, O] + B[1, O] * off[i, A].mean()
            err += ((off[i, O] - pred) ** 2).sum(); base += ((off[i, O] - off[tr][:, O].mean(0)) ** 2).sum()
        return 1 - err / base
    cand = list(np.where(ok)[0]); order = []
    for _ in range(min(max(BUDGETS), len(cand))):
        order.append(max((c for c in cand if c not in order), key=lambda c: loo_r2(np.array(order + [c]))))
    E = np.setdiff1d(np.arange(nP), order)                                  # fixed evaluation proteins
    fam = np.array([family_of(p) for p in prot])
    print(f"[{name} {seed}] anchor order: {[prot[a] for a in order]}")
    print(f"  evaluation proteins: {len(E)} | by family:", pd.Series(fam[E]).value_counts().to_dict(), flush=True)

    # ---- maps from K anchors to every protein's participant offset (fit on design participants only)
    def ridge_fit(X, Y, lam):
        mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc); return mx, my, W
    def shift_for(K, kind):
        if K == 0: return np.broadcast_to(off[dz].mean(0), off.shape).copy()   # no anchors: design-mean offset only
        A = np.array(order[:K])
        if kind == "shared" or K == 1:
            B = fit_shared(A, dz); return B[0] + np.outer(off[:, A].mean(1), B[1])
        X, Y = off[np.ix_(dz, A)], off[dz]; s2 = np.trace(np.cov(X.T, ddof=0).reshape(K, K)) + 1e-9
        best = None
        for lam in s2 * np.array([0.01, 0.1, 1, 10, 100]):              # leave-one-design-participant-out choice
            e = 0.0
            for j in range(len(dz)):
                m = np.arange(len(dz)) != j; mx, my, W = ridge_fit(X[m], Y[m], lam)
                e += ((Y[j, E] - (my + (X[j] - mx) @ W)[E]) ** 2).sum()
            if best is None or e < best[0]: best = (e, lam)
        mx, my, W = ridge_fit(X, Y, best[1]); return my + (off[:, A] - mx) @ W

    # ---- positivity boundaries (as in cell M2) and per-participant cell-type indices
    rng0 = np.random.default_rng(0); sub = rng0.choice(dcells, min(20000, len(dcells)), replace=False)
    cut = np.full(nP, np.nan)
    for j in E:
        v = y[sub, j].reshape(-1, 1)
        if v.std() < 1e-3: continue
        g = GaussianMixture(2, random_state=0).fit(v); m = np.sort(g.means_.ravel())
        if m[1] - m[0] < 0.5: continue
        grid = np.linspace(m[0], m[1], 200).reshape(-1, 1)
        cut[j] = grid[np.argmax(g.predict_proba(grid)[:, np.argmax(g.means_.ravel())] > 0.5), 0]
    use = E[np.isfinite(cut[E])]
    cidx = {(i, c): np.arange(sl[i].start, sl[i].stop)[ct[sl[i]] == c] for i in range(len(pats)) for c in CELLTYPES}
    T = {c: np.full((len(pats), nP), np.nan) for c in CELLTYPES}
    for c in CELLTYPES:
        for i in range(len(pats)):
            ix = cidx[(i, c)]
            if len(ix) >= MIN_CELLS: T[c][i, use] = (y[ix][:, use] > cut[use]).mean(0)

    rows, prow = [], []
    for kind in ["shared", "multi"]:
        for K in BUDGETS:
            if kind == "multi" and K <= 1: continue                       # identical to shared at K = 0, 1
            SH = shift_for(K, kind)
            # 1) offset recovery on non-design participants: R2 per protein vs. the design mean
            ev = rest; base = off[dz].mean(0)
            r2p = 1 - ((off[ev] - SH[ev]) ** 2).sum(0) / ((off[ev] - base) ** 2).sum(0)
            mae = np.abs(np.stack([y[sl[i]].mean(0) - (mid[sl[i]] + SH[i]).mean(0) for i in ev])).mean(0)
            # 2) participant-block calibration
            sc = np.stack([np.quantile(np.maximum(lo[sl[i]][:, E] + SH[i, E] - y[sl[i]][:, E],
                                                  y[sl[i]][:, E] - hi[sl[i]][:, E] - SH[i, E]), 1 - ALPHA, axis=0, method="higher")
                           for i in range(len(pats))])
            # 3) biomarker calls (fraction positive per participant x cell type)
            P = {c: np.full((len(pats), nP), np.nan) for c in CELLTYPES}
            for c in CELLTYPES:
                for i in range(len(pats)):
                    ix = cidx[(i, c)]
                    if len(ix) >= MIN_CELLS: P[c][i, use] = (mid[ix][:, use] + SH[i, use] > cut[use]).mean(0)
            rng = np.random.default_rng(7)
            for rep in range(N_REPS):
                perm = rng.permutation(rest); cal, test = perm[:N_CAL], perm[N_CAL:]
                k = int(np.ceil((len(cal) + 1) * (1 - DELTA))); lam = np.sort(sc[cal], 0)[k - 1]
                cov = np.zeros((len(test), len(E))); isc = cov.copy(); wid = []
                for t_, i in enumerate(test):
                    s = sl[i]; L = lo[s][:, E] + SH[i, E] - lam; H = hi[s][:, E] + SH[i, E] + lam; Y = y[s][:, E]
                    cov[t_] = ((Y >= L) & (Y <= H)).mean(0)
                    isc[t_] = ((H - L) + (2 / ALPHA) * (np.maximum(L - Y, 0) + np.maximum(Y - H, 0))).mean(0)
                    wid.append(np.median(H - L, 0))
                wid = np.stack(wid)
                ncall = np.zeros(nP); nwrong = np.zeros(nP); ntot = np.zeros(nP)
                kq = min(int(np.ceil((len(cal) + 1) * (1 - ALPHA))), len(cal))
                for c in CELLTYPES:
                    Tc, Pc = T[c], P[c]; vd = ~np.isnan(Tc[dz][:, use[0]]) if len(use) else np.zeros(len(dz), bool)
                    vc, vt = cal[~np.isnan(Tc[cal][:, use[0]])], test[~np.isnan(Tc[test][:, use[0]])]
                    if vd.sum() < 5 or len(vc) < 5 or len(vt) < 1: continue
                    tau = np.median(Tc[dz][vd], 0); inf_ = use[(Tc[dz][vd][:, use] > 0.02).mean(0) > 0.5]
                    q = np.sort(np.abs(Tc[vc] - Pc[vc]), 0)[min(kq, len(vc)) - 1]
                    e, tr = Pc[vt][:, inf_], Tc[vt][:, inf_] > tau[inf_]
                    call = np.where(e - q[inf_] > tau[inf_], 1, np.where(e + q[inf_] < tau[inf_], 0, -1)); conf = call >= 0
                    ncall[inf_] += conf.sum(0); nwrong[inf_] += ((call != tr.astype(int)) & conf).sum(0); ntot[inf_] += conf.shape[0]
                for j_, j in enumerate(E):
                    prow.append(dict(fit=seed, predictor=name, map=kind, K=K, rep=rep, protein=prot[j], family=fam[j],
                                     offset_r2=r2p[j], pseudobulk_mae=mae[j], below_85=(cov[:, j_] < 0.85).mean(),
                                     interval_score=isc[:, j_].mean(), width=wid[:, j_].mean(),
                                     n_calls=ncall[j], n_wrong=nwrong[j], n_decisions=ntot[j]))
            print(f"  {name} {seed} {kind} K={K} done", flush=True)
    return pd.DataFrame(prow)

R = pd.concat([run(s, n, f.format(seed=s)) for s in FITS for n, f in PREDS.items() if (WORK / f.format(seed=s)).exists()])
R.to_csv(WORK / "anchor_budget.csv", index=False)

def summ(g):
    return pd.Series(dict(offset_r2=g.groupby("protein").offset_r2.first().median(),
                          pseudobulk_mae=g.pseudobulk_mae.mean(), below_85=g.below_85.mean(),
                          interval_score=g.interval_score.mean(), width=g.width.mean(),
                          call_rate=g.n_calls.sum() / max(g.n_decisions.sum(), 1),
                          error_among_calls=g.n_wrong.sum() / max(g.n_calls.sum(), 1)))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print("\nOverall (evaluation proteins; mean over fits; offset_r2 = median over proteins):")
print(R.groupby(["predictor", "map", "K"]).apply(lambda g: g.groupby("fit").apply(summ).mean()).round(3).to_string())
print("\nBy protein family (totalVI):")
F = R[(R.predictor == "totalVI") & (R.family != "other")]
print(F.groupby(["family", "map", "K"]).apply(lambda g: g.groupby("fit").apply(summ).mean())[
    ["offset_r2", "below_85", "call_rate", "error_among_calls"]].round(3).to_string())
