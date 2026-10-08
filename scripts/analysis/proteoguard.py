# %% Cell P1 - ProteoGuard: anchor selection + patient-shift correction + patient-block calibration + abstention
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
SEED = int(os.environ.get("STEPH_FIT", "2026"))
PREDS = {"totalVI": f"pred_fit{SEED}.npz", "ridge": f"pred_ridge_fit{SEED}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"]}
N_ANCHOR = 5
N_CAL = int(os.environ.get("STEPH_NCAL", "15"))                     # calibration patients per resample (pooled sites)
N_REPS = int(os.environ.get("STEPH_REPS", "200"))
ALPHA, DELTA = 0.10, 0.20
DECISION_MARKERS = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69"]
CANONICAL = ["CD3", "CD4", "CD8", "CD14", "CD19"]
design = pd.read_csv(WORK / f"design_split_fit{SEED}.csv")["patient"].astype(str).to_numpy()

def run(pred_file, name):
    D = np.load(WORK / pred_file, allow_pickle=True)
    y, lo, hi = D["y"].astype(np.float32), D["lo"].astype(np.float32), D["hi"].astype(np.float32)
    mid = (lo + hi) / 2; prot = D["protein"].astype(str); nP = len(prot)
    pats, start = np.unique(D["patient"].astype(str), return_index=True)
    o = np.argsort(start); pats, start = pats[o], start[o]
    stop = np.r_[start[1:], len(y)]; sl = [slice(a, b) for a, b in zip(start, stop)]
    site = pd.Series(D["site"].astype(str)).groupby(D["patient"].astype(str)).first().reindex(pats).to_numpy()
    off = np.stack([np.median(y[s] - mid[s], 0) for s in sl])          # patient x protein shift (needs labels)
    is_d = np.isin(pats, design); dz = np.where(is_d)[0]
    rest = np.where(~is_d & (site != "Sanger"))[0]; external = np.where(~is_d & (site == "Sanger"))[0]
    dcells = np.concatenate([np.arange(len(y))[sl[i]] for i in np.where(np.isin(pats, design))[0]])
    detected = (y[dcells] > 0).mean(0) >= 0.5                          # measured in at least half of design cells
    ok = detected & np.array([not any(m in p.upper() for m in DECISION_MARKERS) and "ISO" not in p.upper()
                              and "IGG" not in p.upper() for p in prot])

    # ---------- design patients only: map anchors -> shifts, and choose anchors ----------
    def fit_map(A, idx):                                               # shift_other = a + b * mean(anchor shifts)
        x = off[np.ix_(idx, A)].mean(1)
        B = np.linalg.lstsq(np.c_[np.ones(len(idx)), x], off[idx], rcond=None)[0]
        return B                                                       # (2, nP)
    def loo_r2(A):
        O = np.setdiff1d(np.arange(nP), A); err = base = 0.0
        for i in dz:
            tr = dz[dz != i]; B = fit_map(A, tr)
            pred = B[0, O] + B[1, O] * off[i, A].mean()
            err += ((off[i, O] - pred) ** 2).sum(); base += ((off[i, O] - off[tr][:, O].mean(0)) ** 2).sum()
        return 1 - err / base
    cand = np.where(ok)[0]; chosen = []
    for _ in range(N_ANCHOR):                                          # greedy forward selection
        best = max((c for c in cand if c not in chosen), key=lambda c: loo_r2(np.array(chosen + [c])))
        chosen.append(best)
    dmean, dvar = y[np.concatenate([np.arange(len(y))[sl[i]] for i in dz])].mean(0), off[dz].var(0)
    canon = [j for c in CANONICAL for j in np.where([p.upper().replace("AB_", "") == c for p in prot])[0]][:N_ANCHOR]
    panels = {"optimised": np.array(chosen),
              "abundant": cand[np.argsort(-dmean[cand])][:N_ANCHOR],
              "variable": cand[np.argsort(-dvar[cand])][:N_ANCHOR]}
    if len(canon) == N_ANCHOR: panels["canonical"] = np.array(canon)
    v1 = np.linalg.svd(off[dz] - off[dz].mean(0), full_matrices=False)[2][0]
    panels["top_PC1_loading"] = cand[np.argsort(-np.abs(v1[cand]))][:N_ANCHOR]
    print(f"[{name}] optimised anchors: {[str(p) for p in prot[chosen]]} | design LOO R2 {loo_r2(np.array(chosen)):.3f}")

    # ---------- calibration + test patients ----------
    def shifts(A):
        B = fit_map(A, dz); x = off[:, A].mean(1)
        return B[0] + np.outer(x, B[1])                                 # predicted shift for every patient (nPat, nP)
    def anchor_misfit(A):                                              # patient disagreement with the shared-shift model
        B = fit_map(A, dz); x = off[:, A].mean(1)
        return np.sqrt(((off[:, A] - (B[0, A] + np.outer(x, B[1, A]))) ** 2).mean(1))
    def pscore(i, sh, O):                                              # half-width patient i needs for 90% coverage
        s = sl[i]; return np.quantile(np.maximum(lo[s][:, O] + sh[O] - y[s][:, O], y[s][:, O] - hi[s][:, O] - sh[O]),
                                      1 - ALPHA, axis=0, method="higher")
    def pcov(i, sh, lam, O):
        s = sl[i]; return ((y[s][:, O] >= lo[s][:, O] + sh[O] - lam) & (y[s][:, O] <= hi[s][:, O] + sh[O] + lam)).mean(0)
    def width(test, lam, O):
        return np.mean([(hi[sl[t]][:, O] - lo[sl[t]][:, O]).mean() for t in test]) + 2 * lam.mean()

    SH = {k: shifts(A) for k, A in panels.items()}
    MIS = anchor_misfit(panels["optimised"])
    zero = np.zeros((len(pats), nP), np.float32)
    rng = np.random.default_rng(7); rows, prow = [], []
    for rep in range(N_REPS):
        perm = rng.permutation(rest); cal, test = perm[:N_CAL], perm[N_CAL:]
        k = int(np.ceil((len(cal) + 1) * (1 - DELTA)))
        A_rand = rng.choice(cand, N_ANCHOR, replace=False); SH_rand = shifts(A_rand)
        O_opt = np.setdiff1d(np.arange(nP), panels["optimised"])
        methods = {}
        # standard: cell-wise conformal, pooled cells, no shift
        sc = np.concatenate([np.maximum(lo[sl[i]][:, O_opt] - y[sl[i]][:, O_opt], y[sl[i]][:, O_opt] - hi[sl[i]][:, O_opt]) for i in cal])
        n = len(sc); methods["cellwise"] = (zero, np.quantile(sc, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1), axis=0, method="higher"), O_opt)
        # patient-block, no anchors
        methods["patient_block"] = (zero, np.sort(np.stack([pscore(i, zero[i], O_opt) for i in cal]), 0)[k - 1], O_opt)
        # ProteoGuard with each anchor panel
        for pname, A in list(panels.items()) + [("random", A_rand)]:
            sh = SH_rand if pname == "random" else SH[pname]; O = np.setdiff1d(np.arange(nP), A)
            lam = np.sort(np.stack([pscore(i, sh[i], O) for i in cal]), 0)[k - 1]
            methods[f"ProteoGuard_{pname}"] = (sh, lam, O)
        for mname, (sh, lam, O) in methods.items():
            for evalset, tt in [("test (Cambridge+Ncl)", test), ("external (Sanger)", external)]:
                if len(tt) == 0: continue
                cov = np.stack([pcov(t, sh[t], lam, O) for t in tt])
                rows.append(dict(predictor=name, rep=rep, eval=evalset, method=mname, below_85=(cov < 0.85).mean(),
                                 coverage=cov.mean(), mean_width=width(tt, lam, O)))
            cov = np.stack([pcov(t, sh[t], lam, O) for t in test])
            if mname in ("cellwise", "patient_block", "ProteoGuard_optimised"):
                for t, c in zip(test, cov):
                    prow.append(dict(predictor=name, rep=rep, method=mname, patient=pats[t], site=site[t],
                                     patient_coverage=c.mean(), misfit=MIS[t]))
    return pd.DataFrame(rows), pd.DataFrame(prow), {k: list(prot[v]) for k, v in panels.items()}

allR, allP = [], []
for name, f in PREDS.items():
    if not (WORK / f).exists(): print("missing", f, "- skipped"); continue
    Rm, Pm, pan = run(f, name); allR.append(Rm); allP.append(Pm)
R = pd.concat(allR); Pt = pd.concat(allP)
R.to_csv(WORK / f"proteoguard_fit{SEED}.csv", index=False); Pt.to_csv(WORK / f"proteoguard_patients_fit{SEED}.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
print("\nResults (averaged over resamples):")
print(R.groupby(["eval", "predictor", "method"])[["below_85", "coverage", "mean_width"]].mean().round(3))

# abstention: flag test patients whose anchors disagree most with the shared-shift model
g = Pt[Pt["method"] == "ProteoGuard_optimised"].copy()
thr = g.groupby(["predictor", "rep"])["misfit"].transform(lambda m: m.quantile(0.8))
g["flagged"] = g["misfit"] > thr
print("\nAbstention (top 20% anchor misfit flagged): patient coverage")
print(g.groupby(["predictor", "flagged"])["patient_coverage"].agg(["mean", lambda c: (c < 0.85).mean(), "size"]).round(3))
