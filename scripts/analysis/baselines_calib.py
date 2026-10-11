# %% Cell B1 - correction controls x calibration rules, identical predictions, splits and evaluation proteins
# corrections: none (design mean) | common offset (mean sentinel error added to every protein, no learned map)
#              | ProteoGuard shared map | multi map (ridge from all sentinels) | isotype panel (shared map)
# calibration: cell-wise split conformal | HCP (Lee, Barber & Willett 2023) | participant-block (ProteoGuard)
import os, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "combat"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz", "ridge + sentinels": "pred_ridgeanchor_fit{seed}.npz"}
if os.environ.get("STEPH_PRED"): PREDS = {"totalVI": os.environ["STEPH_PRED"], "ridge": "pred_ridge_fit{seed}.npz"}
# Optional explicit mapping lets the CPU submission demo run real ridge outputs
# without manufacturing a file labelled as a trained totalVI prediction.
if os.environ.get("STEPH_PREDICTORS"):
    PREDS = json.loads(os.environ["STEPH_PREDICTORS"])
    if not isinstance(PREDS, dict) or not PREDS or not all(isinstance(v, str) for v in PREDS.values()):
        raise ValueError("STEPH_PREDICTORS must be a nonempty JSON name-to-NPZ mapping")
K, N_CAL, N_REPS, ALPHA, GAMMA, DELTA = 5, int(os.environ.get("STEPH_NCAL", "15")), int(os.environ.get("STEPH_REPS", "50")), 0.10, 0.90, 0.20
EXCLUDE = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69", "CD274", "PD-L1", "CTLA4"]
ISO = re.compile(r"(?i)(isotyp|igg1|igg2|igg_|iggfc|mouse_?ig|rat_?ig|hamster)")

def load(f, seed):
    D = np.load(WORK / f, allow_pickle=True)
    design = pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    y, lo, hi = D["y"].astype(np.float32), D["lo"].astype(np.float32), D["hi"].astype(np.float32)
    prot = D["protein"].astype(str)
    pats, start = np.unique(D["patient"].astype(str), return_index=True)
    o = np.argsort(start); pats, start = pats[o], start[o]
    sl = [slice(a, b) for a, b in zip(start, np.r_[start[1:], len(y)])]
    site = pd.Series(D["site"].astype(str)).groupby(D["patient"].astype(str)).first().reindex(pats).to_numpy()
    mid = (lo + hi) / 2; off = np.stack([np.median(y[s] - mid[s], 0) for s in sl])
    dz = np.where(np.isin(pats, design))[0]; rest = np.where(~np.isin(pats, design) & (site != "Sanger"))[0]
    dcells = np.concatenate([np.arange(len(y))[sl[i]] for i in dz])
    ok = ((y[dcells] > 0).mean(0) >= 0.5) & np.array([not any(m in p.upper() for m in EXCLUDE) and not ISO.search(p)
                                                     and "ISO" not in p.upper() for p in prot])
    return dict(y=y, lo=lo, hi=hi, prot=prot, pats=pats, sl=sl, off=off, dz=dz, rest=rest, ok=ok)

def fit_shared(d, A):
    x = d["off"][np.ix_(d["dz"], A)].mean(1)
    B = np.linalg.lstsq(np.c_[np.ones(len(x)), x], d["off"][d["dz"]], rcond=None)[0]
    return B[0] + np.outer(d["off"][:, A].mean(1), B[1])

def greedy(d):
    off, dz, nP = d["off"], d["dz"], len(d["prot"])
    def loo(A):
        O = np.setdiff1d(np.arange(nP), A); err = base = 0.0
        for i in dz:
            tr = dz[dz != i]; x = off[np.ix_(tr, A)].mean(1)
            B = np.linalg.lstsq(np.c_[np.ones(len(tr)), x], off[tr], rcond=None)[0]
            err += ((off[i, O] - B[0, O] - B[1, O] * off[i, A].mean()) ** 2).sum(); base += ((off[i, O] - off[tr][:, O].mean(0)) ** 2).sum()
        return 1 - err / base
    A = []
    for _ in range(K): A.append(max((c for c in np.where(d["ok"])[0] if c not in A), key=lambda c: loo(A + [c])))
    return A

def multi(d, A, E):
    off, dz = d["off"], d["dz"]; X, Y = off[np.ix_(dz, A)], off[dz]
    def rf(X, Y, lam):
        mx, my = X.mean(0), Y.mean(0); return mx, my, np.linalg.solve((X - mx).T @ (X - mx) + lam * np.eye(X.shape[1]), (X - mx).T @ (Y - my))
    s2 = np.trace(np.atleast_2d(np.cov(X.T, ddof=0))) + 1e-9; best = None
    for lam in s2 * np.array([0.01, 0.1, 1, 10, 100]):
        e = sum((((Y[j] - (lambda m: m[1] + (X[j] - m[0]) @ m[2])(rf(np.delete(X, j, 0), np.delete(Y, j, 0), lam)))[E]) ** 2).sum()
                for j in range(len(dz)))
        if best is None or e < best[0]: best = (e, lam)
    mx, my, W = rf(X, Y, best[1]); return my + (off[:, A] - mx) @ W

def calibrate(rule, S, cal):
    """S[i] = cell scores (cells x proteins) for participant i; returns lambda per protein."""
    n = len(cal)
    if rule == "cell-wise conformal":
        sc = np.concatenate([S[i] for i in cal]); m = len(sc)
        return np.quantile(sc, min(np.ceil((m + 1) * (1 - ALPHA)) / m, 1), axis=0, method="higher")
    if rule == "HCP (Lee et al.)":       # smallest lam with (1/(n+1)) * sum_k F_k(lam) >= 1 - alpha
        sc = np.concatenate([S[i] for i in cal]); w = np.concatenate([np.full(len(S[i]), 1 / ((n + 1) * len(S[i]))) for i in cal])
        o = np.argsort(sc, 0); cw = np.cumsum(w[o], 0); idx = np.argmax(cw >= 1 - ALPHA, 0)
        lam = np.take_along_axis(sc, np.take_along_axis(o, idx[None], 0), 0)[0]
        return np.where(cw[-1] >= 1 - ALPHA, lam, np.inf)
    k = int(np.ceil((n + 1) * (1 - DELTA)))                                       # participant-block (ProteoGuard)
    r = np.stack([np.quantile(S[i], GAMMA, axis=0, method="higher") for i in cal]); return np.sort(r, 0)[k - 1]

rows = []
for seed in FITS:
    data = {nm: load(f.format(seed=seed), seed) for nm, f in PREDS.items() if (WORK / f.format(seed=seed)).exists()}
    if not data:
        raise FileNotFoundError(f"No prediction files for fit {seed} in {WORK}")
    anchors = {nm: greedy(d) for nm, d in data.items() if nm != "ridge + sentinels"}
    if not anchors:
        raise ValueError("A base predictor is required before ridge + sentinels")
    prot = next(iter(data.values()))["prot"]; iso = [j for j, p in enumerate(prot) if ISO.search(p)]
    print(f"[fit {seed}] sentinels:", {nm: [prot[a] for a in A] for nm, A in anchors.items()}, "| isotypes:", [prot[j] for j in iso])
    json.dump({nm: [str(prot[a]) for a in A] for nm, A in anchors.items()}, open(WORK / f"sentinels_fit{seed}.json", "w"))
    E = np.setdiff1d(np.arange(len(prot)), sorted(set(sum(anchors.values(), [])) | set(iso)))   # common evaluation proteins
    for nm, d in data.items():
        assert (d["prot"] == prot).all()
        A = anchors.get(nm, anchors.get("ridge"))
        corr = {"none (design mean)": np.broadcast_to(d["off"][d["dz"]].mean(0), d["off"].shape)}
        if nm != "ridge + sentinels":
            corr["common offset"] = np.repeat(d["off"][:, A].mean(1, keepdims=True), len(prot), 1)
            corr["multi map"] = multi(d, A, E)
            if iso: corr["isotype panel"] = fit_shared(d, iso)
        corr["ProteoGuard shared map"] = fit_shared(d, A)
        W0 = [(d["hi"][s] - d["lo"][s])[:, E] for s in d["sl"]]
        for cname, SH in corr.items():
            S = [np.maximum(d["lo"][s][:, E] + SH[i, E] - d["y"][s][:, E], d["y"][s][:, E] - d["hi"][s][:, E] - SH[i, E])
                 for i, s in enumerate(d["sl"])]
            rng = np.random.default_rng(7)
            for rep in range(N_REPS):
                perm = rng.permutation(d["rest"]); cal, test = perm[:N_CAL], perm[N_CAL:]
                for rule in ["cell-wise conformal", "HCP (Lee et al.)", "participant-block"]:
                    lam = calibrate(rule, S, cal)
                    cov = np.stack([(S[i] <= lam).mean(0) for i in test])
                    wid = np.stack([np.median(W0[i] + 2 * lam, 0) for i in test])
                    isc = np.stack([(W0[i] + 2 * lam + (2 / ALPHA) * np.maximum(S[i] - lam, 0)).mean(0) for i in test])
                    rows.append(dict(fit=seed, predictor=nm, correction=cname, calibration=rule, rep=rep,
                                     coverage=cov.mean(), below_85=(cov < 0.85).mean(),
                                     frac_participants_ge90=(cov >= GAMMA).mean(),
                                     interval_score=isc.mean(), width=np.median(wid)))
            print(f"  {nm} {seed} {cname} done", flush=True)

R = pd.DataFrame(rows); R.to_csv(WORK / "baselines_calibration.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None); pd.set_option("display.max_rows", 200)
print(f"\nTest participants, {len(E)} common evaluation proteins, mean over resamples and both fits.")
print("frac_participants_ge90 = fraction of test participant-protein pairs with >= 90% coverage (target >= 0.81)")
print(R.groupby(["predictor", "correction", "calibration"])[["coverage", "below_85", "frac_participants_ge90", "interval_score", "width"]]
      .mean().round(3).to_string())
