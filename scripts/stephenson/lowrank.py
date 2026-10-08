# %% Cell L1 - patient-level residuals (cell-type balanced) and a frozen design split
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
SEED = int(os.environ.get("STEPH_FIT", "2026"))                 # which fit's predictions to analyse
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz").format(seed=SEED)
META_H5AD = WORK / "haniffa21.processed.h5ad"                      # original file, for patient metadata
N_DESIGN = int(os.environ.get("STEPH_NDESIGN", "20"))
MIN_CELLS = int(os.environ.get("STEPH_MINCELLS", "20"))
DECISION_MARKERS = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69"]

D = np.load(WORK / PRED, allow_pickle=True)
resid = (D["y"] - D["mean"]).astype(np.float64)                   # log1p observed - log1p likelihood mean (py_norm_)
prot = D["protein"].astype(str); pat = D["patient"].astype(str); ct = D["celltype"].astype(str)
site_of = pd.Series(D["site"].astype(str)).groupby(pat).first()

# patient x cell type medians, then equal-weight average over cell types common to most patients
df = pd.DataFrame(resid, columns=prot); df["patient"] = pat; df["ct"] = ct
size = df.groupby(["patient", "ct"]).size()
med = df.groupby(["patient", "ct"]).median()[size >= MIN_CELLS]
present = med.reset_index().groupby("ct")["patient"].nunique() / len(np.unique(pat))
common = present[present >= 0.8].index
R = med[med.index.get_level_values("ct").isin(common)].groupby(level="patient").mean()   # patient x protein
R = R.loc[:, R.std(axis=0) > 1e-6]                                                             # drop constant proteins
patients = R.index.to_numpy(); site = site_of.reindex(patients).to_numpy()
print(f"patients: {len(patients)} | proteins: {R.shape[1]} | cell types used: {list(common)}")

# frozen design split (stratified by site); saved so later cells reuse exactly the same patients
split_file = WORK / f"design_split_fit{SEED}.csv"
if split_file.exists():
    design = pd.read_csv(split_file)["patient"].astype(str).to_numpy()
else:
    rng = np.random.default_rng(2026); design = []
    for st in np.unique(site):
        p = patients[site == st]; n = max(1, round(N_DESIGN * len(p) / len(patients)))
        design += list(rng.choice(p, min(n, len(p) - 1), replace=False))
    design = np.array(design); pd.DataFrame({"patient": design}).to_csv(split_file, index=False)
is_design = np.isin(patients, design)
print("design patients:", is_design.sum(), "| held-out patients for reconstruction:", (~is_design).sum())

anchor_ok = np.array([not any(m in p.upper() for m in DECISION_MARKERS) and "ISOTYPE" not in p.upper()
                      and "IGG" not in p.upper() for p in R.columns])
print("anchor candidates:", anchor_ok.sum(), "| excluded (decision markers / isotypes):", (~anchor_ok).sum())

# %% Cell L2 - is the patient-level error low-rank? (design patients only)
Xd = R.values[is_design]; mu = Xd.mean(0); sdv = Xd.std(0) + 1e-9

def eig_share(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return s / s.sum()

rng = np.random.default_rng(0); rows = []
for kind, X in [("covariance", Xd - mu), ("correlation", (Xd - mu) / sdv)]:
    share = eig_share(X)
    null = np.array([eig_share(np.column_stack([rng.permutation(c) for c in X.T]))[:5] for _ in range(500)])
    thr = np.quantile(null, 0.95, axis=0)
    for k in range(5):
        rows.append(dict(pca=kind, component=k + 1, var_explained=share[k], null_95pct=thr[k],
                         above_chance=share[k] > thr[k]))
print("Variance explained vs parallel analysis (500 column permutations):")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# leave-one-patient-out stability of PC1/PC2 loadings (covariance PCA)
V = np.linalg.svd(Xd - mu, full_matrices=False)[2]
stab = []
for i in range(len(Xd)):
    Xi = np.delete(Xd, i, 0); Vi = np.linalg.svd(Xi - Xi.mean(0), full_matrices=False)[2]
    stab.append([abs(V[0] @ Vi[0]), abs(V[1] @ Vi[1])])
stab = np.array(stab)
print(f"\nLOO loading stability |cos|: PC1 median {np.median(stab[:,0]):.3f} (min {stab[:,0].min():.3f}); "
      f"PC2 median {np.median(stab[:,1]):.3f} (min {stab[:,1].min():.3f})")
top = pd.Series(V[0], index=R.columns).sort_values()
print("PC1 sign: mean loading", round(float(V[0].mean()), 3), "| fraction same sign:", round(float((np.sign(V[0]) == np.sign(V[0].mean())).mean()), 3))
print("PC1 lowest loadings:", top.head(5).round(3).to_dict()); print("PC1 highest loadings:", top.tail(5).round(3).to_dict())

# %% Cell L3 - what does the factor track? (all held-out patients; scores from design loadings)
scores = (R.values - mu) @ V[:2].T
meta = pd.DataFrame(index=patients); meta["site"] = site; meta["PC1"] = scores[:, 0]; meta["PC2"] = scores[:, 1]
if META_H5AD.exists():
    import anndata as ad
    obs = ad.read_h5ad(META_H5AD, backed="r").obs
    g = obs.groupby("patient_id", observed=True)
    sev_map = {"Healthy": 0, "Asymptomatic": 1, "Mild": 2, "Moderate": 3, "Severe": 4, "Critical": 5}
    meta["log_rna_depth"] = np.log10(g["total_counts"].median().reindex(patients).astype(float))
    meta["severity"] = g["Status_on_day_collection_summary"].agg(lambda s: s.astype(str).map(sev_map).max()).reindex(patients)
    meta["sex"] = (g["Sex"].first().astype(str).reindex(patients) == "Male").astype(float)
    age = g["Age_interval"].first().astype(str).reindex(patients)
    meta["age_mid"] = age.str.extract(r"(\d+)\D+(\d+)").astype(float).mean(1)
sub = WORK / "stephenson_sub1000.h5ad"
if sub.exists():                                                   # ADT depth from the subsample actually modelled
    import anndata as ad
    o = ad.read_h5ad(sub, backed="r").obs
    meta["log_adt_depth"] = np.log10(o.groupby("patient_id", observed=True)["adt_total"].median().reindex(patients).astype(float))
print("\nSpearman correlation of factor scores with patient covariates:")
num = meta.drop(columns="site").apply(pd.to_numeric, errors="coerce")
print(num.corr(method="spearman")[["PC1", "PC2"]].drop(["PC1", "PC2"]).round(3))
print("\nPC1 by site:", meta.groupby("site")["PC1"].agg(["mean", "std", "size"]).round(3).to_dict("index"))

if "severity" in meta and meta["severity"].notna().sum() > 10:      # severity effect after site + ADT depth
    m = meta.dropna(subset=["severity", "log_adt_depth"])
    Xs = pd.get_dummies(m["site"], drop_first=True).astype(float); Xs["depth"] = m["log_adt_depth"]
    def coef(sev):
        X = np.c_[np.ones(len(m)), Xs.values, sev]; return np.linalg.lstsq(X, m["PC1"].values, rcond=None)[0][-1]
    obs_b = coef(m["severity"].values); rng = np.random.default_rng(1); null = []
    for _ in range(2000):                                         # permute severity within site
        perm = m["severity"].copy()
        for st in m["site"].unique():
            idx = m["site"] == st; perm[idx] = rng.permutation(perm[idx].values)
        null.append(coef(perm.values))
    print(f"\nPC1 ~ severity | site + ADT depth: coefficient {obs_b:.3f}, permutation p = {(np.abs(null) >= abs(obs_b)).mean():.3f}"
          f" (n = {len(m)} patients)")

# %% Cell L4 - reconstruct unseen patients' errors from a few anchors (loadings/maps learned on design only)
held = ~is_design; Xh = R.values[held]; P = R.shape[1]
cand = np.where(anchor_ok)[0]

def predict(method, A, Xtr, Xte):
    O = np.setdiff1d(np.arange(P), A); m0 = Xtr.mean(0)
    if method == "shared_offset":                                  # one offset: mean anchor residual, scaled per protein
        x_tr, x_te = Xtr[:, A].mean(1), Xte[:, A].mean(1)
        B = np.linalg.lstsq(np.c_[np.ones(len(x_tr)), x_tr], Xtr[:, O], rcond=None)[0]
        return O, B[0] + np.outer(x_te, B[1])
    if method.startswith("rank"):                                  # low-rank: infer factor scores from anchors
        k = int(method[-1]); Vk = np.linalg.svd(Xtr - m0, full_matrices=False)[2][:k]
        VA = Vk[:, A]; lam_f = 0.1 * (VA ** 2).sum() / k                 # light ridge: few anchors -> unstable scores
        Z = np.linalg.solve(VA @ VA.T + lam_f * np.eye(k), VA @ (Xte[:, A] - m0[A]).T).T
        return O, m0[O] + Z @ Vk[:, O]
    XA = Xtr[:, A] - Xtr[:, A].mean(0)                             # full ridge map W
    lam = 1.0 * np.trace(XA.T @ XA) / len(A)                       # penalty scaled to the anchors' variance
    W = np.linalg.solve(XA.T @ XA + lam * np.eye(len(A)), XA.T @ (Xtr[:, O] - m0[O]))
    return O, m0[O] + (Xte[:, A] - Xtr[:, A].mean(0)) @ W

def r2(Y, Yhat, base):
    return 1 - ((Y - Yhat) ** 2).sum() / ((Y - base) ** 2).sum()

rng = np.random.default_rng(5); rows = []
for n_a in [1, 3, 5, 10]:
    for rep in range(100):
        A = rng.choice(cand, n_a, replace=False)
        for meth in ["shared_offset", "rank1", "rank2", "ridge_map"]:
            if meth == "rank2" and n_a < 2: continue
            O, Yhat = predict(meth, A, Xd, Xh)
            rows.append(dict(n_anchor=n_a, method=meth, heldout_R2=r2(Xh[:, O], Yhat, Xd.mean(0)[O]),
                             sd_reduction=1 - np.sqrt(((Xh[:, O] - Yhat) ** 2).mean()) / np.sqrt(((Xh[:, O] - Xd.mean(0)[O]) ** 2).mean())))
L4 = pd.DataFrame(rows); L4.to_csv(WORK / f"lowrank_reconstruction_fit{SEED}.csv", index=False)
print("\nHeld-out patients: errors on non-anchor proteins predicted from random anchors (loadings learned on design patients)")
print(L4.groupby(["n_anchor", "method"])[["heldout_R2", "sd_reduction"]].agg(["median", lambda x: np.quantile(x, 0.1)]).round(3))
