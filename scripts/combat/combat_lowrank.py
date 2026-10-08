# %% Cell K3 - COMBAT: design split + is participant-level error one technical factor? (both data splits)
import os
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import spearmanr, kruskal

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "combat"))
N_DESIGN, MIN_CELLS = 20, 20
for SEED in [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]:
    print(f"\n===== fit {SEED} =====")
    D = np.load(WORK / f"pred_fit{SEED}.npz", allow_pickle=True)
    prot = D["protein"].astype(str); pat = D["patient"].astype(str); ct = D["celltype"].astype(str)
    obs = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad", backed="r").obs
    gb = obs.groupby("patient_id", observed=True); meta = gb.agg(site=("Site", "first"))
    for c, k in [("Status", "status"), ("Pool", "pool")]:
        if c in obs: meta[k] = gb[c].agg(lambda s: s.value_counts().index[0])

    # participant offsets: median within participant x cell type, equal-weight average over common cell types
    df = pd.DataFrame(D["y"] - D["mean"], columns=prot); df["p"] = pat; df["ct"] = ct
    size = df.groupby(["p", "ct"]).size(); med = df.groupby(["p", "ct"]).median()[size >= MIN_CELLS]
    common = (med.reset_index().groupby("ct")["p"].nunique() / len(np.unique(pat)))
    common = common[common >= 0.8].index
    R = med[med.index.get_level_values("ct").isin(common)].groupby(level="p").mean()
    R = R.loc[:, R.std(axis=0) > 1e-6]; P = R.index.to_numpy(); site = meta.reindex(P)["site"].astype(str).to_numpy()
    print(f"held-out participants: {len(P)} | proteins: {R.shape[1]} | cell types: {list(common)}")

    split_file = WORK / f"design_split_fit{SEED}.csv"
    if not split_file.exists():
        rng = np.random.default_rng(2026); design = []
        for st in np.unique(site):
            p = P[site == st]; n = max(1, round(N_DESIGN * len(p) / len(P)))
            design += list(rng.choice(p, min(n, len(p) - 1), replace=False))
        pd.DataFrame({"patient": design}).to_csv(split_file, index=False)
    design = pd.read_csv(split_file)["patient"].astype(str).to_numpy()
    print("design participants:", np.isin(P, design).sum(), "by site:", pd.Series(site[np.isin(P, design)]).value_counts().to_dict())

    # low rank: all held-out participants (unsupervised; no labels used), permutation null
    X = R.values - R.values.mean(0)
    def share(M):
        s = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2; return s / s.sum()
    rng = np.random.default_rng(0); sh = share(X)
    null = np.array([share(np.column_stack([rng.permutation(c) for c in X.T]))[:3] for _ in range(300)])
    V = np.linalg.svd(X, full_matrices=False)[2][0]; pc1 = X @ V
    print("variance explained PC1-3:", np.round(sh[:3], 3), "| permutation 95th pct:", np.round(np.quantile(null, .95, 0), 3))
    print("PC1 same-sign fraction of proteins:", round(float((np.sign(V) == np.sign(V.mean())).mean()), 3))
    depth = pd.Series(np.expm1(D["y"]).sum(1)).groupby(pat).median().reindex(P).to_numpy()
    print("PC1 vs log ADT depth: rho =", round(spearmanr(pc1, np.log10(depth)).statistic, 3))
    m = meta.reindex(P)
    for col in ["site", "pool", "status"]:
        if col in m:
            groups = [pc1[(m[col] == g).to_numpy()] for g in m[col].unique() if (m[col] == g).sum() >= 2]
            print(f"PC1 differs by {col}: Kruskal-Wallis p = {kruskal(*groups).pvalue:.3g} ({len(groups)} groups)")
