# %% Cell B2 - "RNA + sentinels" ridge: the same 5 measured sentinels given directly as inputs (cell values + participant medians)
# Run after B1 (needs sentinels_fit{seed}.json). Then re-run B1: it picks up pred_ridgeanchor_fit{seed}.npz automatically.
import os, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "combat"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
a = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad")
a = a[a.obs["adt_total"].to_numpy() > 0].copy()
pat = a.obs["patient_id"].astype(str).to_numpy()
p_site = a.obs.groupby("patient_id", observed=True)["Site"].first().astype(str)
b = ad.AnnData(a.layers["counts"].copy() if "counts" in a.layers else a.X.copy())
sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
Z = TruncatedSVD(50, random_state=0).fit_transform(b.X)
Y = np.log1p(a.obsm["protein_counts"].to_numpy(np.float32)); prot = a.obsm["protein_counts"].columns.astype(str).to_numpy()

for SEED in FITS:
    train_p, held_p = train_test_split(p_site.index.to_numpy(), test_size=0.5, random_state=SEED, stratify=p_site.to_numpy())
    train = np.isin(pat, train_p); held = np.where(~train)[0]; held = held[np.argsort(pat[held], kind="stable")]
    A = [list(prot).index(p) for p in json.load(open(WORK / f"sentinels_fit{SEED}.json"))["ridge"]]
    cellS = Y[:, A]                                                                  # measured sentinels, per cell
    pmed = pd.DataFrame(cellS).groupby(pat).median().reindex(pat).to_numpy()        # measured sentinels, participant median
    X = np.c_[Z, cellS, pmed]; mu, sd = X[train].mean(0), X[train].std(0) + 1e-6; X = (X - mu) / sd
    model = Ridge(alpha=1.0).fit(X[train], Y[train]); pred = model.predict(X).astype(np.float32)
    q = np.quantile(Y[train] - pred[train], [0.05, 0.95], axis=0)
    np.savez_compressed(WORK / f"pred_ridgeanchor_fit{SEED}.npz", y=Y[held], mean=pred[held], lo=pred[held] + q[0], hi=pred[held] + q[1],
                        sd=np.zeros_like(pred[held]), patient=pat[held], site=a.obs["Site"].astype(str).to_numpy()[held],
                        celltype=a.obs["initial_clustering"].astype(str).to_numpy()[held], protein=prot)
    O = np.setdiff1d(np.arange(len(prot)), A)
    r0 = np.load(WORK / f"pred_ridge_fit{SEED}.npz")
    print(f"fit {SEED}: held-out RMSE on non-sentinel proteins | RNA-only ridge:",
          round(float(np.sqrt(((r0['mean'][:, O] - Y[held][:, O]) ** 2).mean(0)).mean()), 4),
          "| RNA + sentinels ridge:", round(float(np.sqrt(((pred[held][:, O] - Y[held][:, O]) ** 2).mean(0)).mean()), 4))
