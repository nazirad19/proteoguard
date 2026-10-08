# %% Cell R1 - ridge baseline predictor (RNA PCA -> log1p ADT), same donor split as the totalVI fit
import os
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
SEED = int(os.environ.get("STEPH_FIT", "2026"))
a = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad")
a = a[a.obs["adt_total"].to_numpy() > 0].copy()
pat = a.obs["patient_id"].astype(str).to_numpy()
p_site = a.obs.groupby("patient_id", observed=True)["Site"].first().astype(str)
train_p, held_p = train_test_split(p_site.index.to_numpy(), test_size=0.5, random_state=SEED, stratify=p_site.to_numpy())
train = np.isin(pat, train_p); held = np.where(~train)[0]; held = held[np.argsort(pat[held], kind="stable")]

b = ad.AnnData(a.layers["counts"].copy() if "counts" in a.layers else a.X.copy())
sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
Z = TruncatedSVD(50, random_state=0).fit_transform(b.X)               # RNA only (no protein used)
Y = np.log1p(a.obsm["protein_counts"].to_numpy(np.float32))
model = Ridge(alpha=1.0).fit(Z[train], Y[train])
pred = model.predict(Z).astype(np.float32)
q = np.quantile(Y[train] - pred[train], [0.05, 0.95], axis=0)           # raw (cell-wise, in-sample) 90% band

np.savez_compressed(WORK / f"pred_ridge_fit{SEED}.npz", y=Y[held], mean=pred[held], lo=pred[held] + q[0], hi=pred[held] + q[1],
                    sd=np.zeros_like(pred[held]), patient=pat[held], site=a.obs["Site"].astype(str).to_numpy()[held],
                    celltype=a.obs["initial_clustering"].astype(str).to_numpy()[held],
                    protein=a.obsm["protein_counts"].columns.to_numpy())
print("ridge held-out RMSLE:", round(float(np.sqrt(((pred[held] - Y[held]) ** 2).mean(0)).mean()), 4), "| cells:", len(held))
