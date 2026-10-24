# Stephenson preparation, mirroring scripts/combat/combat_prep.py
# Filters per the manuscript: drop in-vitro LPS-stimulated samples, cap 1,000 cells per participant,
# drop cells without ADT counts. Writes stephenson_sub1000.h5ad in the COMBAT-compatible schema.
import os
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.utils import check_random_state

WORK = Path(os.environ.get("STEPH_DIR", "data"))
A = ad.read_h5ad(WORK / "haniffa21.processed.h5ad", backed="r")
obs = A.obs
print("sites:", obs["Site"].value_counts().to_dict())
lps = obs["time_after_LPS"].astype(str)
keep = ~lps.isin(["10h", "90m"])                      # in-vitro stimulated samples
print("cells after LPS filter:", int(keep.sum()), "| participants:", obs.loc[keep.values, "patient_id"].nunique())

rng = np.random.default_rng(0); idx = []
for pid, ix in pd.Series(np.where(keep)[0]).groupby(obs["patient_id"].astype(str).to_numpy()[keep.to_numpy()]):
    ix = ix.to_numpy(); idx.append(rng.choice(ix, min(1000, len(ix)), replace=False))
idx = np.sort(np.concatenate(idx)); print("cells sampled:", len(idx))

sub = A[idx].to_memory()
raw = sub.layers["raw"]; raw = raw.tocsr() if sp.issparse(raw) else sp.csr_matrix(raw)
ab = sub.var["feature_types"].astype(str).to_numpy() == "Antibody Capture"
prot = pd.DataFrame(raw[:, ab].toarray(), index=sub.obs_names, columns=sub.var_names[ab])
adt_total = prot.sum(1).values
nz = adt_total > 0
print("cells with zero ADT dropped:", int((~nz).sum()), "-> remaining:", int(nz.sum()))
prot = prot.loc[nz]; raw = raw[nz]; sub = sub[nz]
adt_total = adt_total[nz]

g = ad.AnnData(raw[:, ~ab].astype(np.float32), obs=pd.DataFrame(index=sub.obs_names),
               var=pd.DataFrame(index=sub.var_names[~ab]))
g.layers["counts"] = g.X.copy()
t = g.copy()
tot = np.asarray(t.X.sum(1)).ravel()
t.X = sp.diags(1e4 / np.maximum(tot, 1)) @ t.X; t.X.data = np.log1p(t.X.data)
import scanpy as sc
sc.pp.highly_variable_genes(t, n_top_genes=min(4000, t.n_vars - 1), flavor="seurat")
g = g[:, t.var["highly_variable"].to_numpy()].copy(); del t

o = sub.obs
g.obs = pd.DataFrame({"patient_id": o["patient_id"].astype(str).values,
                      "sample_id": o["sample_id"].astype(str).values,
                      "Site": o["Site"].astype(str).values,
                      "Pool": o["Site"].astype(str).values,          # no pool metadata in Stephenson
                      "Status": o["Status_on_day_collection_summary"].astype(str).values,
                      "initial_clustering": o["initial_clustering"].astype(str).values,
                      "adt_total": adt_total}, index=sub.obs_names)
for c in ["patient_id","sample_id","Site","Pool","Status","initial_clustering"]:
    g.obs[c] = g.obs[c].astype("category")
g.obsm["protein_counts"] = prot
g.write_h5ad(WORK / "stephenson_sub1000.h5ad")

p = g.obs.groupby("patient_id", observed=True).agg(site=("Site","first"), cells=("Site","size"))
print(g, "\nparticipants:", len(p), "| by site:", p.site.value_counts().to_dict())
print("total cells:", g.n_obs)
print("cell types:", g.obs["initial_clustering"].value_counts().head(8).to_dict())
