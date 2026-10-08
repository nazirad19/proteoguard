# %% Cell C3 - COMBAT subsample in the same format as stephenson_sub1000.h5ad
# participant = COMBAT_ID (timepoints pooled, as in Stephenson); Site = Institute (Oxford / St_Georges); Pool_ID kept.
# Dropped: LDN preparations (different cell fraction, like Stephenson's LPS samples), doublets and unannotated cells.
import os
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp
from pathlib import Path
WORK = Path(os.environ.get("STEPH_DIR", os.environ.get("COMBAT_DIR", Path.home() / "projects" / "combat")))
WORK.mkdir(parents=True, exist_ok=True)
A = ad.read_h5ad(WORK / "COMBAT-CITESeq-DATA.h5ad", backed="r")
obs = A.obs
print("Source:", obs["Source"].value_counts().to_dict())
keep = (~obs["Source"].astype(str).str.contains("LDN")) & (~obs["GEX_region"].astype(str).str.startswith("E")) \
       & (obs["Annotation_major_subset"].astype(str) != "nan")
rng = np.random.default_rng(0); idx = []
for pid, ix in pd.Series(np.where(keep)[0]).groupby(obs["COMBAT_ID"].astype(str).to_numpy()[keep.to_numpy()]):
    ix = ix.to_numpy(); idx.append(rng.choice(ix, min(1000, len(ix)), replace=False))
idx = np.sort(np.concatenate(idx)); print("cells kept:", len(idx))

sub = A[idx].to_memory()                                      # loads X and layers['raw'] for these rows only
raw = sub.layers["raw"]; raw = raw.tocsr() if sp.issparse(raw) else sp.csr_matrix(raw)
ab = sub.var_names.str.startswith("AB_")
prot = pd.DataFrame(raw[:, ab].toarray(), index=sub.obs_names, columns=sub.var_names[ab])
g = ad.AnnData(raw[:, ~ab].astype(np.float32), obs=pd.DataFrame(index=sub.obs_names), var=pd.DataFrame(index=sub.var_names[~ab]))
g.layers["counts"] = g.X.copy()
t = g.copy(); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)            # HVGs without extra dependencies
sc.pp.highly_variable_genes(t, n_top_genes=min(4000, t.n_vars - 1), flavor="seurat")
g = g[:, t.var["highly_variable"].to_numpy()].copy(); del t
o = sub.obs
g.obs = pd.DataFrame({"patient_id": o["COMBAT_ID"].astype(str).values, "sample_id": o["scRNASeq_sample_ID"].astype(str).values,
                      "Site": o["Institute"].astype(str).values, "Pool": o["Pool_ID"].astype(str).values,
                      "Status": o["Source"].astype(str).values, "initial_clustering": o["Annotation_major_subset"].astype(str).values,
                      "adt_total": prot.sum(1).values}, index=sub.obs_names)
for c in ["patient_id", "sample_id", "Site", "Pool", "Status", "initial_clustering"]: g.obs[c] = g.obs[c].astype("category")
g.obsm["protein_counts"] = prot
g.write_h5ad(WORK / "stephenson_sub1000.h5ad")                # same file name so every existing cell runs with STEPH_DIR=combat

p = g.obs.groupby("patient_id", observed=True).agg(site=("Site", "first"), status=("Status", "first"),
                                                   pools=("Pool", "nunique"), cells=("Site", "size"))
print(g, "\nparticipants:", len(p), "| by site:", p.site.value_counts().to_dict())
print("by status:", p.status.value_counts().to_dict(), "| participants spanning >1 pool:", int((p.pools > 1).sum()))
print("cell types:", g.obs["initial_clustering"].value_counts().head(12).to_dict())
print("isotype-like ADTs:", [c for c in prot.columns if any(k in c.lower() for k in ["iso", "igg", "mouse", "rat", "hamster"])])
print("cells with zero ADT:", int((g.obs["adt_total"] == 0).sum()))
