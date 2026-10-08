"""Explicit count-matrix adapter; a reconstruction, not the missing original prep.

Inspect a source first. Supply actual count-layer, protein and metadata keys;
normalized expression is rejected. Cohort-specific exclusions must be supplied
as an allowlist of cells. A manifest records every choice and input checksum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def check_counts(values, label):
    values = values.data if hasattr(values, "tocsr") else np.asarray(values)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{label}: counts must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-5):
        raise ValueError(f"{label}: expected raw integer counts, not normalized values")


def validate(a):
    for key in ["patient_id", "Site", "initial_clustering", "adt_total"]:
        if key not in a.obs or a.obs[key].isna().any():
            raise ValueError(f"Missing/null obs column: {key}")
    if not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique")
    if "counts" not in a.layers:
        raise ValueError("Missing raw RNA layers['counts']")
    check_counts(a.layers["counts"], "RNA")
    proteins = a.obsm.get("protein_counts")
    if not isinstance(proteins, pd.DataFrame):
        raise ValueError("protein_counts must be a DataFrame with protein names")
    if not proteins.columns.is_unique or not proteins.index.equals(a.obs_names):
        raise ValueError("Protein names must be unique and cell order must match")
    check_counts(proteins.to_numpy(), "ADT")
    if not np.allclose(a.obs.adt_total.to_numpy(), proteins.sum(axis=1).to_numpy()):
        raise ValueError("adt_total disagrees with protein_counts")
    if (a.obs.groupby("patient_id", observed=True).Site.nunique() != 1).any():
        raise ValueError("A participant occurs at multiple sites; resolve before splitting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ["inspect", "validate", "prepare"]:
        p = sub.add_parser(action); p.add_argument("input", type=Path)
        if action == "prepare":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--counts-layer", required=True, help="Explicit layer key, or X")
            p.add_argument("--protein-key", required=True, help="obsm key containing a named raw-count DataFrame")
            p.add_argument("--patient-key", required=True)
            p.add_argument("--site-key", required=True)
            p.add_argument("--celltype-key", required=True)
            p.add_argument("--cells", type=Path, required=True, help="QC-approved obs_names, one per line")
            p.add_argument("--max-cells", type=int, default=1000)
            p.add_argument("--n-genes", type=int, default=4000)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import anndata as ad
    if args.action == "inspect":
        a = ad.read_h5ad(args.input, backed="r")
        print(json.dumps({"shape": a.shape, "obs": list(a.obs), "layers": list(a.layers),
                          "obsm": list(a.obsm)}, indent=2)); a.file.close(); return
    if args.action == "validate":
        a = ad.read_h5ad(args.input); validate(a)
        print(f"Validated {a.n_obs} cells, {a.n_vars} genes, {a.obsm['protein_counts'].shape[1]} proteins"); return
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Output or manifest exists; choose a new path")
    if args.max_cells < 1 or args.n_genes < 2:
        parser.error("max-cells must be positive and n-genes at least two")
    import scanpy as sc
    import scipy.sparse as sp
    a = ad.read_h5ad(args.input)
    if not a.obs_names.is_unique:
        raise ValueError("Input cell IDs are not unique")
    approved = args.cells.read_text().splitlines()
    if len(approved) != len(set(approved)) or not approved or not set(approved) <= set(a.obs_names):
        raise ValueError("QC allowlist must contain unique, existing cell IDs")
    a = a[a.obs_names.isin(approved)].copy()
    for key in [args.patient_key, args.site_key, args.celltype_key]:
        if a.obs[key].isna().any():
            raise ValueError(f"Null labels in {key}")
    proteins = a.obsm[args.protein_key]
    if not isinstance(proteins, pd.DataFrame):
        raise ValueError("Protein obsm must be a named DataFrame; map names explicitly before preparation")
    check_counts(proteins.to_numpy(), "ADT")
    a = a[(proteins.sum(axis=1) > 0).to_numpy()].copy()
    rng = np.random.default_rng(args.seed); indices = []
    for _, positions in a.obs.groupby(args.patient_key, observed=True, sort=True).indices.items():
        indices.extend(rng.choice(positions, min(args.max_cells, len(positions)), replace=False))
    a = a[np.sort(indices)].copy()
    counts = a.X if args.counts_layer == "X" else a.layers[args.counts_layer]
    check_counts(counts, "RNA")
    g = ad.AnnData(sp.csr_matrix(counts, dtype=np.float32), obs=a.obs.copy(), var=a.var.copy())
    g.layers["counts"] = g.X.copy()
    g.obs["patient_id"] = a.obs[args.patient_key].astype("category")
    g.obs["Site"] = a.obs[args.site_key].astype("category")
    g.obs["initial_clustering"] = a.obs[args.celltype_key].astype("category")
    g.obsm["protein_counts"] = a.obsm[args.protein_key].copy()
    g.obs["adt_total"] = g.obsm["protein_counts"].sum(axis=1).to_numpy()
    t = g.copy(); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
    sc.pp.highly_variable_genes(t, n_top_genes=min(args.n_genes, t.n_vars - 1), flavor="seurat")
    g = g[:, t.var.highly_variable.to_numpy()].copy(); validate(g)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(input_sha256=digest, cells_sha256=hashlib.sha256(args.cells.read_bytes()).hexdigest(),
                    cells=list(g.obs_names), genes=list(g.var_names), proteins=list(g.obsm["protein_counts"].columns),
                    status="reconstructed preparation; compare with original cohort manifest before claiming reproduction")
    g.write_h5ad(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
