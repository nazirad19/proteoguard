# Data preparation

The archive contains the original COMBAT preparation, but not the original Stephenson/Hao preparation cells. A new preparation does not reproduce their exact cell/gene selection until compared with the original server manifests.

## Existing prepared data

Place `stephenson_sub1000.h5ad` in the cohort directory (same filename for COMBAT). Required schema:

- `layers['counts']`: finite, nonnegative integer RNA counts.
- `obsm['protein_counts']`: pandas DataFrame of raw ADT counts with unique protein names and exactly matching cell index.
- `obs['patient_id']`, `obs['Site']`, `obs['initial_clustering']`: non-null labels.
- `obs['adt_total']`: sum of ADT counts per cell.
- Unique cell/gene names. Each participant belongs to one site for this split implementation.

```bash
python scripts/prepare_data.py validate /path/to/stephenson_sub1000.h5ad
```

Stephenson low-rank analysis also reads `haniffa21.processed.h5ad`, with patient_id, total_counts, Status_on_day_collection_summary, Sex and Age_interval. Keep it in the same directory. Cell-type label spelling matters.

## COMBAT original preparation

Set `STEPH_DIR` (or `COMBAT_DIR`) to the folder containing `COMBAT-CITESeq-DATA.h5ad`, then run `python scripts/combat/combat_prep.py`.

The script uses `layers['raw']`, separates `AB_` features, and excludes LDN preparations, GEX_region starting E, and missing major-subset annotation. It samples at most 1,000 cells per participant (seed 0), then selects up to 4,000 Seurat-flavor HVGs after normalization/log1p. Raw counts are retained for fitting. Compare printed cohort counts with the original run before training.

## Reconstructed adapter

```bash
python scripts/prepare_data.py inspect /path/to/source.h5ad
python scripts/prepare_data.py prepare /path/to/source.h5ad --output /path/to/stephenson_sub1000.h5ad --counts-layer RAW_LAYER --protein-key RAW_ADT_OBSM --patient-key PARTICIPANT_COLUMN --site-key SITE_COLUMN --celltype-key CELLTYPE_COLUMN --cells /path/to/qc_cells.txt
```

Uppercase arguments are placeholders: supply actual keys from inspection. The QC allowlist contains approved cell IDs, one per line, implementing cohort-specific exclusions. This adapter requires a named raw ADT DataFrame; it does not infer names or invert CLR/DSB values. Its manifest records input SHA-256, selected cells/genes/proteins and parameters. Existing outputs are protected. HVGs use all included RNA, consistent with a transductive workflow.

Count validation is unit-tested, but the full AnnData/HVG path has not been run on these cohorts locally. This is not a verified replacement for the missing original preparation. Hao donor/day grouping needs an explicit mapping; do not reuse COVID cohort grouping blindly.

## Predictions

`pred_fit{seed}.npz`, `pred_ridge_fit{seed}.npz`, `pred_ridgeanchor_fit{seed}.npz` contain `y`, `mean`, `lo`, `hi` (cells × proteins, log1p raw ADT scale), cell-level `patient`, `site`, `celltype`, ordered `protein` names and optional `sd`.

Cells must be contiguous by participant because analyses construct participant slices. Preserve identical cell/target order across predictors. Load only trusted NPZ files; historical scripts use `allow_pickle=True` for labels.
