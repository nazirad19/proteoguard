# Provenance manifests

Frozen participant and protein manifests exported by `scripts/analysis/provenance_audit.py`
(run 7 Oct 2026 on the working copies of both cohorts):

- `stephenson/provenance_participants.csv`, `stephenson/provenance_proteins.csv`
- `combat/provenance_participants.csv`, `combat/provenance_proteins.csv`

Each participant row records cohort, fit (2026/2027), role (train / design /
calibration_test_pool / external_sanger), site/pool, and mechanism-analysis membership.
Each protein row records anchor/isotype status and evaluation-set membership
(177 proteins for Stephenson fit 1, 179 for fit 2; 179 for both COMBAT fits).

To refresh: `STEPH_DIR=<cohort dir> python scripts/analysis/provenance_audit.py`,
then copy the two CSVs from the cohort directory into the matching folder here.
