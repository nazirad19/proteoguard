# Reproducibility status — 8 October 2026

## Fixed

- Stephenson totalVI reads `STEPH_FIT`; COMBAT training/ridge/low-rank scripts read `STEPH_FITS`.
- Fixed cohort paths now accept overrides; the runner supplies active paths and cell-type labels.
- An ordered runner creates sentinel JSON before measured-input ridge, then reruns calibration including it. Default is plan-only mode.
- Figure 7 reads actual `anchor_budget.csv`, aggregates within fit before averaging fits, and writes its summary. It rejects missing columns, duplicates and incomplete fit sets. The pasted-number version is labelled historical.
- A count-schema validator and explicit preparation adapter are provided. Missing original preparation is not presented as recovered.
- Eight CPU tests cover aggregation, no-call denominators, incomplete results, dependency ordering and invalid counts. GitHub Actions repeats them on Python 3.11.
- Documentation distinguishes transductive evaluation, coverage assumptions and selective biomarker calls. No unconditional per-patient clinical guarantee is asserted.

## Locally checked

Python syntax parsing of every script/test; eight CPU unit tests; dry-run plans for both cohorts. These do not establish GPU compatibility or reproduce research results. The lightweight local runtime differs from the recommended training environment.

## Remaining work before full reproduction

1. Recover original Stephenson/Hao preparation, or compare reconstructed selections with server manifests.
2. Export original participant/protein manifests, result CSVs, environment lock, hardware and prediction/checkpoint checksums. Check distribution terms before committing participant-level data.
3. Run the ordered pipeline in the research environment on cohort matrices. No cohort training was run in this revision.
4. Re-render all paper figures from results. Original plotting scripts can skip missing panels; inspect each output.
5. Confirm the longer scVAEIT run and its count-likelihood/input-scale assumptions. Existing predictions are skipped; changing SV_EPOCHS alone does not retrain them.
6. Audit quantile edge cases before changing calibration sizes: historical participant-block branches can index past available scores; some cell-wise/marker branches cap ranks instead of using an infinite endpoint. Reported n=15 block settings do not hit that overflow. Experimental outputs were not silently recomputed with altered statistical logic.
7. Resolve inductive versus transductive guarantee scope. Repeated allocations and two overlapping data splits are not independent participants or cohorts.

Source archive commit: `4cd21a01fed12e6176061fdc78ebc2f49f6892d9`. This revision changes execution and documentation; numerical tables retain their original provenance. See the report for scientific limitations, including in-sample ridge residual intervals and cohort-dependent offset correction.
