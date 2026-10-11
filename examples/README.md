# Runnable CPU demonstration

This example is an execution check, **not a reproduction of the report's biological results**. It includes synthetic integer RNA and ADT counts for 53 artificial participants, 24 cells each, 12 genes and 16 proteins. It contains no real participant records. The shared participant effect is deliberately built into the generator.

## Install and run

From the repository/submission root, with Python 3.11 or 3.12:

```bash
python -m venv .venv-demo
# Windows PowerShell: .\.venv-demo\Scripts\Activate.ps1
# macOS/Linux: source .venv-demo/bin/activate
python -m pip install -r requirements-demo.txt
python examples/run_toy.py
python -m unittest discover -s tests -v
```

No GPU, totalVI installation or data download is needed after installing the two lightweight dependencies. The example was tested on Windows with Python 3.12, NumPy 1.26.4 and pandas 2.2.3. The actual observed runtime and environment are recorded in `examples/expected_output/run_manifest.json`. A rerun needs a new directory, e.g. `python examples/run_toy.py --output runs/toy_second`.

## What executes

1. Read the bundled `toy_data/synthetic_counts.csv`; validate counts, cell IDs and disjoint participant roles.
2. Fit an RNA ridge predictor with training-only scaling on **12 training participants**. The penalty is fixed at 10, not selected on test data.
3. Estimate initial interval widths on **4 separate validation participants**.
4. Run the project's real `scripts/analysis/baselines_calib.py`: select five anchors and fit correction maps on **12 design participants** using leave-one-design-participant-out selection.
5. Refit ridge using RNA plus the five measured anchor channels on the same training participants. Target channels are not inputs. Anchor truth is available by design.
6. Run the real evaluator again, comparing correction maps and three calibration rules. For each of three deterministic repeated allocations, **15 calibration + 10 test participants** come from the remaining 25. Allocation repeats overlap and are not independent replicates.
7. Score only target proteins outside the selected anchors and synthetic isotype control; export tables, selected anchors, exact allocations, input checksum and environment information.

The example uses log1p raw ADT targets, library-normalized log1p RNA and NumPy ridge. It does not train totalVI or scVAEIT, reproduce the full RNA-SVD research predictor, simulate true isotype biology, or validate clinical marker calls. Initial intervals are estimated on separate validation participants here; some historical research scripts instead use in-sample residuals, as disclosed in the report.

## Outputs and expected behavior

`runs/toy_demo/` contains:

| File | Meaning |
| --- | --- |
| `summary.csv` | 21 predictor/correction/calibration combinations, averaged across the three allocations |
| `baselines_calibration.csv` | All 63 allocation-level rows |
| `point_metrics.csv` | Test RMSLE for RNA-only and RNA-plus-anchor ridge; identical targets |
| `sentinels_fit2026.json` | Five anchors selected using design participants only |
| `evaluation_splits.csv` | Exact calibration/test roles for each allocation |
| `run_manifest.json` | Input checksum, software versions, parameters and runtime |
| `*.npz` | Generated predictions in the research evaluator's schema |
| `*.log` | Captured evaluator messages |

Successful execution prints `PASS: synthetic CPU demo completed`. Small expected outputs are included in `expected_output/` so a reviewer can compare results. Floating-point differences across systems are possible. Do not expect every method to improve every metric; the sample is small and calibration is not a deterministic guarantee for every realized split. Historical `none (design mean)` labels mean a design-mean correction, not completely uncorrected predictions.

`python examples/make_toy_data.py` regenerates the bundled toy CSV with fixed seed 20261011. The main runner never modifies it. The fit label `2026` in prediction filenames is an interface convention, not the data-generation seed.

## Relation to full research runs

The optional `STEPH_PREDICTORS` JSON mapping was added to the evaluator to support a ridge-only first pass without fake totalVI predictions. Its ordinary full-research defaults are unchanged. Full cohort preparation, neural training, raw-data links and known reproducibility gaps are documented in `../docs/DATA_SOURCES.md`, `../docs/DATA_PREPARATION.md` and `../docs/REPRODUCIBILITY.md`.
