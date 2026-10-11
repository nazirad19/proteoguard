# ProteoGuard

Participant-level calibration and small measured-protein panels for RNA-to-protein imputation.

**Biological question:** can RNA, supplemented by a small measured surface-protein panel, support reliable immune-marker calls in participants whose target proteins were not measured?

ProteoGuard evaluates existing imputers (totalVI and ridge, with a completed scVAEIT comparison reported), anchor-based correction, participant-block calibration, and selective biomarker calls. It does not introduce a new totalVI architecture. The reported shared-offset mechanism is strongest in Stephenson and weaker in COMBAT. Measured-panel inputs and calibration must be distinguished from post-hoc offset correction. These are research results, not validated clinical decisions.

## Deliverables

**Course submission:** start with [START_HERE.md](START_HERE.md). The repository includes [synthetic toy data and a runnable CPU example](examples/README.md), a pinned lightweight environment (`requirements-demo.txt`), tests, and [links to full-size source data](docs/DATA_SOURCES.md).

```bash
python -m pip install -r requirements-demo.txt
python examples/run_toy.py
python -m unittest discover -s tests -v
```

Use a fresh Python 3.11/3.12 virtual environment (instructions in START_HERE). The toy example trains ridge predictors and executes the actual calibration evaluator; it does not train totalVI/scVAEIT or recreate the cohort tables.

- [Technical report](deliverables/TECHNICAL_REPORT.md)
- [Word report](deliverables/ProteoGuard_Final_Project_Report.docx)
- [PDF report](deliverables/ProteoGuard_Final_Project_Report.pdf)
- [Presentation: 11 main slides + 7 backup slides](deliverables/ProteoGuard_Final_Project.pptx)
- [Illustrated study design](deliverables/figures/ProteoGuard_study_design.png)
- [Deliverable provenance and format notes](docs/DELIVERABLES.md)
- [Reproducibility status](docs/REPRODUCIBILITY.md)
- [Data preparation and file schemas](docs/DATA_PREPARATION.md)

## Status

The 11 October 2026 course update aligns the report and slides with the six assessment criteria, adds an illustrated study design, and records the reported completed scVAEIT comparison. Anel Umirbaeva and Nazira Dunbayeva receive equal overall contribution credit. Detailed role allocations are proposed for author confirmation. This is a documentation update, not a new training run.


The source archive contained 33 analysis scripts at commit `4cd21a01fed12e6176061fdc78ebc2f49f6892d9`. This revision fixes configuration, pipeline ordering and budget-figure regeneration, and adds a count-data adapter and CPU tests. **Full cohort training and result reproduction have not been rerun as part of these fixes.** Matrices, predictions, exact split manifests and result CSVs were not supplied in the archive. Report numbers remain supplied results, not newly verified outputs.

The analyses are **transductive**: held-out participants' RNA contributes to totalVI fitting while target proteins are hidden; RNA SVD for ridge also uses all RNA. This is not fully inductive deployment to a participant arriving after training.

## Data

| Cohort | Source | Input |
| --- | --- | --- |
| Stephenson | COVID-19 Cell Atlas / E-MTAB-10026 | `haniffa21.processed.h5ad` plus prepared counts |
| COMBAT | [Zenodo 6120249](https://zenodo.org/records/6120249) | `COMBAT-CITESeq-DATA.h5ad` |
| Hao (exploratory) | GSE164378 | separately prepared counts/predictions; original preparation absent |
| Original totalVI benchmark | scvi-tools datasets | `scripts/benchmark/gate0_uncertainty_audit.py` |

No individual-level data or checkpoints are committed. Obtain cohort data under the source's terms, then follow [data preparation](docs/DATA_PREPARATION.md). Do not assume a processed H5AD object contains raw ADT counts.

## Environment

Target: Python 3.11 and `scvi-tools==1.4.2`. Install in an isolated environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

This is a dependency specification, not the original server's environment lock. Record `python -m pip freeze`, GPU/driver information, seeds and input checksums for each full run. scVAEIT is optional and should use a separate TensorFlow environment; see requirements.txt. The updated source report describes completed 200-epoch runs per split. Reported metrics are included in the report; raw logs, convergence and prediction-level outputs have not been independently verified locally.

## Core pipeline

From the repository root, inspect the plan first; these commands **do not train**:

```bash
python scripts/run_pipeline.py --cohort stephenson --work /path/to/stephenson --seeds 2026 2027
python scripts/run_pipeline.py --cohort combat --work /path/to/combat --seeds 2026 2027
```

After preparing/validating data, add `--execute` to run. The runner checks the prepared schema and stops on a failed subprocess. Preserve its `pipeline_run.json` with the environment snapshot. Training can be expensive.

The actual order is:

1. totalVI and RNA ridge predictions for each seed.
2. Low-rank analysis and frozen design split.
3. `baselines_calib.py` first pass: selects anchors on design participants and writes `sentinels_fit{seed}.json`.
4. `ridge_sentinels.py`: consumes those JSON files.
5. `baselines_calib.py` second pass: now includes measured-input ridge.
6. Participant metrics, anchor budgets, marker calls and risk–coverage analyses.
7. CSV-driven budget figure. Other figures/audits remain explicit commands below.

The runner sets cohort-specific cell-type labels. Override `STEPH_CTS` only with verified metadata labels. `STEPH_FIT` selects one fit in Stephenson training/ridge/lowrank; `STEPH_FITS` selects multiple fits in other scripts. `STEPH_DIR` selects the active cohort. For two-cohort figures/provenance use `STEPH_DIR_MAIN` and `COMBAT_DIR`; `HAO_DIR` selects Hao.

Example, running only one Stephenson fit in PowerShell:

```powershell
$env:STEPH_DIR = 'D:\data\stephenson'
$env:STEPH_FIT = '2027'
python scripts/stephenson/stephenson_donor_calibration.py
```

Replot Figure 7 from actual outputs:

```bash
python scripts/figures/fig_budget.py --input /path/to/anchor_budget.csv --output /path/to/figures/f7.png
```

This writes the figure and an aggregated `.summary.csv`. Missing columns, duplicate rows and incomplete seed/budget coverage cause errors. `fig_budget_historical.py` preserves the pasted-number illustration for provenance only.

## Figure / table map

Paths below are under `scripts/`. Numbering follows the full paper draft; the course deck groups results differently.

| Paper item | Analysis / rendering |
| --- | --- |
| Figure 1: simulation | `simulation/sim_donor_blocks.py` |
| Figures 2–6; supplementary Hao | `figures/paper_figures.py` reads saved outputs |
| Figure 7: anchor budget | `analysis/anchor_budget.py` → `figures/fig_budget.py` |
| Figures 8–9: controls, COMBAT | `analysis/baselines_calib.py`, `combat/combat_lowrank.py`, `figures/paper_figures2.py` |
| Tables 1–2 | `analysis/proteoguard_metrics.py` |
| Tables 3–4 | `analysis/baselines_calib.py` (second pass) |
| Balanced-offset sensitivity | `analysis/sensitivity_balanced.py` |
| Biomarker calls / matched call rates | `analysis/marker_calls_all.py`, `analysis/risk_coverage_b.py` |
| Participant / protein manifests | `analysis/provenance_audit.py` |
| Cohort / anchor detection audit | `stephenson/cohort_audit.py` |

Original plotting scripts can skip missing panels. Inspect logs and files; successful exit alone does not confirm all figures were regenerated. Negative-result scripts are retained in `scripts/stephenson/`. `scripts/hao/hao_mechanism.py` analyzes saved predictions; it is not a complete Hao training pipeline.

## Statistical interpretation

For one protein, a participant-block rank argument with 15 exchangeable calibration participants and the 13th order statistic gives a marginal lower bound of 13/16 on achieving at least 90% empirical coverage in a new evaluated participant block. This assumes the required exchangeability and a fixed or appropriately symmetric prediction procedure. It is not a guarantee for every participant, simultaneously across proteins, or for a new site. Transductive fitting requires a symmetry justification before using this as a deployment guarantee. See the report for finite-sample details and unresolved edge cases.

## License

MIT, attributed to ProteoGuard contributors. See [LICENSE](LICENSE).
