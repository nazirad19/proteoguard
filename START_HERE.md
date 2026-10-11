# ProteoGuard — final project submission

**Authors:** Anel Umirbaeva and Nazira Dunbayeva (equal overall contribution).

## 1. Read the technical report

- `deliverables/ProteoGuard_Final_Project_Report.pdf` — reading copy.
- `deliverables/ProteoGuard_Final_Project_Report.docx` — editable Word copy.
- `deliverables/TECHNICAL_REPORT.md` — searchable source text.
- `deliverables/ProteoGuard_Final_Project.pptx` — optional presentation.

The report includes the problem, literature limitations, methods, rationale, positive/negative findings and limitations. Detailed contribution allocations remain proposed for author confirmation.

## 2. Run the included example

```bash
python -m venv .venv-demo
# Windows: .\.venv-demo\Scripts\Activate.ps1
# macOS/Linux: source .venv-demo/bin/activate
python -m pip install -r requirements-demo.txt
python examples/run_toy.py
python -m unittest discover -s tests -v
```

Python 3.11 or 3.12; CPU only. The two dependencies are pinned. On systems where `python` is not the Python 3 command, use `python3` or `py -3.12` to create the environment. The demo reads bundled **synthetic** counts, trains ridge predictors, selects five anchors, and executes the actual anchor/correction/calibration evaluator. It writes `runs/toy_demo/` and prints `PASS`. See `examples/README.md` for detailed steps, output definitions and expected results.

## 3. Review the source or run full cohorts

- `scripts/`: research preparation, training, calibration, evaluation, sensitivity and plotting scripts.
- `tests/`: CPU unit tests.
- `README.md`: research pipeline order and figure/table-to-script map.
- `docs/DATA_SOURCES.md`: links to the large raw/source datasets.
- `docs/DATA_PREPARATION.md`: required counts and metadata schemas.
- `docs/REPRODUCIBILITY.md`: what has and has not been independently rerun.
- `SUBMISSION_VALIDATION.md`: submission-package checks and exact software versions.

Full training uses the separate `requirements.txt` environment and may require substantial compute. **The toy run proves the included example executes; it does not reproduce the paper's numerical cohort tables.** Exact historical cohort selections, checkpoints and result exports are not bundled. These limitations are disclosed rather than replaced with synthetic research results.

Repository: https://github.com/nazirad19/proteoguard

The submission ZIP omits `.git`, virtual environments, real participant data, credentials and large checkpoints. `MANIFEST.sha256` lists the files in this submission snapshot for integrity checking.
