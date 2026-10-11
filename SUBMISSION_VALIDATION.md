# Submission validation — 11 October 2026

Prepared for Anel Umirbaeva and Nazira Dunbayeva.

## Scope

The submission includes the technical report in PDF, Word and Markdown, documented source, bundled synthetic count data, a CPU demonstration, tests, expected example outputs, and links to full-size source datasets. The presentation and study-design figure are optional supporting materials.

## Checks

- Installed `requirements-demo.txt` into a new isolated virtual environment.
- Windows, Python 3.12.14, NumPy 1.26.4, pandas 2.2.3; remaining versions are in `examples/tested-environment.txt`.
- Synthetic CPU run completed in approximately 2 seconds of computation on this machine; first-time dependency installation takes longer and runtime varies.
- The real calibration evaluator generated 63 rows and 21 aggregated combinations; all checked metrics were finite and coverages were in [0, 1].
- All 11 CPU unit tests passed, including participant-role leakage and train-only-scaling checks.
- Every Python source was syntax-checked; full research pipelines were inspected in dry-run mode.
- The final ZIP is extracted into a separate directory, its checksum manifest verified, and the demo and tests rerun there before delivery.
- Source files prior to submission edits matched GitHub snapshot `ecd422cad547f6d3e31a6943867136fc2a093198`. Submission-specific changes add the CPU example and optional ridge-only evaluator mapping.

The PDF and slides were visually checked in the preceding deliverable update. Word pagination remains unverified; its content matches the report but the PDF was separately typeset.

## Limits

This is not a new full-cohort reproduction. The submission does not include patient-level cohort matrices, historical checkpoints, original server logs or exact historical selection manifests. Source links and data schemas are provided; known preparation and statistical edge cases remain documented in `docs/REPRODUCIBILITY.md`. The synthetic example is not evidence for any empirical result or clinical guarantee in the report. Full totalVI/scVAEIT training dependencies were not installed or tested in the lightweight demo environment.

Detailed team-role allocations in the report are proposed for the authors' confirmation; equal overall credit follows their instruction.
