# Full-size data sources

Large cohort matrices and checkpoints are not in the submission. The included `examples/toy_data/` is entirely synthetic and sufficient to run the CPU example. The reported biological results use the external studies below.

## Main cohorts

### Stephenson et al., COVID-19 PBMC atlas

- Original study/data accession: [E-MTAB-10026 at EMBL-EBI BioStudies/ArrayExpress](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026).
- Authors' processed-data portal: [COVID-19 Cell Atlas, patient datasets](https://www.covid19cellatlas.org/index.patient.html).
- The project references `haniffa21.processed.h5ad`, plus a prepared count object `stephenson_sub1000.h5ad`.
- Obtain RNA count matrices, ADT counts and participant/site/cell-type metadata from the deposition. Inspect the layer names and scale: processed or normalized ADTs cannot simply replace raw counts. The original preparation cells and exact server selection manifests were not supplied, so reconstructed preparation must be checked against those manifests before claiming exact reproduction.

### COMBAT, COVID-19 Multi-omics Blood Atlas

- Authors' data record: [Zenodo 6120249](https://zenodo.org/records/6120249).
- [Dataset descriptions](https://zenodo.org/records/6120249/files/Dataset_descriptions.pdf).
- Select the CITE-seq AnnData deposit containing `COMBAT-CITESeq-DATA.h5ad` (rather than a separately downsampled third-party dataset).
- The original preparation script is `scripts/combat/combat_prep.py`. It expects RNA and `AB_` features in the raw-count layer and applies the documented cohort/QC filters.
- Some separate modalities have controlled-access terms. Only obtain/use the required publicly released CITE-seq data under its source terms; the project does not redistribute participant-level cohort data.

## Exploratory/legacy scripts retained for provenance

- Hao multimodal PBMC: [GEO GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378). The repository contains analysis of saved predictions, not a complete original Hao preparation/training pipeline.
- Original totalVI engineering benchmark: [totalVI reproducibility repository](https://github.com/YosefLab/totalVI_reproducibility) and the dataset loader used in `scripts/benchmark/gate0_uncertainty_audit.py`.

Use `docs/DATA_PREPARATION.md` for exact required fields, input inspection, count validation and preparation commands. Save download dates, checksums, source terms, selected cells/features and environment information with each real run. Links identify the source datasets; they do not substitute for the missing exact historical manifests.
