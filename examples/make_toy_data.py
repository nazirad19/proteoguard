"""Regenerate the included entirely synthetic counts (no participant data).

A latent cell state generates RNA and proteins. An independent participant
factor shifts all proteins: this intentionally illustrates the shared-offset
assumption, not evidence that the assumption holds in a real cohort.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261011)
    roles = ["train"] * 12 + ["width_validation"] * 4 + ["design"] * 12 + ["evaluation_pool"] * 25
    n_cells, n_genes, n_proteins = 24, 12, 16
    rna_load = rng.normal(0, .35, (3, n_genes))
    protein_load = rng.normal(0, .25, (3, n_proteins))
    shared_load = rng.uniform(.7, 1.2, n_proteins)
    rows = []
    for i, role in enumerate(roles):
        state = rng.normal(size=(n_cells, 3))
        shift = rng.normal(0, .55)
        rna = rng.poisson(np.exp(3.8 + state @ rna_load))
        adt = rng.poisson(np.exp(3.1 + state @ protein_load + shift * shared_load))
        for c in range(n_cells):
            row = dict(cell_id=f"SYN{i:03d}_{c:03d}", patient=f"SYN{i:03d}",
                       site="synthetic_site", celltype="type_A" if state[c, 0] > 0 else "type_B", role=role)
            row.update({f"RNA_{j:02d}": int(rna[c, j]) for j in range(n_genes)})
            row.update({f"AB_P{j:02d}": int(adt[c, j]) for j in range(n_proteins - 1)})
            row["AB_Mouse_IgG1_Isotype"] = int(adt[c, -1])
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "synthetic_counts.csv", index=False)
    (output / "generation.json").write_text(json.dumps(dict(seed=20261011,
        participants=len(roles), cells_per_participant=n_cells, genes=n_genes,
        proteins=n_proteins, synthetic=True), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "toy_data")
