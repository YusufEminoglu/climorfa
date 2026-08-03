# Data Staging

Use this folder as a strict data pipeline.

| Folder | Role | Examples |
|---|---|---|
| `00_external` | source notes, DOI/URL records, immutable metadata | reference PDF metadata, GEE collection notes |
| `01_raw` | untouched raw data | surface model, building footprints, road network, admin boundaries |
| `02_interim` | cleaned, clipped, projected data | EPSG:32635 layers, repaired geometry |
| `03_processed` | analysis-ready features | 250 m grid, feature matrix, sample tiles |
| `04_training_labels` | LCZ labels and audits | weak labels, manual validation points |
| `05_model_outputs` | predictions and explanation outputs | class probabilities, embeddings, Grad-CAM maps |

Do not overwrite raw data. Add a manifest whenever a new layer becomes analysis-ready.

