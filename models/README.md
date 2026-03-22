# Model Artifacts

This directory is intentionally kept out of version control except for this note.

Generate deployment artifacts locally by running:
- `Dog_breed_model.ipynb` export cells, or
- `python scripts/export_from_training_env.py`, or
- `python scripts/convert_model.py` (when compatible conversion is possible)

Preferred runtime artifact:
- `models/model.keras`

Fallback runtime artifact:
- `models/saved_model`
