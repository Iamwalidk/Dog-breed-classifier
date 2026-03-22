# Dog Breed Classifier

Production-oriented full-stack dog breed classification project with:
- TensorFlow transfer-learning training workflow in `Dog_breed_model.ipynb`
- Flask inference service with robust model-loading fallbacks
- modern browser UI with ranked top-k predictions and breed metadata

## Why This Repository
- reproducible notebook pipeline for 120-class dog breed classification
- stable runtime behavior across mixed TensorFlow/Keras serialization formats
- practical diagnostics and conversion tooling for model portability
- clean API surface for web UI and programmatic clients

## Architecture
- `Dog_breed_model.ipynb`: training, staged fine-tuning, evaluation, export
- `my_flask_app/app.py`: app bootstrap, model loading strategy, health checks
- `my_flask_app/routes/predict.py`: upload validation + inference endpoint
- `my_flask_app/templates/index.html`: frontend shell
- `my_flask_app/static/`: modern responsive CSS and modular JS components
- `backend/data/breeds_info.json`: breed metadata enrichment
- `scripts/doctor.py`: runtime and artifact diagnostics
- `scripts/verify_project.py`: end-to-end project smoke test
- `scripts/convert_model.py`: model conversion utility for incompatible artifacts

## Runtime Model Resolution
The Flask app attempts model loading in this order:
1. `models/model.keras`
2. `models/saved_model`
3. `my_dog_breed_model.h5`

This allows the service to recover from common format/runtime mismatches.

## Prerequisites
- Python `3.11`
- Windows PowerShell (commands below are PowerShell-native)
- TensorFlow/Keras runtime compatible with exported artifacts

## Local Setup
```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run the Application
```powershell
venv\Scripts\python.exe scripts\verify_project.py
venv\Scripts\python.exe my_flask_app\app.py
```

Application URL:
- `http://127.0.0.1:5000`

## API Contract
### `POST /predict`
Request:
- content type: `multipart/form-data`
- field: `file=<image>`

Response:
```json
{
  "predictions": [
    {
      "breed": "Labrador Retriever",
      "confidence": 0.91,
      "confidence_percent": 91.0,
      "info": {
        "description": "...",
        "temperament": "...",
        "size": "...",
        "life_span": "..."
      }
    }
  ],
  "meta": {
    "top_k": 3,
    "inference_ms": 123.4,
    "image_size": 224,
    "model_source": "model.keras",
    "input_scaling": "model_preprocessing",
    "tta_steps": 3
  }
}
```

### `GET /health`
Returns:
- `status`
- `model_loaded`
- `model_source`
- `class_count`
- `image_size`
- `input_scaling`

## Notebook Workflow
`Dog_breed_model.ipynb` includes:
- deterministic setup and path configuration
- tf.data input pipeline
- EfficientNetV2S transfer learning
- stage 1 training + stage 2 fine-tuning
- top-1 and top-3 validation metrics
- export helpers for deployment artifacts

Expected exported artifacts:
- `models/model.keras` (preferred)
- `models/saved_model`
- `my_dog_breed_model.h5` (legacy fallback)
- `index_to_class.json`

## Environment Variables
### Training and notebook configuration
- `DOG_BREED_PROJECT_ROOT`
- `DOG_BREED_DATA_DIR`
- `DOG_BREED_SAMPLE_IMAGE`
- `DOG_BREED_IMAGE_SIZE`
- `DOG_BREED_BATCH_SIZE`
- `DOG_BREED_STAGE1_EPOCHS`
- `DOG_BREED_STAGE2_EPOCHS`
- `DOG_BREED_FINE_TUNE_LAYERS`
- `DOG_BREED_MIXUP_ALPHA`
- `DOG_BREED_LABEL_SMOOTHING`
- `DOG_BREED_STAGE1_LR`
- `DOG_BREED_STAGE2_LR`

### Flask runtime controls
- `DOG_BREED_MODEL_PATH`
- `DOG_BREED_CLASS_INDEX_PATH`
- `DOG_BREED_IMAGE_SIZE`
- `DOG_BREED_TOP_K`
- `DOG_BREED_TTA_STEPS`
- `DOG_BREED_INPUT_SCALING`
- `DOG_BREED_ALLOW_STALE_MODEL_FALLBACK`
- `DOG_BREED_MAX_UPLOAD_MB`
- `DOG_BREED_ALLOWED_EXTENSIONS`

`DOG_BREED_INPUT_SCALING` options:
- `normalize_0_1`
- `model_preprocessing`

## Diagnostics and Recovery
### Verify full project health
```powershell
venv\Scripts\python.exe scripts\doctor.py
venv\Scripts\python.exe scripts\verify_project.py
```

### Diagnose legacy `.h5` compatibility issues
```powershell
venv\Scripts\python.exe scripts\inspect_h5_config.py
venv\Scripts\python.exe scripts\convert_model.py
venv\Scripts\python.exe scripts\try_load_with_custom_objects.py
```

### Re-export from original training environment
```powershell
venv\Scripts\python.exe scripts\export_from_training_env.py
```

## Repository Hygiene
Large and environment-specific artifacts are intentionally ignored:
- virtual environments
- dataset directory (`archive (1)/`)
- model binaries (`*.h5`, `*.keras`, `models/`)
- runtime upload output (`Dog_breed_model/`)

If you need a runnable local model, generate/export artifacts with the notebook or scripts above.

## License
Add a `LICENSE` file before public distribution.
