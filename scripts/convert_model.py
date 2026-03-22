import ast
import json
import os
import shutil
import sys
import traceback
from datetime import datetime


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(REPO_ROOT, "my_flask_app", "app.py")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
KERAS_OUTPUT_PATH = os.path.join(MODELS_DIR, "model.keras")
SAVED_MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "saved_model")
METADATA_OUTPUT_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def print_section(title):
    print(f"\n=== {title} ===")


def read_app_constants():
    constants = {}
    if not os.path.exists(APP_PATH):
        return constants

    with open(APP_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=APP_PATH)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in {"MODEL_PATH", "CLASS_INDEX_PATH"}:
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                constants[target.id] = node.value.value
    return constants


def repo_path(value):
    if not value:
        return None
    if os.path.isabs(value):
        return value
    return os.path.join(REPO_ROOT, value)


def to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)


def detect_h5_weights_only(model_path):
    if not str(model_path).lower().endswith(".h5"):
        return False, "not an h5 file"

    try:
        import h5py  # type: ignore
    except Exception as exc:
        return False, f"h5py unavailable ({exc})"

    try:
        with h5py.File(model_path, "r") as h5f:
            has_model_config = "model_config" in h5f.attrs
            has_training_config = "training_config" in h5f.attrs
            if has_model_config:
                return False, "contains model_config"
            if "layer_names" in h5f.attrs or "model_weights" in h5f:
                return True, "appears to contain weights only (no model_config in H5 attrs)"
            return False, "H5 structure not clearly weights-only"
    except Exception as exc:
        return False, f"failed to inspect h5 ({exc})"


def import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception:
        print("Failed to import TensorFlow.")
        traceback.print_exc()
        print("\nNext step: pip install -r requirements.txt")
        return None


def try_tf_keras_load(tf, model_path):
    print(f"Trying tf.keras.models.load_model(..., compile=False): {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def try_standalone_keras_load(model_path):
    print(f"Trying keras.models.load_model(..., compile=False): {model_path}")
    import keras  # type: ignore

    return keras.models.load_model(model_path, compile=False)


def attempt_model_load(tf, model_path):
    failures = []

    try:
        model = try_tf_keras_load(tf, model_path)
        return model, failures
    except Exception:
        failures.append(("tf.keras.models.load_model", traceback.format_exc()))

    is_weights_only, reason = detect_h5_weights_only(model_path)
    if str(model_path).lower().endswith(".h5"):
        print(f"H5 inspection: {reason}")
        if is_weights_only:
            print("This file looks like H5 weights only, not a full serialized model.")
            print("Automatic conversion cannot rebuild architecture without the original model definition code.")

    try:
        model = try_standalone_keras_load(model_path)
        return model, failures
    except Exception:
        failures.append(("keras.models.load_model", traceback.format_exc()))

    return None, failures


def print_incompatibility_guidance(failures):
    combined = "\n".join(trace for _, trace in failures)

    if "Unrecognized keyword arguments: ['batch_shape']" in combined:
        print("\nDetected incompatibility: legacy InputLayer config uses `batch_shape`.")
        print("Try the targeted patch first:")
        print("- python scripts/patch_h5.py")

    if "Unknown dtype policy: 'DTypePolicy'" in combined or "DTypePolicy" in combined:
        print("\nDetected incompatibility: model was likely serialized with a newer Keras format/config")
        print("that TensorFlow/Keras 2.12 cannot fully deserialize from this H5.")
        print("This usually requires re-exporting the model from the original training environment.")

    if "batch_shape" in combined or "DTypePolicy" in combined:
        print("\nExplicit re-export steps (original training environment):")
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"- Created/verified target folder: {MODELS_DIR}")
        print("1) Activate the environment that can load the original model.")
        print("2) Load the model there (compile=False if needed).")
        print("3) Save one of these artifacts in this repo:")
        print("   - models/model.keras")
        print("   - models/saved_model")
        print("4) Start the app again; app.py will prefer converted artifacts first.")
        print("\nTemplate snippet to run in the training environment:")
        print("import tensorflow as tf")
        print("model = tf.keras.models.load_model('my_dog_breed_model.h5', compile=False)")
        print("model.save('models/model.keras')  # preferred")
        print("# or: model.save('models/saved_model')")


def clear_output_target(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def load_label_count(class_index_path):
    if not class_index_path or not os.path.exists(class_index_path):
        return None
    try:
        with open(class_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return len(data)
    except Exception:
        pass
    return None


def extract_class_count_from_model(model):
    output_shape = getattr(model, "output_shape", None)
    if isinstance(output_shape, (list, tuple)) and output_shape and isinstance(output_shape[0], (list, tuple)):
        last_shape = output_shape[0]
    else:
        last_shape = output_shape

    if isinstance(last_shape, (list, tuple)) and last_shape:
        last_dim = last_shape[-1]
        if isinstance(last_dim, int):
            return last_dim
    return None


def save_metadata(tf, model, source_model_path, class_index_path, saved_artifact):
    os.makedirs(MODELS_DIR, exist_ok=True)

    keras_version = getattr(getattr(tf, "keras", None), "__version__", None)
    if keras_version is None:
        try:
            import keras  # type: ignore

            keras_version = getattr(keras, "__version__", None)
        except Exception:
            keras_version = None

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_model_path": source_model_path,
        "converted_artifact": saved_artifact,
        "tensorflow_version": getattr(tf, "__version__", None),
        "keras_version": keras_version,
        "input_shape": to_jsonable(getattr(model, "input_shape", None)),
        "output_shape": to_jsonable(getattr(model, "output_shape", None)),
        "class_count": load_label_count(class_index_path) or extract_class_count_from_model(model),
        "preprocessing": {
            "image_size": [224, 224],
            "dtype": "float32",
            "normalization": "divide by 255.0",
            "batch_dimension": "expand_dims(axis=0)",
            "source": "my_flask_app/routes/predict.py",
        },
    }

    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {METADATA_OUTPUT_PATH}")


def save_converted_model(tf, model):
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        if os.path.exists(KERAS_OUTPUT_PATH):
            clear_output_target(KERAS_OUTPUT_PATH)
        model.save(KERAS_OUTPUT_PATH)
        print(f"Saved converted model (.keras): {KERAS_OUTPUT_PATH}")
        return {"type": "keras", "path": KERAS_OUTPUT_PATH}
    except Exception:
        print("Failed to save .keras artifact. Falling back to SavedModel format.")
        traceback.print_exc()

    try:
        if os.path.exists(SAVED_MODEL_OUTPUT_PATH):
            clear_output_target(SAVED_MODEL_OUTPUT_PATH)
        model.save(SAVED_MODEL_OUTPUT_PATH, save_format="tf")
        print(f"Saved converted model (SavedModel): {SAVED_MODEL_OUTPUT_PATH}")
        return {"type": "saved_model", "path": SAVED_MODEL_OUTPUT_PATH}
    except Exception:
        print("Failed to save converted model in both .keras and SavedModel formats.")
        traceback.print_exc()
        return None


def main():
    print("Dog Breed Classifier Model Converter")
    print(f"Repo root: {REPO_ROOT}")

    constants = read_app_constants()
    model_path = repo_path(constants.get("MODEL_PATH"))
    class_index_path = repo_path(constants.get("CLASS_INDEX_PATH"))

    print_section("Source Model")
    print(f"MODEL_PATH (from app.py): {constants.get('MODEL_PATH')}")
    print(f"Resolved path          : {model_path}")

    if not model_path or not os.path.exists(model_path):
        print("Model file not found. Aborting.")
        print("Next step: verify MODEL_PATH in my_flask_app/app.py")
        return 1

    if os.path.isdir(model_path):
        print("Detected model type    : directory (likely SavedModel)")
    else:
        print(f"Detected model type    : file ({os.path.splitext(model_path)[1] or 'no extension'})")
        print(f"File size              : {os.path.getsize(model_path)} bytes")

    tf = import_tensorflow()
    if tf is None:
        return 1

    print_section("Load Attempt")
    model, failures = attempt_model_load(tf, model_path)
    if model is None:
        print("Model conversion failed: could not load source model in this environment.")
        print("You cannot convert here because the model cannot be loaded here. Re-export is required.")
        for loader_name, failure_trace in failures:
            print(f"\n--- Failure: {loader_name} ---")
            print(failure_trace)

        print_incompatibility_guidance(failures)
        print("\nModel appears incompatible; need to re-export from the training environment. See instructions:")
        print("1) Activate the original training environment.")
        print("2) Load the model there.")
        print("3) Re-save with tf.keras using: model.save('models/model.keras') or model.save('models/saved_model')")
        print("4) Or run in that environment: python scripts/export_from_training_env.py")
        return 1

    print("Model load succeeded.")
    print(f"Model class : {model.__class__.__name__}")
    print(f"Input shape : {getattr(model, 'input_shape', 'unknown')}")
    print(f"Output shape: {getattr(model, 'output_shape', 'unknown')}")

    print_section("Save Converted Artifact")
    saved_artifact = save_converted_model(tf, model)
    if saved_artifact is None:
        return 1

    print_section("Metadata")
    save_metadata(tf, model, model_path, class_index_path, saved_artifact)

    print("\nDone. Original model file was not modified or deleted.")
    print("Next steps:")
    print("- Run: python scripts/doctor.py")
    print("- Start app: python my_flask_app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
