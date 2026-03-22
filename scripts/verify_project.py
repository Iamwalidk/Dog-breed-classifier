import ast
import json
import os
import subprocess
import sys
import traceback


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(REPO_ROOT, "my_flask_app", "app.py")
MODELS_DIR = os.path.join(REPO_ROOT, "models")


def print_section(title):
    print(f"\n=== {title} ===")


def read_app_constants():
    constants = {}
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
    if os.path.isabs(value):
        return value
    return os.path.join(REPO_ROOT, value)


def run_doctor():
    print_section("Doctor")
    doctor_path = os.path.join(REPO_ROOT, "scripts", "doctor.py")
    result = subprocess.run([sys.executable, doctor_path], cwd=REPO_ROOT, check=False)
    print(f"doctor.py exit code: {result.returncode}")
    return result.returncode


def check_labels():
    print_section("Labels")
    constants = read_app_constants()
    class_index_path = repo_path(constants["CLASS_INDEX_PATH"])
    print(f"Resolved labels path: {class_index_path}")
    if not os.path.exists(class_index_path):
        print("ERROR: index_to_class.json not found.")
        return False
    try:
        with open(class_index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = len(data) if isinstance(data, dict) else "unknown"
        print(f"Labels file OK ({count} classes)")
        return True
    except Exception:
        print("ERROR: Could not parse labels file.")
        traceback.print_exc()
        return False


def find_model_artifact():
    print_section("Model Artifact")
    os.makedirs(MODELS_DIR, exist_ok=True)
    candidates = [
        os.path.join(MODELS_DIR, "model.keras"),
        os.path.join(MODELS_DIR, "saved_model"),
    ]
    found = []
    for path in candidates:
        exists = os.path.isdir(path) or os.path.isfile(path)
        print(f"{path} -> {'FOUND' if exists else 'missing'}")
        if exists:
            found.append(path)
    if found:
        return found
    print("No converted model artifact found.")
    print("Expected one of:")
    print("- models/model.keras (preferred)")
    print("- models/saved_model")
    print("Next step: run python scripts/export_from_training_env.py in the environment that can load the original H5.")
    return None


def try_load_and_forward(model_paths):
    print_section("Model Smoke Test")
    try:
        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore
    except Exception:
        print("ERROR: TensorFlow/NumPy imports failed.")
        traceback.print_exc()
        return False, None

    model = None
    loaded_path = None
    for model_path in model_paths:
        print(f"Loading model (compile=False): {model_path}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded_path = model_path
            break
        except Exception:
            print(f"WARNING: Could not load artifact: {model_path}")
            traceback.print_exc()
            continue

    if model is None:
        print("ERROR: All detected model artifacts failed to load.")
        return False, None

    print(f"Load OK: {model.__class__.__name__}")
    print(f"Input shape : {getattr(model, 'input_shape', None)}")
    print(f"Output shape: {getattr(model, 'output_shape', None)}")

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        print("Skipping dummy forward pass: multi-input model detected.")
        return True, loaded_path
    if not isinstance(input_shape, tuple) or not input_shape:
        print("Skipping dummy forward pass: unknown input shape.")
        return True, loaded_path

    dims = []
    for i, dim in enumerate(input_shape):
        if dim is None:
            dims.append(1)
        elif isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            print("Skipping dummy forward pass: unsupported input shape component.")
            return True, loaded_path

    try:
        dummy = np.random.rand(*dims).astype("float32")
        output = model(dummy, training=False)
        output_shape = getattr(output, "shape", None)
        if isinstance(output, (list, tuple)) and output:
            output_shape = [getattr(item, "shape", None) for item in output]
        print(f"Dummy forward pass OK. Output shape: {output_shape}")
        return True, loaded_path
    except Exception:
        print("WARNING: Model loaded, but dummy forward pass failed.")
        traceback.print_exc()
        return False, loaded_path


def main():
    print("Dog Breed Classifier Project Verifier")
    print(f"Repo root : {REPO_ROOT}")
    print(f"Python    : {sys.version.splitlines()[0]}")
    print(f"Executable: {sys.executable}")

    doctor_exit = run_doctor()
    labels_ok = check_labels()
    model_artifacts = find_model_artifact()

    if not labels_ok:
        return 1

    if not model_artifacts:
        print_section("Summary")
        print("Environment and labels appear OK, but no converted model artifact is present.")
        print("Result: project is not runnable yet in this runtime until a converted model is exported.")
        return 1

    smoke_ok, loaded_artifact = try_load_and_forward(model_artifacts)

    print_section("Summary")
    print(f"doctor.py exit code     : {doctor_exit}")
    print(f"labels check            : {'OK' if labels_ok else 'FAILED'}")
    print(f"model artifacts         : {', '.join(model_artifacts)}")
    print(f"artifact loaded         : {loaded_artifact or 'none'}")
    print(f"artifact smoke test     : {'OK' if smoke_ok else 'FAILED'}")
    return 0 if smoke_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
