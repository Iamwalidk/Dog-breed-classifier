import ast
import json
import os
import platform
import subprocess
import sys
import traceback


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(REPO_ROOT, "my_flask_app", "app.py")


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


def format_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def get_pip_version():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or result.stderr).strip()
        return output or "pip version unavailable"
    except Exception as exc:
        return f"Failed to query pip version: {exc}"


def print_runtime_info():
    print_section("Runtime")
    print(f"Python version : {sys.version.splitlines()[0]}")
    print(f"Platform       : {platform.platform()}")
    print(f"Executable     : {sys.executable}")
    print(f"Pip            : {get_pip_version()}")


def import_optional_modules():
    print_section("Packages")
    modules = {
        "tensorflow": None,
        "numpy": None,
        "cv2": None,
    }

    for name in list(modules):
        try:
            modules[name] = __import__(name)
            version = getattr(modules[name], "__version__", "unknown")
            print(f"{name:<12} : {version}")
        except Exception:
            print(f"{name:<12} : IMPORT FAILED")
            traceback.print_exc()

    tf_module = modules.get("tensorflow")
    keras_version = None
    if tf_module is not None:
        keras_version = getattr(getattr(tf_module, "keras", None), "__version__", None)

    if keras_version is None:
        try:
            import keras  # type: ignore

            keras_version = getattr(keras, "__version__", "unknown")
        except Exception:
            keras_version = None

    print(f"{'keras':<12} : {keras_version or 'unavailable'}")
    return modules


def print_tf_gpu_info(tf_module):
    print_section("TensorFlow Devices")
    if tf_module is None:
        print("TensorFlow not available; cannot list GPUs.")
        return

    try:
        gpus = tf_module.config.list_physical_devices("GPU")
        if not gpus:
            print("No GPUs visible to TensorFlow.")
        else:
            print(f"GPUs visible to TensorFlow: {len(gpus)}")
            for gpu in gpus:
                print(f"- {gpu}")
    except Exception:
        print("Failed to list TensorFlow devices.")
        traceback.print_exc()


def inspect_project_files():
    print_section("Project Files")
    constants = read_app_constants()
    if not constants:
        print(f"Could not parse constants from {APP_PATH}")
        return constants, None, None

    model_rel = constants.get("MODEL_PATH")
    class_rel = constants.get("CLASS_INDEX_PATH")
    model_abs = repo_path(model_rel)
    class_abs = repo_path(class_rel)

    print(f"MODEL_PATH      : {model_rel}")
    print(f"Resolved model  : {model_abs}")
    print(f"CLASS_INDEX_PATH: {class_rel}")
    print(f"Resolved labels : {class_abs}")

    if model_abs and os.path.exists(model_abs):
        print(f"Model exists    : yes ({format_size(os.path.getsize(model_abs))})")
    else:
        print("Model exists    : no")

    if class_abs and os.path.exists(class_abs):
        try:
            with open(class_abs, "r", encoding="utf-8") as f:
                labels = json.load(f)
            label_count = len(labels) if isinstance(labels, dict) else "unknown"
            print(f"Labels file     : yes ({label_count} classes)")
        except Exception:
            print("Labels file     : yes (failed to parse)")
            traceback.print_exc()
    else:
        print("Labels file     : no")

    return constants, model_abs, class_abs


def dry_run_model_load(tf_module, model_path):
    print_section("Model Load Dry-Run")
    if tf_module is None:
        print("Skipped: TensorFlow is not importable.")
        print("Next step: pip install -r requirements.txt")
        return 1

    if not model_path or not os.path.exists(model_path):
        print(f"Skipped: model path not found: {model_path}")
        print("Next step: confirm MODEL_PATH in my_flask_app/app.py and the model file location.")
        return 1

    print(f"Attempting load_model(..., compile=False) from:\n{model_path}")
    try:
        model = tf_module.keras.models.load_model(model_path, compile=False)
        print("Model load      : SUCCESS")
        print(f"Model class     : {model.__class__.__name__}")
        print(f"Input shape     : {getattr(model, 'input_shape', 'unknown')}")
        print(f"Output shape    : {getattr(model, 'output_shape', 'unknown')}")
        return 0
    except Exception as exc:
        print("Model load      : FAILED")
        traceback.print_exc()
        print("\nActionable hints:")
        print("- Run: python scripts/inspect_h5_config.py")
        print("- Run: python scripts/convert_model.py")
        if "DTypePolicy" in str(exc):
            print("- This H5 uses newer Keras serialization (DTypePolicy); export models/model.keras or models/saved_model from the original training environment.")
            print("- Helper script to run there: python scripts/export_from_training_env.py")
        elif "batch_shape" in str(exc):
            print("- This may be the legacy InputLayer batch_shape issue; run: python scripts/patch_h5.py")
        else:
            print("- If the error mentions InputLayer/batch_shape or legacy H5 config, re-export the model from the original training environment.")
        print("- Confirm TensorFlow/NumPy/OpenCV versions match requirements.txt.")
        return 1


def main():
    print("Dog Breed Classifier Environment Doctor")
    print(f"Repo root: {REPO_ROOT}")

    print_runtime_info()
    modules = import_optional_modules()
    print_tf_gpu_info(modules.get("tensorflow"))
    _, model_path, _ = inspect_project_files()
    return dry_run_model_load(modules.get("tensorflow"), model_path)


if __name__ == "__main__":
    raise SystemExit(main())
