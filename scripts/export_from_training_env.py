import ast
import os
import sys
import traceback


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(REPO_ROOT, "my_flask_app", "app.py")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEFAULT_KERAS_PATH = os.path.join(MODELS_DIR, "model.keras")
DEFAULT_SAVED_MODEL_PATH = os.path.join(MODELS_DIR, "saved_model")


def read_model_path_from_app():
    if not os.path.exists(APP_PATH):
        return "my_dog_breed_model.h5"

    with open(APP_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=APP_PATH)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "MODEL_PATH":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    return "my_dog_breed_model.h5"


def resolve_repo_path(path_value):
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(REPO_ROOT, path_value)


def prompt_with_default(prompt_text, default_value):
    try:
        response = input(f"{prompt_text} [{default_value}]: ").strip()
    except EOFError:
        response = ""
    return response or default_value


def prompt_yes_no(prompt_text, default=False):
    suffix = "Y/n" if default else "y/N"
    try:
        response = input(f"{prompt_text} [{suffix}]: ").strip().lower()
    except EOFError:
        response = ""
    if not response:
        return default
    return response in {"y", "yes"}


def main():
    print("Dog Breed Classifier Re-export Helper (run in training environment)")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Python  : {sys.version.splitlines()[0]}")
    print(f"Exe     : {sys.executable}")

    default_source_rel = read_model_path_from_app()
    default_source_abs = resolve_repo_path(default_source_rel)
    print(f"Default source from app.py: {default_source_rel}")
    print(f"Resolved default source  : {default_source_abs}")

    source_input = prompt_with_default("Source model path", default_source_abs or default_source_rel)
    source_path = source_input if os.path.isabs(source_input) else os.path.join(REPO_ROOT, source_input)
    source_path = os.path.abspath(source_path)

    print(f"Using source path       : {source_path}")
    if not os.path.exists(source_path):
        print("ERROR: Source model file/folder not found.")
        print("Run this script in the repository or provide an absolute path to the loadable model.")
        return 1

    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        print("ERROR: TensorFlow is not importable in this environment.")
        traceback.print_exc()
        print("This script must be run in the original training environment (or any environment that can load the source model).")
        return 1

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Target models dir       : {MODELS_DIR}")

    print("Loading source model with compile=False...")
    try:
        model = tf.keras.models.load_model(source_path, compile=False)
    except Exception:
        print("ERROR: Could not load the source model in this environment.")
        traceback.print_exc()
        print("This confirms you must run this script in the environment that CAN load the source model.")
        return 1

    print("Load succeeded.")
    print(f"Model class            : {model.__class__.__name__}")
    print(f"Input shape            : {getattr(model, 'input_shape', 'unknown')}")
    print(f"Output shape           : {getattr(model, 'output_shape', 'unknown')}")

    save_saved_model = prompt_yes_no("Also save TensorFlow SavedModel folder?", default=False)

    try:
        if os.path.exists(DEFAULT_KERAS_PATH):
            print(f"Overwriting existing   : {DEFAULT_KERAS_PATH}")
            if os.path.isdir(DEFAULT_KERAS_PATH):
                raise RuntimeError(f"Expected file path for .keras artifact, found directory: {DEFAULT_KERAS_PATH}")
        print(f"Saving .keras artifact : {DEFAULT_KERAS_PATH}")
        model.save(DEFAULT_KERAS_PATH)
        print("Saved .keras artifact  : SUCCESS")
    except Exception:
        print("ERROR: Failed to save models/model.keras")
        traceback.print_exc()
        return 1

    if save_saved_model:
        try:
            print(f"Saving SavedModel      : {DEFAULT_SAVED_MODEL_PATH}")
            model.save(DEFAULT_SAVED_MODEL_PATH, save_format='tf')
            print("Saved SavedModel       : SUCCESS")
        except Exception:
            print("ERROR: Failed to save models/saved_model")
            traceback.print_exc()
            return 1

    print("Re-export complete.")
    print("Next steps in this repo:")
    print("- python scripts/verify_project.py")
    print("- python my_flask_app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

