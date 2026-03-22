import ast
import inspect
import os
import sys
import traceback


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(REPO_ROOT, "my_flask_app", "app.py")


def read_model_path_from_app():
    if not os.path.exists(APP_PATH):
        return None

    with open(APP_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=APP_PATH)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "MODEL_PATH":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    return None


def resolve_repo_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(REPO_ROOT, path_value)


def print_runtime():
    print("Runtime")
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Exe   : {sys.executable}")


def list_candidates(tf, keras_module):
    print("\nMixed precision candidates")

    tf_mp = getattr(tf.keras, "mixed_precision", None)
    print(f"tf.keras.mixed_precision exists: {tf_mp is not None}")
    if tf_mp is not None:
        names = [name for name in dir(tf_mp) if "Policy" in name]
        print(f"tf.keras.mixed_precision policy-like names: {names}")
        for name in names:
            obj = getattr(tf_mp, name, None)
            if obj is not None:
                print(f"  - {name}: {obj}")

    keras_mp = getattr(keras_module, "mixed_precision", None)
    print(f"keras.mixed_precision exists   : {keras_mp is not None}")
    if keras_mp is not None:
        names = [name for name in dir(keras_mp) if "Policy" in name]
        print(f"keras.mixed_precision policy-like names: {names}")
        for name in names:
            obj = getattr(keras_mp, name, None)
            if obj is not None:
                print(f"  - {name}: {obj}")


def get_policy_candidates(tf, keras_module):
    candidates = []

    for label, obj in (
        ("tf.keras.mixed_precision.Policy", getattr(getattr(tf.keras, "mixed_precision", None), "Policy", None)),
        ("keras.mixed_precision.Policy", getattr(getattr(keras_module, "mixed_precision", None), "Policy", None)),
    ):
        if obj is not None:
            candidates.append((label, obj))

    # Deduplicate by object identity.
    seen_ids = set()
    unique = []
    for label, obj in candidates:
        if id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))
        unique.append((label, obj))
    return unique


def try_normal_load(tf, model_path):
    print("\nAttempt 1: normal tf.keras load")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("SUCCESS (normal load)")
        print(f"Model class : {model.__class__.__name__}")
        print(f"Input shape : {getattr(model, 'input_shape', None)}")
        print(f"Output shape: {getattr(model, 'output_shape', None)}")
        return True
    except Exception:
        print("FAILED (normal load)")
        traceback.print_exc()
        return False


def try_custom_objects_load(tf, keras_module, model_path):
    print("\nAttempt 2+: custom_objects / custom_object_scope for DTypePolicy")
    candidates = get_policy_candidates(tf, keras_module)
    if not candidates:
        print("No Policy class candidates found in this runtime. Cannot build DTypePolicy shim.")
        return False

    for label, policy_class in candidates:
        print(f"\nTrying candidate: {label}")
        try:
            print(f"Policy class signature: {inspect.signature(policy_class)}")
        except Exception:
            pass

        custom_objects = {"DTypePolicy": policy_class}

        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
            )
            print("SUCCESS via custom_objects argument")
            print(f"Model class : {model.__class__.__name__}")
            print(f"Input shape : {getattr(model, 'input_shape', None)}")
            print(f"Output shape: {getattr(model, 'output_shape', None)}")
            return True
        except Exception:
            print("FAILED via custom_objects argument")
            traceback.print_exc()

        try:
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(model_path, compile=False)
            print("SUCCESS via custom_object_scope")
            print(f"Model class : {model.__class__.__name__}")
            print(f"Input shape : {getattr(model, 'input_shape', None)}")
            print(f"Output shape: {getattr(model, 'output_shape', None)}")
            return True
        except Exception:
            print("FAILED via custom_object_scope")
            traceback.print_exc()

    return False


def main():
    try:
        import tensorflow as tf  # type: ignore
        import keras  # type: ignore
    except Exception:
        print("TensorFlow/Keras import failed.")
        traceback.print_exc()
        return 1

    model_rel = read_model_path_from_app()
    model_path = resolve_repo_path(model_rel)

    print("DTypePolicy Shim Load Probe")
    print(f"Repo root      : {REPO_ROOT}")
    print_runtime()
    print(f"TensorFlow/Keras: {tf.__version__} / {keras.__version__}")
    print(f"MODEL_PATH      : {model_rel}")
    print(f"Resolved path   : {model_path}")

    if not model_path or not os.path.exists(model_path):
        print("Model file not found.")
        return 1

    list_candidates(tf, keras)

    if try_normal_load(tf, model_path):
        print("\nSummary: model already loads normally; no DTypePolicy shim needed.")
        return 0

    if try_custom_objects_load(tf, keras, model_path):
        print("\nSummary: DTypePolicy shim load succeeded.")
        return 0

    print("\nSummary: DTypePolicy shim load failed in this runtime.")
    print("Next step: export models/model.keras or models/saved_model from the original training environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
