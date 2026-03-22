import ast
import json
import os
import shutil
import sys
import traceback

import h5py


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


def backup_path_for(model_path):
    root, ext = os.path.splitext(model_path)
    if ext.lower() == ".h5":
        return f"{root}.backup{ext}"
    return f"{model_path}.backup"


def walk_input_layers(config_obj):
    stack = [config_obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("class_name") == "InputLayer" and isinstance(item.get("config"), dict):
                yield item["config"]
            for value in item.values():
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)


def patch_model_config_text(model_config_text):
    config_obj = json.loads(model_config_text)
    input_layer_count = 0
    patched_count = 0

    for layer_config in walk_input_layers(config_obj):
        input_layer_count += 1
        has_batch_shape = "batch_shape" in layer_config
        has_batch_input_shape = "batch_input_shape" in layer_config
        if has_batch_shape and not has_batch_input_shape:
            layer_config["batch_input_shape"] = layer_config.pop("batch_shape")
            patched_count += 1

    return json.dumps(config_obj), input_layer_count, patched_count


def main():
    print("Dog Breed Classifier H5 Patch Tool")
    print(f"Repo root: {REPO_ROOT}")
    print("Scope: fixes only InputLayer config key rename 'batch_shape' -> 'batch_input_shape'.")
    print("This script does NOT fix newer Keras dtype-policy serialization issues (e.g. DTypePolicy).")

    rel_model_path = read_model_path_from_app()
    model_path = resolve_repo_path(rel_model_path)
    print(f"MODEL_PATH from app.py: {rel_model_path}")
    print(f"Resolved model path   : {model_path}")

    if not model_path or not os.path.exists(model_path):
        print("ERROR: Model file not found.")
        print("Next step: verify MODEL_PATH in my_flask_app/app.py")
        return 1

    backup_path = backup_path_for(model_path)
    print(f"Backup path           : {backup_path}")

    try:
        with h5py.File(model_path, "r") as h5f:
            if "model_config" not in h5f.attrs:
                print("ERROR: This H5 does not contain model_config (likely weights-only or non-Keras H5).")
                print("Patching is not applicable.")
                return 1

            raw_model_config = h5f.attrs["model_config"]
            raw_was_bytes = isinstance(raw_model_config, (bytes, bytearray))
            model_config_text = (
                raw_model_config.decode("utf-8") if raw_was_bytes else raw_model_config
            )

            patched_text, input_layers, patched_layers = patch_model_config_text(model_config_text)

            print(f"InputLayer count      : {input_layers}")
            print(f"Layers to patch       : {patched_layers}")

            if patched_layers == 0:
                print("No applicable batch_shape -> batch_input_shape changes found.")
                print("If your error mentions DTypePolicy, re-export to models/model.keras or models/saved_model from the training environment.")
                print("No file changes were made.")
                return 0

        if not os.path.exists(backup_path):
            shutil.copy2(model_path, backup_path)
            print("Backup created        : yes")
        else:
            print("Backup created        : already exists (reusing)")

        with h5py.File(model_path, "r+") as h5f:
            to_write = patched_text.encode("utf-8") if raw_was_bytes else patched_text
            h5f.attrs.modify("model_config", to_write)

        print("Patch applied         : SUCCESS")
        print(f"Patched InputLayers   : {patched_layers}")
        return 0
    except Exception:
        print("Patch applied         : FAILED")
        traceback.print_exc()
        print("Next steps:")
        print("- Restore backup if needed.")
        print("- Run: python scripts/doctor.py")
        print("- If still incompatible, re-export from the original training environment.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
