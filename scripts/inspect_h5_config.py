import ast
import json
import os
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


def read_model_config(h5f):
    if "model_config" in h5f.attrs:
        raw = h5f.attrs["model_config"]
        source = "attribute"
    elif "model_config" in h5f:
        raw = h5f["model_config"][()]
        source = "dataset"
    else:
        return None, None, None

    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8")
    else:
        text = raw
    if isinstance(text, str):
        return text, source, "str"
    return None, source, type(text).__name__


def find_layers(config_obj):
    stack = [config_obj]
    layers = []
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "class_name" in item and "config" in item:
                layers.append(item)
            for value in item.values():
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return layers


def main():
    model_rel_path = read_model_path_from_app()
    model_path = resolve_repo_path(model_rel_path)

    print("H5 Model Config Inspector")
    print(f"Repo root          : {REPO_ROOT}")
    print(f"MODEL_PATH (app.py): {model_rel_path}")
    print(f"Resolved model path: {model_path}")

    if not model_path or not os.path.exists(model_path):
        print("Model file not found.")
        return 1

    try:
        with h5py.File(model_path, "r") as h5f:
            model_config_text, source, text_type = read_model_config(h5f)
            print(f"model_config exists : {model_config_text is not None}")
            print(f"model_config source : {source}")
            print(f"model_config type   : {text_type}")

            if model_config_text is None:
                print("No model_config found (weights-only or non-Keras H5).")
                return 0

            config_obj = json.loads(model_config_text)
            print(f"Top-level keys      : {sorted(list(config_obj.keys())) if isinstance(config_obj, dict) else 'not a dict'}")

            layers = []
            if isinstance(config_obj, dict) and isinstance(config_obj.get("config"), dict):
                maybe_layers = config_obj["config"].get("layers")
                if isinstance(maybe_layers, list):
                    layers = maybe_layers

            all_layer_objects = find_layers(config_obj)
            print(f"layers (top-level)  : {len(layers)}")
            print(f"layer objects found : {len(all_layer_objects)}")

            dtype_policy_count = 0
            preview_count = 0
            for layer in layers[:10]:
                class_name = layer.get("class_name")
                layer_cfg = layer.get("config") if isinstance(layer.get("config"), dict) else {}
                dtype_cfg = layer_cfg.get("dtype")
                has_dtype_policy = isinstance(dtype_cfg, dict) and (
                    dtype_cfg.get("class_name") == "DTypePolicy" or "DTypePolicy" in json.dumps(dtype_cfg)
                )
                if has_dtype_policy:
                    dtype_policy_count += 1
                preview_count += 1
                print(
                    f"Layer[{preview_count}] class={class_name} "
                    f"dtype_policy={has_dtype_policy} "
                    f"dtype={dtype_cfg if isinstance(dtype_cfg, (dict, str)) else type(dtype_cfg).__name__}"
                )

            for layer in layers[10:]:
                layer_cfg = layer.get("config") if isinstance(layer.get("config"), dict) else {}
                dtype_cfg = layer_cfg.get("dtype")
                has_dtype_policy = isinstance(dtype_cfg, dict) and (
                    dtype_cfg.get("class_name") == "DTypePolicy" or "DTypePolicy" in json.dumps(dtype_cfg)
                )
                if has_dtype_policy:
                    dtype_policy_count += 1

            raw_hits = {
                "DTypePolicy": model_config_text.count("DTypePolicy"),
                "dtype_policy": model_config_text.count("dtype_policy"),
                "keras.DTypePolicy": model_config_text.count("keras.DTypePolicy"),
            }

            print("\nSummary")
            print(f"Layers referencing DTypePolicy : {dtype_policy_count}")
            for key, count in raw_hits.items():
                print(f"Raw JSON hits '{key}': {count}")

        return 0
    except Exception:
        print("Inspection failed.")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

