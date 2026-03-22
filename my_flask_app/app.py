import json
import os

import tensorflow as tf
from flask import Flask, jsonify, render_template
from tensorflow.keras.models import load_model

try:
    from flask_cors import CORS
except ImportError:  # Optional at runtime until requirements are installed
    CORS = None

try:
    from routes.predict import register_predict_routes
except ImportError:
    from my_flask_app.routes.predict import register_predict_routes


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = "my_dog_breed_model.h5"
CLASS_INDEX_PATH = "index_to_class.json"
MODELS_DIR = os.path.join(REPO_ROOT, "models")
UPLOAD_DIR = os.path.join(REPO_ROOT, "Dog_breed_model")


def _repo_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(REPO_ROOT, path_value)


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default=False):
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _build_runtime_settings():
    allowed_extensions = tuple(
        sorted(
            {
                item.strip().lower().lstrip(".")
                for item in os.getenv(
                    "DOG_BREED_ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp,bmp"
                ).split(",")
                if item.strip()
            }
        )
    )
    if not allowed_extensions:
        allowed_extensions = ("jpg", "jpeg", "png")

    return {
        "MODEL_PATH": os.getenv("DOG_BREED_MODEL_PATH", MODEL_PATH),
        "CLASS_INDEX_PATH": os.getenv("DOG_BREED_CLASS_INDEX_PATH", CLASS_INDEX_PATH),
        "MODELS_DIR": os.getenv("DOG_BREED_MODELS_DIR", MODELS_DIR),
        "UPLOAD_DIR": os.getenv("DOG_BREED_UPLOAD_DIR", UPLOAD_DIR),
        "IMAGE_SIZE": _as_int(os.getenv("DOG_BREED_IMAGE_SIZE"), 224),
        "IMAGE_SIZE_OVERRIDE": os.getenv("DOG_BREED_IMAGE_SIZE") is not None,
        "TOP_K": max(1, _as_int(os.getenv("DOG_BREED_TOP_K"), 3)),
        "TTA_STEPS": max(1, min(5, _as_int(os.getenv("DOG_BREED_TTA_STEPS"), 3))),
        "MAX_UPLOAD_MB": max(1, _as_int(os.getenv("DOG_BREED_MAX_UPLOAD_MB"), 8)),
        "ALLOWED_EXTENSIONS": allowed_extensions,
        "INPUT_SCALING_OVERRIDE": os.getenv("DOG_BREED_INPUT_SCALING", "").strip(),
        "ALLOW_STALE_MODEL_FALLBACK": _as_bool(
            os.getenv("DOG_BREED_ALLOW_STALE_MODEL_FALLBACK"), False
        ),
    }


def _compact_error_text(exc, max_chars=280):
    raw = " ".join(str(exc).strip().split())
    if not raw:
        return exc.__class__.__name__
    if len(raw) <= max_chars:
        return raw
    return f"{raw[: max_chars - 3]}..."


def _model_has_rescaling_layer(model):
    visited = set()
    stack = [model]

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        for layer in getattr(current, "layers", []):
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
            nested_layers = getattr(layer, "layers", None)
            if nested_layers:
                stack.append(layer)
    return False


def _resolve_input_scaling(model, scaling_override):
    normalized_override = str(scaling_override or "").strip().lower()
    if normalized_override in {"normalize_0_1", "model_preprocessing"}:
        return normalized_override

    if _model_has_rescaling_layer(model):
        return "model_preprocessing"

    has_predict = hasattr(model, "predict")
    has_signatures = bool(getattr(model, "signatures", None))
    if has_signatures and not has_predict:
        # Keras 3 `model.export()` artifacts load as signature-only `_UserObject`
        # in TF/Keras 2.12. Those typically contain internal preprocessing.
        return "model_preprocessing"

    return "normalize_0_1"


def _extract_spatial_size_from_shape(shape):
    if not shape or len(shape) < 3:
        return None
    height = shape[1]
    width = shape[2]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return int(height)
    return None


def _resolve_image_size(model, configured_size):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]
    resolved = _extract_spatial_size_from_shape(input_shape)
    if resolved:
        return resolved

    signatures = getattr(model, "signatures", None) or {}
    infer_fn = signatures.get("serving_default") or next(iter(signatures.values()), None)
    if infer_fn is not None:
        try:
            _args, kwargs = infer_fn.structured_input_signature
            first_spec = next(iter(kwargs.values()), None)
            if first_spec is not None:
                shape = tuple(first_spec.shape.as_list())
                resolved = _extract_spatial_size_from_shape(shape)
                if resolved:
                    return resolved
        except Exception:
            pass

    return configured_size


def load_model_safe(model_path, allow_stale_model_fallback=False):
    primary_path = _repo_path(model_path)
    candidate_paths = [primary_path]

    if primary_path.lower().endswith(".h5"):
        candidate_paths = [
            _repo_path(os.path.join("models", "model.keras")),
            _repo_path(os.path.join("models", "saved_model")),
            primary_path,
        ]

    def _path_priority(path_value):
        lowered = str(path_value).lower()
        if lowered.endswith("model.keras"):
            return 0
        if lowered.endswith("saved_model"):
            return 1
        if lowered.endswith(".h5"):
            return 2
        return 3

    existing_candidates = [
        path
        for path in candidate_paths
        if os.path.exists(path)
    ]
    candidate_paths = sorted(
        candidate_paths,
        key=lambda p: (
            -os.path.getmtime(p) if os.path.exists(p) else float("-inf"),
            _path_priority(p),
        ),
    )

    tried = []
    errors = []
    seen = set()
    failed_mtime_by_path = {}

    for candidate_path in candidate_paths:
        if candidate_path in seen:
            continue
        seen.add(candidate_path)
        tried.append(candidate_path)

        if not os.path.exists(candidate_path):
            errors.append(f"{candidate_path} (missing)")
            continue

        try:
            loaded_model = load_model(candidate_path, compile=False)

            if existing_candidates and not allow_stale_model_fallback:
                loaded_mtime = os.path.getmtime(candidate_path)
                newer_failed_paths = [
                    path
                    for path, _failed_mtime in failed_mtime_by_path.items()
                    if _failed_mtime > loaded_mtime + 1.0
                ]
                if newer_failed_paths:
                    newer_paths_text = "\n".join(f"- {path}" for path in newer_failed_paths)
                    raise RuntimeError(
                        "Refusing to serve a stale fallback model. "
                        "A newer model artifact exists but failed to load in this runtime.\n"
                        f"Loaded candidate: {candidate_path}\n"
                        "Newer failed artifacts:\n"
                        f"{newer_paths_text}\n"
                        "Fix by exporting a runtime-compatible artifact from the latest model "
                        "(for Keras 3, export/update models/saved_model from models/model.keras), "
                        "or set DOG_BREED_ALLOW_STALE_MODEL_FALLBACK=true to bypass this check."
                    )
            return loaded_model, candidate_path
        except Exception as exc:
            compact = _compact_error_text(exc)
            errors.append(f"{candidate_path} ({exc.__class__.__name__}: {compact})")
            if os.path.exists(candidate_path):
                failed_mtime_by_path[candidate_path] = os.path.getmtime(candidate_path)

    message_lines = [
        "Failed to load dog breed model.",
        "Paths tried:",
        *[f"- {path}" for path in tried],
        "Error summary:",
        *[f"- {err}" for err in errors],
        f"Runtime: TensorFlow {getattr(tf, '__version__', 'unknown')} / Keras {getattr(tf.keras, '__version__', 'unknown')}",
    ]

    combined_errors = "\n".join(errors)
    if "keras.src.models.functional" in combined_errors:
        message_lines.extend(
            [
                "Detected incompatibility: models/model.keras was likely exported by Keras 3 and cannot be loaded by TensorFlow/Keras 2.12.",
                "Re-export the model using the same runtime as this app (TensorFlow/Keras 2.12),",
                "or export a compatible SavedModel from the original training environment.",
            ]
        )
    if "DTypePolicy" in combined_errors:
        message_lines.extend(
            [
                "Detected incompatibility: this .h5 model uses newer Keras serialization (DTypePolicy) than this runtime can deserialize.",
                "Provide a converted artifact exported from the original training environment:",
                "- models/model.keras (preferred)",
                "- models/saved_model",
            ]
        )
    if "batch_shape" in combined_errors:
        message_lines.append(
            "Detected incompatibility: legacy H5 uses InputLayer `batch_shape`; TensorFlow/Keras 2.12 cannot deserialize this artifact directly."
        )

    message_lines.extend(
        [
            "Next steps:",
            "- Run: python scripts/doctor.py",
            "- If the H5 still fails after patching, create/export a converted artifact:",
            "- Run: python scripts/convert_model.py",
            "- Run in the original training environment: python scripts/export_from_training_env.py",
            "- If conversion fails, re-export the model from the original training environment.",
        ]
    )
    raise RuntimeError("\n".join(message_lines))


def _load_class_index(index_path):
    with open(_repo_path(index_path), "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("index_to_class.json must contain a JSON object.")
    return data


def create_app():
    settings = _build_runtime_settings()
    app = Flask(__name__)

    models_dir = _repo_path(settings["MODELS_DIR"])
    upload_dir = _repo_path(settings["UPLOAD_DIR"])
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)

    app.config["DOG_BREED_MODEL_PATH"] = settings["MODEL_PATH"]
    app.config["DOG_BREED_CLASS_INDEX_PATH"] = settings["CLASS_INDEX_PATH"]
    app.config["DOG_BREED_MODELS_DIR"] = models_dir
    app.config["DOG_BREED_UPLOAD_DIR"] = upload_dir

    model, model_source = load_model_safe(
        settings["MODEL_PATH"],
        allow_stale_model_fallback=settings["ALLOW_STALE_MODEL_FALLBACK"],
    )
    class_index = _load_class_index(settings["CLASS_INDEX_PATH"])

    resolved_image_size = settings["IMAGE_SIZE"]
    if not settings["IMAGE_SIZE_OVERRIDE"]:
        resolved_image_size = _resolve_image_size(model, settings["IMAGE_SIZE"])

    app.config["DOG_BREED_IMAGE_SIZE"] = resolved_image_size
    app.config["DOG_BREED_TOP_K"] = settings["TOP_K"]
    app.config["DOG_BREED_TTA_STEPS"] = settings["TTA_STEPS"]
    app.config["DOG_BREED_ALLOWED_EXTENSIONS"] = settings["ALLOWED_EXTENSIONS"]
    app.config["MAX_CONTENT_LENGTH"] = settings["MAX_UPLOAD_MB"] * 1024 * 1024
    app.config["DOG_BREED_INPUT_SCALING_OVERRIDE"] = settings["INPUT_SCALING_OVERRIDE"]
    app.config["DOG_BREED_ALLOW_STALE_MODEL_FALLBACK"] = settings[
        "ALLOW_STALE_MODEL_FALLBACK"
    ]

    app.config["DOG_BREED_MODEL"] = model
    app.config["DOG_BREED_MODEL_SOURCE"] = model_source
    app.config["DOG_BREED_CLASS_INDEX"] = class_index
    app.config["DOG_BREED_INPUT_SCALING"] = _resolve_input_scaling(
        model, settings["INPUT_SCALING_OVERRIDE"]
    )

    if CORS is not None:
        CORS(app)

    register_predict_routes(app)

    @app.errorhandler(413)
    def file_too_large(_error):
        limit_mb = settings["MAX_UPLOAD_MB"]
        return jsonify({"error": f"File is too large. Maximum upload size is {limit_mb}MB."}), 413

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_loaded": True,
                "model_source": app.config.get("DOG_BREED_MODEL_SOURCE"),
                "class_count": len(app.config.get("DOG_BREED_CLASS_INDEX", {})),
                "image_size": app.config.get("DOG_BREED_IMAGE_SIZE", 224),
                "input_scaling": app.config.get("DOG_BREED_INPUT_SCALING", "unknown"),
                "tta_steps": app.config.get("DOG_BREED_TTA_STEPS", 1),
                "allow_stale_model_fallback": app.config.get(
                    "DOG_BREED_ALLOW_STALE_MODEL_FALLBACK", False
                ),
            }
        )

    return app


try:
    app = create_app()
except Exception as exc:
    if __name__ == "__main__":
        print(str(exc))
        raise SystemExit(1)
    raise


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    port = _as_int(os.getenv("PORT"), 5000)
    print("Dog Breed App Startup")
    print(f"- Python executable : {os.sys.executable}")
    print(f"- Model source      : {app.config.get('DOG_BREED_MODEL_SOURCE')}")
    print(f"- Input scaling     : {app.config.get('DOG_BREED_INPUT_SCALING')}")
    print(f"- Port              : {port}")
    print(f"- Debug             : {debug_mode}")
    app.run(debug=debug_mode, port=port)
