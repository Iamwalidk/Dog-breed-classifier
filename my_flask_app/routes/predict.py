import json
import os
import re
import time
import uuid
from functools import lru_cache

import numpy as np
import tensorflow as tf
from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

predict_bp = Blueprint("predict_bp", __name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BREEDS_INFO_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "breeds_info.json")
INFO_FIELDS = ("description", "temperament", "size", "life_span")


def register_predict_routes(app):
    app.register_blueprint(predict_bp)


def _normalize_breed_key(value):
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _format_breed_name(raw_label):
    return raw_label.replace("_", " ").title()


def _empty_breed_info():
    return {field: None for field in INFO_FIELDS}


def _sanitize_breed_info(data):
    info = _empty_breed_info()
    if isinstance(data, dict):
        for field in INFO_FIELDS:
            value = data.get(field)
            info[field] = value if isinstance(value, str) and value.strip() else None
    return info


@lru_cache(maxsize=1)
def _load_breeds_info_lookup():
    if not os.path.exists(BREEDS_INFO_PATH):
        return {}

    try:
        with open(BREEDS_INFO_PATH, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    lookup = {}
    if isinstance(raw_data, dict):
        for breed_name, info in raw_data.items():
            lookup[_normalize_breed_key(breed_name)] = _sanitize_breed_info(info)
    return lookup


def _get_breed_info(raw_label, display_breed):
    info_lookup = _load_breeds_info_lookup()
    for key_candidate in (display_breed, raw_label, raw_label.replace("_", " ")):
        normalized = _normalize_breed_key(key_candidate)
        if normalized in info_lookup:
            return info_lookup[normalized]
    return _empty_breed_info()


def _load_rgb_image(image_path, img_size):
    image = tf.keras.utils.load_img(
        image_path,
        color_mode="rgb",
        target_size=(img_size, img_size),
    )
    return tf.keras.utils.img_to_array(image, dtype="float32")


def _center_crop_and_resize(image_array, crop_factor=0.9):
    height, width = image_array.shape[:2]
    crop_h = max(2, int(round(height * crop_factor)))
    crop_w = max(2, int(round(width * crop_factor)))
    offset_y = max(0, (height - crop_h) // 2)
    offset_x = max(0, (width - crop_w) // 2)
    cropped = image_array[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w, :]
    resized = tf.image.resize(cropped, (height, width), antialias=True)
    return resized.numpy().astype("float32")


def _build_tta_batch(image_array, tta_steps):
    views = [image_array]
    if tta_steps >= 2:
        views.append(np.fliplr(image_array).copy())
    if tta_steps >= 3:
        views.append(_center_crop_and_resize(image_array, crop_factor=0.9))
    if tta_steps >= 4:
        views.append(_center_crop_and_resize(image_array, crop_factor=0.85))
    if tta_steps >= 5:
        views.append(np.flipud(image_array).copy())
    return np.stack(views[: max(1, tta_steps)], axis=0).astype("float32")


def _to_probability_vector(preds):
    values = np.asarray(preds, dtype="float32")
    if values.ndim == 0:
        raise ValueError("Model returned a scalar output for classification.")
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2:
        vector = values.mean(axis=0)
    else:
        vector = np.squeeze(values)
        if vector.ndim != 1:
            raise ValueError(f"Unexpected prediction shape: {values.shape}")

    if np.all(np.isfinite(vector)) and np.all(vector >= 0.0):
        total = float(vector.sum())
        if total > 0.0:
            vector = vector / total
            return vector

    vector = tf.nn.softmax(vector).numpy()
    total = float(vector.sum())
    if total > 0.0:
        vector = vector / total
    return vector


def _predict_top_k(model, image_path, index_to_class, img_size=224, top_k=3, tta_steps=1):
    img = _load_rgb_image(image_path, img_size=img_size)
    input_scaling = current_app.config.get("DOG_BREED_INPUT_SCALING", "normalize_0_1")
    if input_scaling == "normalize_0_1":
        img = img / 255.0

    inference_batch = _build_tta_batch(img, tta_steps=tta_steps)
    preds = _run_inference(model, inference_batch)
    probabilities = _to_probability_vector(preds)
    k = min(top_k, len(probabilities))
    top_indices = np.argsort(probabilities)[-k:][::-1]

    predictions = []
    for index in top_indices:
        raw_label = (
            index_to_class.get(str(int(index)))
            or index_to_class.get(int(index))
            or index_to_class.get(str(index))
            or f"class_{int(index)}"
        )
        breed_name = _format_breed_name(raw_label)
        confidence = float(probabilities[index])
        predictions.append(
            {
                "breed": breed_name,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100.0, 2),
                "info": _get_breed_info(raw_label, breed_name),
            }
        )
    return predictions


def _run_inference(model, batch_array):
    if hasattr(model, "predict"):
        return model.predict(batch_array, verbose=0)

    signatures = getattr(model, "signatures", None)
    if not signatures:
        raise TypeError("Loaded model does not expose predict() or SavedModel signatures.")

    infer_fn = signatures.get("serving_default") or next(iter(signatures.values()), None)
    if infer_fn is None:
        raise TypeError("Loaded model signatures are empty.")

    outputs = infer_fn(tf.convert_to_tensor(batch_array))
    if isinstance(outputs, dict):
        if not outputs:
            raise ValueError("SavedModel signature returned an empty output dictionary.")
        first_output = next(iter(outputs.values()))
    elif isinstance(outputs, (list, tuple)):
        if not outputs:
            raise ValueError("SavedModel signature returned an empty output sequence.")
        first_output = outputs[0]
    else:
        first_output = outputs

    if hasattr(first_output, "numpy"):
        return first_output.numpy()
    return np.asarray(first_output)


def _is_allowed_extension(filename):
    extension = os.path.splitext(filename or "")[1].lower().lstrip(".")
    if not extension:
        return False
    allowed = current_app.config.get("DOG_BREED_ALLOWED_EXTENSIONS", ("jpg", "jpeg", "png"))
    return extension in {item.lower() for item in allowed}


def _is_supported_mimetype(file):
    mimetype = (getattr(file, "mimetype", "") or "").lower()
    if not mimetype:
        return True
    return mimetype.startswith("image/")


def _wants_json_response():
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept


def _error_response(message, status_code):
    payload = {"error": message}
    if _wants_json_response():
        return jsonify(payload), status_code
    return render_template("index.html", error=message), status_code


@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return _error_response("No file part", 400)

    file = request.files["file"]
    if file.filename == "":
        return _error_response("No selected file", 400)
    if not _is_allowed_extension(file.filename):
        allowed = ", ".join(current_app.config.get("DOG_BREED_ALLOWED_EXTENSIONS", []))
        return _error_response(f"Unsupported file extension. Allowed: {allowed}", 400)
    if not _is_supported_mimetype(file):
        return _error_response("Unsupported file type. Please upload an image.", 400)

    upload_dir = current_app.config.get("DOG_BREED_UPLOAD_DIR", "Dog_breed_model")
    os.makedirs(upload_dir, exist_ok=True)

    original_name = secure_filename(file.filename)
    extension = os.path.splitext(original_name)[1] or ".jpg"
    filepath = os.path.join(upload_dir, f"{uuid.uuid4().hex}{extension}")

    try:
        file.save(filepath)

        top_k = max(1, min(5, request.args.get("top_k", type=int) or current_app.config.get("DOG_BREED_TOP_K", 3)))
        tta_steps = max(
            1,
            min(
                5,
                request.args.get("tta", type=int)
                or current_app.config.get("DOG_BREED_TTA_STEPS", 1),
            ),
        )
        inference_start = time.perf_counter()
        predictions = _predict_top_k(
            model=current_app.config["DOG_BREED_MODEL"],
            image_path=filepath,
            index_to_class=current_app.config["DOG_BREED_CLASS_INDEX"],
            img_size=current_app.config.get("DOG_BREED_IMAGE_SIZE", 224),
            top_k=top_k,
            tta_steps=tta_steps,
        )
        inference_ms = round((time.perf_counter() - inference_start) * 1000.0, 2)

        payload = {
            "predictions": predictions,
            "meta": {
                "top_k": len(predictions),
                "inference_ms": inference_ms,
                "image_size": current_app.config.get("DOG_BREED_IMAGE_SIZE", 224),
                "model_source": os.path.basename(
                    str(current_app.config.get("DOG_BREED_MODEL_SOURCE", "unknown"))
                ),
                "input_scaling": current_app.config.get("DOG_BREED_INPUT_SCALING", "unknown"),
                "tta_steps": tta_steps,
            },
        }

        if _wants_json_response():
            return jsonify(payload)

        top_prediction = predictions[0] if predictions else None
        return render_template(
            "index.html",
            prediction=top_prediction["breed"] if top_prediction else None,
            confidence=f"{top_prediction['confidence'] * 100:.2f}" if top_prediction else None,
            predictions=predictions,
            prediction_meta=payload.get("meta"),
        )
    except Exception:
        current_app.logger.exception("Prediction failed")
        return _error_response("Prediction failed. Please try again.", 500)
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                current_app.logger.warning("Could not remove temporary file: %s", filepath)
