import os
import json
import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = "my_dog_breed_model.h5"
CLASS_INDEX_PATH = "index_to_class.json"
model = load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH, "r") as f:
    index_to_class = json.load(f)

def predict_single_image(model, image_path, img_size=224):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img)
    predicted_class_index = np.argmax(preds[0])
    predicted_label = index_to_class[str(predicted_class_index)]
    confidence = preds[0][predicted_class_index]
    return predicted_label, float(confidence)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    filepath = os.path.join("Dog_breed_model", file.filename)
    file.save(filepath)
    
    label, conf = predict_single_image(model, filepath)
    
    os.remove(filepath)
    
    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{conf*100:.2f}"
    )

if __name__ == "__main__":
    os.makedirs("Dog_breed_model", exist_ok=True)
    app.run(debug=True, port=5000)
