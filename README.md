# Dog-breed-classifier
Dog Breed Classifier is a simple Flask web app that uses a fine-tuned MobileNetV2 model to identify dog breeds from images. Upload a dog photo, and the app displays the predicted breed and confidence. It’s built with Python 3.11, TensorFlow, and OpenCV, featuring data augmentation and transfer learning for higher accuracy.




Dog Breed Classifier - Instructions
===================================

1. Prerequisites:
   - Python 3.11 (or higher).
   - Installed packages: Flask, TensorFlow (or tf.keras), OpenCV, NumPy, etc.
     Make sure to install them with pip:
       pip install flask tensorflow opencv-python numpy pandas scikit-learn

2. Folder Structure:
   my_flask_app/
   ├── app.py
   ├── templates/
   │   └── index.html
   └── static/
       ├── css/
       │   └── styles.css
       └── js/
           └── script.js

3. Dataset:
   - Download the dataset (e.g., from Kaggle or another source).
   - Place the downloaded ZIP file in the **Dog-breed-classifier** folder.
   - Extract the contents right there in the same folder so that your code can access `train/` and `test/` (or however your dataset is structured).

4. Model File:
   - Place your trained model file (e.g., my_dog_breed_model.h5) in the same folder as app.py.
   - Ensure your index_to_class.json mapping is also in the same folder if required by the code.

5. How to Run:
   - Open a terminal/command prompt.
   - Navigate to the my_flask_app folder:
       cd path/to/my_flask_app
   - Run the Flask app:
       python app.py
   - Flask will start a local server on http://127.0.0.1:5000

6. How to Use:
   - Open your web browser and go to http://127.0.0.1:5000
   - Upload an image of a dog, then click "Predict".
   - The predicted breed and confidence will be displayed.

7. Notes:
   - This application is meant for local use only (development mode).
   - For production deployment, use a production-grade WSGI server.

Enjoy classifying dog breeds!
