import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename # For secure file naming

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Folder to temporarily save uploaded images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Machine Learning Model Loading ---
MODEL_PATH = 'waste_segregation_model.h5' # Path to your saved Keras model

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure 'waste_segregation_model.h5' is in the same directory as app.py")
    # You might want to exit or raise an error here in a production environment
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, maybe exit the app
    exit()


# --- Model Parameters (MUST match your training setup) ---
# These class names MUST be in the exact same order as your model was trained on.
# If you used the Kaggle 'garbage_classification' dataset and sorted directories alphabetically,
# this order should be correct. Double-check this against your Jupyter Notebook's CLASS_NAMES.
CLASS_NAMES = sorted([
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
])
IMG_HEIGHT = 224 # Image height expected by MobileNetV2
IMG_WIDTH = 224  # Image width expected by MobileNetV2

# --- Helper Functions ---

def preprocess_image(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Loads an image, resizes it, and preprocesses it for model prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize pixel values to [0, 1]
    return img_array

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page for image upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and returns prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the uploaded image
            processed_image = preprocess_image(filepath)

            # Make a prediction using the loaded model
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index]) * 100 # Convert to percentage

            # Prepare results to send back to the frontend
            result = {
                'prediction': predicted_class_name,
                'confidence': f"{confidence:.2f}%",
                'image_url': f"/{filepath}" # URL to display the uploaded image
            }
            return jsonify(result)

        except Exception as e:
            # General error handling during prediction
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
        finally:
            # Clean up: remove the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({'error': 'An unexpected error occurred.'}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Run in debug mode for development. Set debug=False for production.
    app.run(debug=True, port=5000)
