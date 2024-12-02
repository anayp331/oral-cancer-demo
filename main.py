import numpy as np
import tensorflow as tf
#import pickle
# from scipy import misc
from io import BytesIO
from flask import Flask, request, render_template, jsonify
#from flask_pymongo import PyMongo
from PIL import Image
#import io
import warnings
import gdown
#import os

warnings.filterwarnings('ignore')

# Download the model from Google Drive
model_url = 'https://drive.google.com/uc?id=19YH-elrzYgF5jpust1M3ZvCnXLk5v4rJ'
output = 'final_model.keras'
gdown.download(model_url, output, quiet=False)

try:
    global model
    model = tf.keras.models.load_model(output)
except Exception as e:
    print(f"Error loading model: {e}")

# Define your class labels 
class_labels = ['Moderately-differentiated Oral Squamous Cell Carcinoma', 'Normal', 'Oral Submucous Fibrosis', 'Poorly-differentiated Oral Squamous Cell Carcinoma', 'Well-differentiated Oral Squamous Cell Carcinoma']

app = Flask(__name__)

#app.config['MONGO_DBNAME'] = 'patient_database'
#app.config["MONGO_URI"] = 'mongodb://localhost:27017'

#mongo = PyMongo(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def index():
    return render_template('indexback.html')

@app.route('/predict', methods=['POST'])
def predict():
    #db = mongo.db.patient_database

    # Check if the image is present in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    

    # Get the uploaded image from the request
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Load image and preprocess
    try:
        img = Image.open(BytesIO(file.read()))
        img = img.resize((512, 512))  # Resize to match model's expected input size (512x512)
        img_array = np.array(img)
        
        # If the image has 4 channels (e.g., RGBA), convert to RGB
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]  # Use only the RGB channels
        
        # Add batch dimension and normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0, 1] range
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500
        # Get prediction from the model
    try:
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)  # Get index of the predicted class
        predicted_class_label = class_labels[predicted_class_index[0]]  # Convert index to class label
    except Exception as e:
        return jsonify({"error": f"Failed to make prediction: {e}"}), 500

    # Return the prediction result
    return jsonify({'prediction': predicted_class_label})

@app.route('/prediict', methods=['POST'])
def prediict():
    #db = mongo.db.patient_database

    # Check if the image is present in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    

    # Get the uploaded image from the request
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Load image and preprocess
    try:
        img = Image.open(BytesIO(file.read()))
        img = img.resize((224, 224))  # Resize to match model's expected input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize image
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500

    # Get prediction from the model
    try:
        prediction = model.predict(img_array)
        print(prediction, prediction.shape)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
    except Exception as e:
        return jsonify({"error": f"Failed to make prediction: {e}"}), 500

    # Return the prediction result
    return jsonify({'prediction': 'OralCancer'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
