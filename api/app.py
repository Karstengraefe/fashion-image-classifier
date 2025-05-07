
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
import base64
import joblib

app = Flask(__name__)

# Load model and define class labels
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/fashion_model.h5')
model = load_model(MODEL_PATH)

# Load label encoder
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), '../model/label_encoder.pkl')
label_encoder = joblib.load(LABEL_ENCODER_PATH)
class_labels = list(label_encoder.classes_)

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img_tensor = preprocess_image(img_bytes)
    predictions = model.predict(img_tensor)[0]
    percentages = predictions / np.sum(predictions) * 100
    result = {label: f"{prob:.2f}%" for label, prob in zip(class_labels, percentages)} 
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
