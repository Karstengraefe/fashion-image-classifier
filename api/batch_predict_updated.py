
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(filename="../prediction_log.csv", level=logging.INFO,
                    format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Load model
model = tf.keras.models.load_model("../model/fashion_model.h5", compile=False)

# Define the image folder
image_folder = "../new_images"

# Function to load and preprocess images
def load_images(image_folder, size=(128, 128)):
    images = []
    image_names = []
    for img_name in os.listdir(image_folder):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(image_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize(size)
                images.append(np.array(img) / 255.0)  # Normalize
                image_names.append(img_name)
            except Exception as e:
                logger.error(f"Error loading image {img_name}: {e}")
    return np.array(images), image_names

# Load and preprocess images
X, image_names = load_images(image_folder)
logger.info(f"Loaded {len(X)} images for prediction.")

# Define the categories (you should modify this if your categories differ)
categories = ['Shirts', 'Watches', 'Handbags', 'Casual Shoes', 'Sports Shoes']

# Function to make predictions
def predict(model, X, categories):
    # Use label encoding
    le = LabelEncoder()
    le.fit(categories)  # Fit the encoder on the known categories

    # Make predictions
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = le.inverse_transform(predicted_classes)
    return predicted_labels

# Get predictions
predicted_labels = predict(model, X, categories)

# Save predictions to CSV
for img_name, label in zip(image_names, predicted_labels):
    logger.info(f"Image: {img_name}, Predicted Category: {label}")

logger.info("Prediction batch completed.")
