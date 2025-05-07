#Fashion Image Classifier

This project demonstrates an end-to-end machine learning workflow to classify fashion product images into one of five categories: `Shirts`, `Watches`, `Handbags`, `Casual Shoes`, and `Sports Shoes`.

The project includes:
- A convolutional neural network (CNN) model trained on image data
- A REST API using Flask to classify uploaded images
- An automated batch prediction script with scheduled execution
- Logging of all predictions to a CSV file

---

## 📁 Project Structure

```
Project From Model to Production/
│
├── FashionImageClassifier.ipynb       # Training notebook
├── prediction_log.csv                 # Log file of predictions
│
├── model/
│   ├── fashion_model.h5               # Trained Keras model
│   └── label_encoder.pkl              # Label encoder for class labels
│
├── api/
│   ├── app.py                         # Flask API for real-time prediction
│   ├── batch_predict_updated.py       # Script for batch prediction
│   └── run_batch_predict.bat          # Batch script for Windows Task Scheduler
│
├── new_images/                        # Folder for new images to classify
```

---

## 🚀 How to Use

### 🔧 1. Install Requirements

```bash
pip install -r requirements.txt
```

*(You can generate this with `pip freeze > requirements.txt`)*

---

### 📸 2. Run the API Server

```bash
cd api
python app.py
```

Then send a POST request to `http://127.0.0.1:5000/predict` with an image using Postman or curl:

```bash
curl -X POST -F "image=@path_to_your_image.jpg" http://127.0.0.1:5000/predict
```

The response is a JSON object showing prediction probabilities for each class.

---

### 🗂️ 3. Batch Prediction

To classify all images in the `new_images/` folder and save results:

```bash
cd api
python batch_predict_updated.py
```

Predictions will be saved in `prediction_log.csv`.

---

### 🕒 4. Automate with Task Scheduler

Use `run_batch_predict.bat` to schedule daily predictions via Windows Task Scheduler.  
This file runs the batch prediction script automatically.

---

## 📊 Model Performance

- CNN with 2 Conv layers + Dense
- Input size: 128x128 RGB images
- Accuracy: ~XX% on test set (fill in your result)

---

## 📁 Notes

- Trained on 5 balanced classes with 500 images each
- Dataset: [Kaggle - Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- Large datasets are excluded from this repo for size reasons

---

## 🙌 Credits

- Built with TensorFlow, Flask, and Python
- Created for the course *Project From Model to Production* at IU