# 🌱 Plant Disease Detection System

An AI-powered web application that identifies plant diseases from leaf images using deep learning. Built with **Python, TensorFlow, Flask, and MongoDB**, the system provides disease predictions, confidence scores, treatment information, and prediction history through a simple web interface.

## 📌 Project Overview

Plant diseases can affect crop health and agricultural productivity. This project uses a trained deep learning model to classify leaf images into plant disease categories and present the prediction results through a web application.

The application also stores prediction records in MongoDB, allowing users to view previous predictions, search records, and delete entries from the history page.

## ✨ Features

* 🌿 **AI-Based Disease Detection:** Predicts plant diseases from uploaded leaf images.
* 🎯 **Top-3 Predictions:** Displays the three most likely classes with confidence scores.
* 📊 **Confidence Scores:** Shows the model's confidence in its predictions.
* 💊 **Disease Information:** Provides available disease descriptions and treatment suggestions.
* 📈 **Dashboard:** Displays total predictions, healthy results, diseased results, and recent predictions.
* 🗂️ **Prediction History:** Stores and displays previous prediction records.
* 🔍 **Search Functionality:** Searches prediction history by plant or disease.
* 🗑️ **Delete Records:** Removes selected prediction records from MongoDB.
* 🍃 **User-Friendly Interface:** Uses a green, agriculture-inspired design.

## 🛠️ Technologies Used

| Technology         | Purpose                           |
| ------------------ | --------------------------------- |
| Python             | Application logic                 |
| TensorFlow / Keras | Deep learning model and inference |
| Flask              | Web application backend           |
| HTML and CSS       | User interface                    |
| MongoDB            | Prediction record storage         |
| PyMongo            | Python–MongoDB connection         |
| NumPy and Pillow   | Image processing                  |

## 🧠 Dataset and Model

The model was trained using the **PlantVillage dataset**, which contains images covering 38 plant-disease and healthy-leaf categories.

* **Model format:** Keras (`.keras`)
* **Input image size:** 224 × 224 pixels
* **Number of classes:** 38
* **Reported validation accuracy:** Approximately 96.47% in the training run

Actual predictions can vary depending on image quality, lighting, background, and whether the uploaded image resembles the training data.

## ⚙️ How It Works

1. The user uploads a leaf image through the web interface.
2. The application validates and preprocesses the image.
3. The trained TensorFlow model generates predictions.
4. The application displays the predicted plant, disease, and confidence score.
5. The top three predictions and available disease information are displayed.
6. The prediction record is saved in MongoDB.
7. The dashboard and history page retrieve stored records for viewing and management.

## 📂 Project Structure

```text
PlantdiseaseDetector/
│
├── app.py
├── train.py
├── test_model.py
├── class_names.json
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── plant_disease_model.keras
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── history.html
│
├── static/
│   └── uploads/
│
└── utils/
    └── disease_info.py
```

*Note: This structure represents the main application files; some local folders may be excluded from GitHub.*

## 🚀 Installation and Setup

### Prerequisites

* Python 3.11 (or another compatible version for the installed TensorFlow release)
* MongoDB Community Server
* Git
* A web browser

### 1. Clone the repository

```bash
git clone https://github.com/hamsha-2030/Plant-Disease-Detector.git
cd Plant-Disease-Detector
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start MongoDB

Install and start MongoDB Community Server, then ensure the local MongoDB service is running.

The application uses the following local database configuration:

* **Database:** `plant_disease_db`
* **Collection:** `predictions`
* **Connection URI:** `mongodb://localhost:27017/`

The database and collection can be created automatically when the application inserts its first prediction.

### 5. Run the application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

Open the dashboard at:

```text
http://127.0.0.1:5000/dashboard
```

View prediction history at:

```text
http://127.0.0.1:5000/history
```

## 🗄️ MongoDB Integration

Each prediction is stored in the `predictions` collection with fields such as:

| Field             | Description                         |
| ----------------- | ----------------------------------- |
| `image_name`      | Uploaded image filename             |
| `plant`           | Predicted plant category            |
| `disease`         | Predicted disease or healthy status |
| `confidence`      | Model confidence score              |
| `prediction_date` | Date and time of prediction         |

MongoDB Compass can be used to inspect the database and view, query, and manage prediction documents.

## 🔮 Future Enhancements

* Grad-CAM visual explanations to highlight image regions influencing predictions.
* Deployment to a cloud platform.
* Mobile-friendly interface improvements.
* Expanded datasets covering more real-world leaf conditions.
* Further model evaluation on images captured outside controlled datasets.

## ⚠️ Disclaimer

This application is an educational project and provides model-based predictions, not a definitive agricultural diagnosis. Treatment suggestions should be verified with reliable agricultural guidance before use.

## 👩‍💻 Author

**Hamshavarshini R**

B.Tech — Artificial Intelligence and Machine Learning

Sri Ramachandra Faculty of Engineering and Technology

## 🔗 Repository

[View Plant Disease Detector on GitHub](https://github.com/hamsha-2030/Plant-Disease-Detector)

![Show the uploaded leaf, predicted disease, confidence score, and treatment information.](image.png)
![Show total predictions, healthy and diseased counts, and recent results.](image-1.png)
![Show your actual plant_disease_db database and predictions collection.](image-2.png)