
import os
import json
import re
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from utils.disease_info import disease_info

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

HIGH_CONF = 0.85
MEDIUM_CONF = 0.60

client = MongoClient(
    "mongodb://localhost:27017/",
    serverSelectionTimeoutMS=3000
)
db = client["plant_disease_db"]
predictions_collection = db["predictions"]

model = tf.keras.models.load_model(
    "models/plant_disease_model.keras",
    compile=False
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_top_predictions(preds, top_k=3):
    top_indices = preds[0].argsort()[-top_k:][::-1]
    results = []

    for i in top_indices:
        class_name = class_names[int(i)]
        plant, disease = class_name.split("___", 1)

        results.append({
            "plant": plant,
            "disease": disease,
            "confidence": float(preds[0][i])
        })

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    top_predictions = None
    disease_details = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = (
                datetime.now().strftime("%Y%m%d%H%M%S%f")
                + "_"
                + filename
            )
            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                unique_name
            )

            try:
                file.save(image_path)

                img = Image.open(image_path).convert("RGB")
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = (
                    tf.keras.applications.efficientnet.preprocess_input(
                        img_array
                    )
                )

                preds = model.predict(img_array, verbose=0)
                top_predictions = get_top_predictions(preds)

                predicted_index = int(np.argmax(preds[0]))
                predicted_class = class_names[predicted_index]
                plant, prediction = predicted_class.split("___", 1)
                confidence = float(np.max(preds[0]))
                disease_details = disease_info.get(
                    predicted_class, {}
                )

                predictions_collection.insert_one({
                    "image_name": unique_name,
                    "plant": plant,
                    "disease": prediction,
                    "confidence": confidence,
                    "prediction_date": datetime.now()
                })

                print("Prediction saved:", predicted_class)

            except Exception as e:
                error = f"Prediction error: {e}"
                print(error)

        else:
            error = "Invalid file. Please upload a PNG, JPG, or JPEG image."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        top_predictions=top_predictions,
        disease_info=disease_details,
        image_path=image_path,
        HIGH_CONF=HIGH_CONF,
        MEDIUM_CONF=MEDIUM_CONF,
        error=error
    )


@app.route("/history")
def history():
    search = request.args.get("search", "").strip()
    query = {}

    if search:
        safe_search = re.escape(search)
        query = {
            "$or": [
                {
                    "plant": {
                        "$regex": safe_search,
                        "$options": "i"
                    }
                },
                {
                    "disease": {
                        "$regex": safe_search,
                        "$options": "i"
                    }
                }
            ]
        }

    try:
        records = list(
            predictions_collection.find(query).sort(
                "prediction_date", -1
            )
        )

        return render_template(
            "history.html",
            records=records,
            search=search
        )

    except Exception as e:
        return f"Unable to load history: {e}", 500


@app.route("/delete/<record_id>", methods=["POST"])
def delete_prediction(record_id):
    if not ObjectId.is_valid(record_id):
        return "Invalid prediction ID", 400

    try:
        predictions_collection.delete_one({
            "_id": ObjectId(record_id)
        })
        return redirect(url_for("history"))

    except Exception as e:
        return f"Unable to delete prediction: {e}", 500


@app.route("/dashboard")
def dashboard():
    try:
        total_predictions = predictions_collection.count_documents({})

        healthy_count = predictions_collection.count_documents({
            "disease": {
                "$regex": "healthy",
                "$options": "i"
            }
        })

        diseased_count = total_predictions - healthy_count

        recent_predictions = list(
            predictions_collection.find().sort(
                "prediction_date", -1
            ).limit(5)
        )

        return render_template(
            "dashboard.html",
            total_predictions=total_predictions,
            healthy_count=healthy_count,
            diseased_count=diseased_count,
            recent_predictions=recent_predictions
        )

    except Exception as e:
        return f"Unable to load dashboard: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)