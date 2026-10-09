import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    "plant_disease_model.keras",
    compile=False
)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load image
img = Image.open("test_images/tomato_leaf3.jpg.jpeg").convert("RGB")
img = img.resize((224, 224))

img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Disease:", predicted_class)
print("Confidence:", round(confidence, 2), "%")