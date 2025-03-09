import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cifar10_model.h5")

# CIFAR-10 class labels
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "house", "ship", "truck"]

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """ Load and preprocess the image """
    img = Image.open(image_path).resize((32, 32))  # Resize to match CIFAR-10 size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess and predict
            img = preprocess_image(filepath)
            predictions = model.predict(img)
            predicted_class = classes[np.argmax(predictions)]

            return render_template("index.html", uploaded_image=filepath, prediction=predicted_class)

    return render_template("index.html", uploaded_image=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
