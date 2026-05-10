from flask import Flask, render_template, request
import numpy as np
import json
import os
import PIL.Image as Image
import tensorflow as tf


import os
import gdown

# Model download - only if not already present
MODEL_PATH = "plant_model.h5"
FILE_ID = "1p37FXVRm5sYNDmy5xtVRaCLq_wd7tipo"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/file/d/1p37FXVRm5sYNDmy5xtVRaCLq_wd7tipo/view?usp=drive_link", MODEL_PATH, quiet=False)
    print("Model downloaded!")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model from SavedModel format
try:
    # Try loading the new SavedModel format first
    model = tf.keras.models.load_model('plant_model_new')
except Exception as e:
    # Fallback to old h5 format
    model = tf.keras.models.load_model('plant_model.h5', compile=False)






# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    # class_indices is now {index: class_name}, so convert if needed
    if isinstance(list(class_indices.keys())[0], str):
        # Keys are strings (like "0", "1"), convert to int keys
        class_names = {int(k): v for k, v in class_indices.items()}
    else:
        # Keys are already ints
        class_names = class_indices

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction[0])
    confidence = float(prediction[0][class_id])
    
    # Debug: Print raw prediction probabilities
    print(f"DEBUG: Raw predictions: {prediction[0]}")
    print(f"DEBUG: Predicted class_id: {class_id}, Confidence: {confidence:.4f}")
    print(f"DEBUG: Predicted class: {class_names[class_id]}")
    
    return class_names[class_id], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('leaf')
        if file and file.filename != "":
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            plant_class, confidence = predict_image(image_path)
            prediction = f"{plant_class}"

    return render_template('index.html',
                           prediction=prediction,
                           image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
