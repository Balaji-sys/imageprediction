from flask import Flask, render_template, request
import numpy as np
import json
import os
import PIL.Image as Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TFLite model
MODEL_PATH = "plant_model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    # Run inference
    interpreter.invoke()
    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])

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
    confidence_score = None

    if request.method == 'POST':
        file = request.files.get('leaf')
        if file and file.filename != "":
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            plant_class, confidence = predict_image(image_path)
            prediction = f"{plant_class}"
            confidence_score = confidence

    return render_template('index.html',
                           prediction=prediction,
                           image=image_path,
                           confidence=confidence_score)

if __name__ == '__main__':
    app.run(debug=True)
