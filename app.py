from flask import Flask, render_template, request
import numpy as np
import os
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# ------------------------------------------------
# ✅ Load TFLite model (instead of TensorFlow .h5)
# ------------------------------------------------
interpreter = tflite.Interpreter(model_path="cattle_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No image selected!"

    # Save uploaded file
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # -----------------------------
    # Preprocess image for TFLite
    # -----------------------------
    img = Image.open(filepath).resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Run prediction
    # -----------------------------
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    pred_class = np.argmax(prediction, axis=1)[0]

    # Result mapping
    if pred_class == 0:
        result = "✅ Healthy Cow"
        cure = "No treatment needed. Keep providing proper diet and clean environment."
    else:
        result = "⚠️ Lumpy Disease Detected"
        cure = "Isolate the infected cow. Contact a veterinarian for antiviral treatment and maintain hygiene."

    return render_template('index.html', result=result, cure=cure, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
