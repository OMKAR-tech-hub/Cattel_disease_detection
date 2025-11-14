from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Clear any previous TensorFlow sessions
tf.keras.backend.clear_session()

# ✅ Load your fixed cattle disease detection model
model = tf.keras.models.load_model("cattle_model_fixed.h5", compile=False)
print("✅ Model loaded successfully!")

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

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
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
