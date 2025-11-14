import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ------------------------------------------------
# Load TFLite model
# ------------------------------------------------
interpreter = tflite.Interpreter(model_path="cattle_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------
st.set_page_config(
    page_title="Cattle Disease Detection",
    page_icon="🐄",
    layout="wide"
)

# ------------------------------------------------
# Beautiful Farming Theme UI (CSS)
# ------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #2d6a4f, #40916c, #74c69d);
    font-family: 'Poppins', sans-serif;
}

/* Center card */
.main-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    padding: 40px;
    border-radius: 25px;
    max-width: 900px;
    margin: auto;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.25);
    animation: fadeIn 1.2s ease-in-out;
}

/* Glowing Title */
h1 {
    text-align: center;
    font-weight: 900;
    color: #eaffd0;
    text-shadow: 0px 0px 12px #b5e48c, 0px 0px 20px #52b788;
    font-size: 3rem;
}

/* Upload button */
input[type="file"] {
    background: #ffffffdd !important;
    padding: 12px;
    border-radius: 12px;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #95d5b2, #52b788);
    padding: 12px 28px;
    color: black;
    font-size: 20px;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #b7e4c7, #74c69d);
}

/* Result Card */
.result-box {
    margin-top: 25px;
    padding: 25px;
    background: rgba(0,0,0,0.55);
    border-radius: 20px;
    color: #d8f3dc;
    border-left: 5px solid #95d5b2;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* Fade Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown("<h1>🐄 Cattle Disease Detection</h1>", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.write("### Upload a cow image to detect if it is healthy or infected (Lumpy Disease).")

# ------------------------------------------------
# File Upload
# ------------------------------------------------
uploaded_file = st.file_uploader("Choose a cow image", type=["jpg", "jpeg", "png"])

# ------------------------------------------------
# Prediction Logic
# ------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="📷 Uploaded Image", width=400)

    if st.button("🔍 Predict Disease"):

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        pred_class = np.argmax(prediction, axis=1)[0]

        # Result mapping
        if pred_class == 0:
            result = "✅ **Healthy Cow**"
            cure = "No treatment needed. Maintain clean shelter, good diet, and fresh water."
        else:
            result = "⚠️ **Lumpy Skin Disease Detected**"
            cure = "• Isolate cow immediately.\n• Contact veterinarian.\n• Provide antiviral medication.\n• Maintain hygiene to stop spreading."

        # Show result
        st.markdown(f"""
        <div class='result-box'>
            <h2>{result}</h2>
            <p><b>Treatment:</b><br>{cure}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
