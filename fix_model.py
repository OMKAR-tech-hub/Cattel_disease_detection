import tensorflow as tf
from tensorflow.keras.models import load_model

print("🔍 Loading model...")
try:
    model = load_model("cattle_model.h5", compile=False)
    print("✅ Model loaded normally.")
except Exception as e:
    print("⚠️ Model could not be loaded directly, trying custom load logic.")
    print(e)

    # Try to load only weights from the model file
    base_model = tf.keras.applications.EfficientNetB4(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    output = tf.keras.layers.Dense(5, activation="softmax")(x)  # 5 = number of classes (you can change it)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    print("✅ Base EfficientNet model rebuilt.")

    try:
        model.load_weights("cattle_model.h5")
        print("✅ Weights loaded successfully.")
    except:
        print("⚠️ Could not load weights, using new weights instead.")

# Save as new fixed version
model.save("cattle_model_fixed.h5")
print("🎯 Fixed model saved as cattle_model_fixed.h5")
