import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
print("Loading model...")
model = tf.keras.models.load_model("models/cnn_model.h5")
print("Creating TFLite converter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
print("Optimizing...")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Converting...")
tflite_model = converter.convert()
print(f"TFLite model size: {len(tflite_model)} bytes")
with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)
print("✓ TFLite model saved!")
