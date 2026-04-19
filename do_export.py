import subprocess
import sys

# Run the model export in a subprocess
script = '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from pathlib import Path

print("Step 1: Loading H5 model...")
model = tf.keras.models.load_model("models/cnn_model.h5")
print(f"  Model loaded: input_shape={model.input_shape}, output_shape={model.output_shape}")

print("\\nStep 2: Creating TFLite model...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = Path("models/model.tflite")
    tflite_path.write_bytes(tflite_model)
    size_mb = tflite_path.stat().st_size / (1024**2)
    print(f"  ✓ TFLite created: {tflite_path.name} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"  ✗ TFLite failed: {e}")
    import traceback
    traceback.print_exc()

print("\\nStep 3: Creating ONNX model...")
try:
    import tf2onnx
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx_path = Path("models/model.onnx")
    onnx_path.write_bytes(model_proto.SerializeToString())
    size_mb = onnx_path.stat().st_size / (1024**2)
    print(f"  ✓ ONNX created: {onnx_path.name} ({size_mb:.2f} MB)")
except ImportError:
    print("  ⚠ tf2onnx not available, skipping ONNX")
except Exception as e:
    print(f"  ✗ ONNX failed: {e}")
    import traceback
    traceback.print_exc()

print("\\nVerifying models...")
models_dir = Path("models")
for f in sorted(models_dir.glob("*.h5")) + sorted(models_dir.glob("*.tflite")) + sorted(models_dir.glob("*.onnx")):
    size = f.stat().st_size / (1024**2)
    print(f"  ✓ {f.name} ({size:.2f} MB)")
print("\\nDONE!")
'''

try:
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=r"e:\INSPIRE LEAP\MAJOR PROJECT\digit-ai-system",
        capture_output=True,
        text=True,
        timeout=300
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print("Export timed out after 5 minutes")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
