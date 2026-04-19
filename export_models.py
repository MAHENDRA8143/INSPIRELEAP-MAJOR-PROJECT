#!/usr/bin/env python
"""Export H5 model to TFLite and ONNX formats."""
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from src.utils import convert_to_tflite, convert_to_onnx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    model_path = Path("models/cnn_model.h5")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return False
    
    logger.info(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(str(model_path))
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Input shape: {model.input_shape}")
        logger.info(f"  Output shape: {model.output_shape}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Export TFLite
    try:
        logger.info("Converting to TFLite...")
        tflite_path = Path("models/model.tflite")
        convert_to_tflite(model, tflite_path)
        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ TFLite exported: {tflite_path} ({size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}", exc_info=True)
        return False
    
    # Export ONNX
    try:
        logger.info("Converting to ONNX...")  
        onnx_path = Path("models/model.onnx")
        convert_to_onnx(model, onnx_path, logger)
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ ONNX exported: {onnx_path} ({size_mb:.2f} MB)")
        else:
            logger.warning("ONNX file was not created")
            return False
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}", exc_info=True)
        return False
    
    logger.info("\n" + "="*60)
    logger.info("ALL MODELS EXPORTED SUCCESSFULLY!")
    logger.info("="*60)
    
    # Summary
    models_dir = Path("models")
    logger.info("\nModel Files:")
    for f in sorted(models_dir.glob("*.h5")) + sorted(models_dir.glob("*.tflite")) + sorted(models_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name:30} {size_mb:8.2f} MB")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
