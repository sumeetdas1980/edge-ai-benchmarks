"""
Adapter for MobileNetV2 fp32, converted via tf.keras.applications.MobileNetV2
+ tf.lite.TFLiteConverter (see models/convert_mobilenetv2.py).

An adapter's job is ONLY to describe what is specific to this model:
- where the .tflite file lives
- how to preprocess an input image into the tensor the model expects
- how to turn the raw output tensor into human-readable predictions

Everything generic (loading the interpreter, picking a backend/delegate,
warmup, timing, memory, CSV logging) lives in harness.py and does not
change when you add a new model.
"""

import os
import numpy as np
from PIL import Image

# Path is relative to the repo root, since harness.py always resolves
# paths from the repo root regardless of where you launch it from.
MODEL_PATH = os.path.join("models", "mobilenetv2_fp32.tflite")

# This model was converted from Keras's own MobileNetV2, so its output
# is 1000-class Keras/ImageNet ordering, not the 1001-class ordering
# used by Google's original mobilenet_v2_1.0_224.tflite release.
# Decode with Keras's own helper instead of a hand-rolled labels.txt
# to avoid a silent class-index mismatch.
IS_QUANTIZED = False  # fp32 model -> not eligible for the vx_npu backend

INPUT_SIZE = (224, 224)


def preprocess(image_path):
    """Load an image file and return a (1, 224, 224, 3) float32 tensor
    normalized to [-1, 1], matching MobileNetV2's expected input range."""
    img = Image.open(image_path).resize(INPUT_SIZE).convert("RGB")
    x = np.array(img, dtype=np.float32)
    x = (x - 127.5) / 127.5
    return np.expand_dims(x, 0)


def decode(output_tensor, top=5):
    """Turn the raw (1000,) output vector into a list of (label, score)."""
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

    preds = decode_predictions(np.expand_dims(output_tensor, 0), top=top)[0]
    return [(label, float(score)) for (_, label, score) in preds]
