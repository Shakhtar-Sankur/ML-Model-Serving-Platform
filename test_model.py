import base64

import numpy as np
import tensorflow as tf
from ml_serving_platform import ModelTrainer, ImageProcessor

def test_model_creation():
    trainer = ModelTrainer()
    model = trainer.create_model()
    assert model is not None
    assert len(model.layers) > 0

def test_image_preprocessing():
    fake_image_bytes = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).tobytes()
    processed = ImageProcessor.preprocess_image(fake_image_bytes)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32

def test_base64_decoding():
    test_data = b"test image data"
    encoded = base64.b64encode(test_data).decode()
    decoded = ImageProcessor.decode_base64_image(encoded)
    assert decoded == test_data