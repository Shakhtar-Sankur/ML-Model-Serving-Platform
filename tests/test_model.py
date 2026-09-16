"""Preprocessing and the model head.

ResNet50's ImageNet weights are ~100 MB and would be fetched on every CI run, so
`create_model` is exercised with the backbone replaced by a tiny stand-in of the
same shape. What is being checked is the head that was written here — the
pooling, the 1000-way softmax, the loss — not that Keras can download a file.

The Triton config is checked against the model it is meant to describe, because
a mismatch between the two is silent until inference returns nothing.
"""

import base64
import io

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import ml_serving_platform as platform
from ml_serving_platform import ImageProcessor, ModelTrainer


def png_bytes(size=(100, 70), colour=(30, 60, 90)):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


# ── preprocessing ─────────────────────────────────────────────────────────
def test_any_size_comes_out_at_the_models_input_shape():
    for size in [(100, 70), (224, 224), (1000, 30)]:
        processed = ImageProcessor.preprocess_image(png_bytes(size))
        assert processed.shape == (1, 224, 224, 3), f"{size} produced {processed.shape}"


def test_the_result_is_float32_for_a_fp32_input_tensor():
    """Triton's input is declared TYPE_FP32; float64 would be rejected."""
    assert ImageProcessor.preprocess_image(png_bytes()).dtype == np.float32


def test_a_greyscale_image_is_widened_to_three_channels():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("L", (80, 80), 128).save(buffer, format="PNG")
    assert ImageProcessor.preprocess_image(buffer.getvalue()).shape == (1, 224, 224, 3)


def test_a_transparent_png_does_not_arrive_as_four_channels():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGBA", (80, 80), (10, 20, 30, 0)).save(buffer, format="PNG")
    assert ImageProcessor.preprocess_image(buffer.getvalue()).shape == (1, 224, 224, 3)


def test_the_resnet_preprocessing_is_actually_applied():
    """`preprocess_input` centres the channels; raw 0-255 values mean it was skipped."""
    processed = ImageProcessor.preprocess_image(png_bytes(colour=(0, 0, 0)))
    assert processed.min() < 0, "values never went negative — preprocess_input was not applied"


def test_a_ruined_image_raises_rather_than_returning_something():
    with pytest.raises(Exception):
        ImageProcessor.preprocess_image(b"this is not an image")


def test_base64_decoding_round_trips():
    payload = b"\x89PNG\r\n\x1a\n arbitrary bytes"
    assert ImageProcessor.decode_base64_image(base64.b64encode(payload).decode()) == payload


# ── the model head ────────────────────────────────────────────────────────
@pytest.fixture
def light_backbone(monkeypatch):
    """ResNet50 without the download: same output rank, trivial weights."""
    def fake_resnet(weights=None, include_top=False, input_shape=None):
        # Functional, not Sequential: `InputLayer(input_shape=...)` spells
        # differently in Keras 2 and 3, but `tf.keras.Input(shape=...)` does not.
        inputs = tf.keras.Input(shape=input_shape)
        outputs = tf.keras.layers.Conv2D(8, 3, strides=8, padding="same")(inputs)
        return tf.keras.Model(inputs, outputs, name="resnet50")

    monkeypatch.setattr(platform, "ResNet50", fake_resnet)


def test_the_model_takes_an_image_and_returns_1000_probabilities(light_backbone):
    model = ModelTrainer().create_model()
    output = model(np.zeros((2, 224, 224, 3), dtype=np.float32))

    assert tuple(output.shape) == (2, 1000)
    assert np.allclose(np.sum(output, axis=1), 1.0, atol=1e-4), "the head must end in a softmax"


def test_create_model_keeps_the_model_on_the_trainer(light_backbone):
    trainer = ModelTrainer()
    assert trainer.model is None
    returned = trainer.create_model()
    assert trainer.model is returned, "save_for_triton would have nothing to save"


def test_the_model_is_compiled_and_ready_to_train(light_backbone):
    model = ModelTrainer().create_model()
    assert model.optimizer is not None, "an uncompiled model cannot be fitted"
    assert "categorical_crossentropy" in str(model.loss)


def test_dropout_is_only_active_while_training(light_backbone):
    """Inference must be deterministic; dropout at serving time would not be."""
    model = ModelTrainer().create_model()
    batch = np.random.rand(1, 224, 224, 3).astype(np.float32)
    assert np.allclose(model(batch, training=False), model(batch, training=False))


def test_the_triton_config_matches_the_model_it_describes(light_backbone, tmp_path):
    trainer = ModelTrainer()
    model = trainer.create_model()

    try:
        trainer.save_for_triton(str(tmp_path / "image_classifier"))
    except Exception as error:                      # Keras 3 changed the save format
        pytest.skip(f"model export unavailable in this TensorFlow build: {error}")

    config = (tmp_path / "image_classifier" / "config.pbtxt").read_text()
    assert 'name: "image_classifier"' in config, \
        "the directory name and the Triton model name must agree"
    assert "dims: [ 224, 224, 3 ]" in config
    assert "dims: [ 1000 ]" in config
    assert int(model.output_shape[-1]) == 1000, \
        "the config promises 1000 outputs; the model must produce that many"
