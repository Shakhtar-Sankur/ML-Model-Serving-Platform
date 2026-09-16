"""The HTTP surface: who gets in, what comes back, and what is cached.

Triton is replaced with a stub, Redis with the in-memory client from conftest.
Nothing here opens a socket. The image is a real PNG built with Pillow — the
previous version of this file base64'd raw array bytes, which Pillow cannot
open, so every request it made returned 500 and the assertions never ran.
"""

import base64
import io
import json

import pytest

pytest.importorskip("tensorflow")

import ml_serving_platform as platform


@pytest.fixture
def png():
    """A small real PNG, base64 encoded the way a client would send it."""
    from PIL import Image
    image = Image.new("RGB", (64, 48), (120, 90, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def stub_triton(monkeypatch):
    """A Triton that answers instantly with a fixed 1000-class distribution."""
    calls = []
    scores = [0.0] * 1000
    scores[7] = 0.9

    async def predict(image_data):
        calls.append(image_data)
        return scores

    monkeypatch.setattr(platform.triton_client, "predict", predict)
    return calls


# ── the endpoints that need no key ────────────────────────────────────────
def test_health_needs_no_api_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert json.loads(response.data)["status"] == "healthy"


def test_metrics_needs_no_api_key(client):
    assert client.get("/metrics").status_code == 200


# ── authentication ────────────────────────────────────────────────────────
def test_a_request_with_no_key_is_refused(client, png):
    response = client.post("/predict", json={"image": png})
    assert response.status_code == 401


def test_a_request_with_a_wrong_key_is_refused(client, png):
    response = client.post("/predict", json={"image": png},
                           headers={"X-API-Key": "not-the-key"})
    assert response.status_code == 401


def test_the_detailed_health_endpoint_is_not_public(client):
    """It reports backend hostnames and errors, so it sits behind the key."""
    assert client.get("/health/detailed").status_code == 401


def test_a_refused_request_never_reaches_the_model(client, png, stub_triton):
    client.post("/predict", json={"image": png}, headers={"X-API-Key": "nope"})
    assert not stub_triton, "an unauthenticated request was served by the model"


def test_a_rate_limited_client_gets_429(client, png, monkeypatch, api_key):
    monkeypatch.setattr(platform.security_manager, "rate_limit_check",
                        lambda *args, **kwargs: False)
    response = client.post("/predict", json={"image": png},
                           headers={"X-API-Key": api_key})
    assert response.status_code == 429


# ── prediction ────────────────────────────────────────────────────────────
def test_a_valid_request_returns_a_class_and_a_confidence(client, png, api_key, stub_triton):
    response = client.post("/predict", json={"image": png},
                           headers={"X-API-Key": api_key})
    assert response.status_code == 200, response.data
    body = json.loads(response.data)
    assert body["predicted_class"] == 7
    assert body["confidence"] == pytest.approx(0.9)
    assert body["processing_time"] >= 0


def test_the_image_reaches_the_model_at_the_size_it_expects(client, png, api_key, stub_triton):
    client.post("/predict", json={"image": png}, headers={"X-API-Key": api_key})
    assert stub_triton, "the model was never called"
    assert stub_triton[0].shape == (1, 224, 224, 3), \
        f"Triton was sent {stub_triton[0].shape}; it is configured for (1, 224, 224, 3)"


def test_a_missing_image_is_a_400_not_a_500(client, api_key):
    response = client.post("/predict", json={}, headers={"X-API-Key": api_key})
    assert response.status_code == 400
    assert "error" in json.loads(response.data)


def test_an_unreadable_image_does_not_return_a_success(client, api_key, stub_triton):
    response = client.post("/predict", json={"image": base64.b64encode(b"not a png").decode()},
                           headers={"X-API-Key": api_key})
    assert response.status_code != 200
    assert not stub_triton, "garbage was forwarded to the model"


def test_the_second_request_for_one_image_is_served_from_cache(client, png, api_key, stub_triton):
    first = client.post("/predict", json={"image": png}, headers={"X-API-Key": api_key})
    second = client.post("/predict", json={"image": png}, headers={"X-API-Key": api_key})

    assert first.status_code == second.status_code == 200
    assert len(stub_triton) == 1, "the cache did not spare the model a second call"
    assert json.loads(second.data)["predicted_class"] == 7


def test_a_different_image_is_not_served_from_the_first_ones_cache(client, png, api_key, stub_triton):
    from PIL import Image
    other = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 200, 10)).save(other, format="PNG")

    client.post("/predict", json={"image": png}, headers={"X-API-Key": api_key})
    client.post("/predict", json={"image": base64.b64encode(other.getvalue()).decode()},
                headers={"X-API-Key": api_key})

    assert len(stub_triton) == 2, "two different images shared one cache entry"


# ── batch prediction ──────────────────────────────────────────────────────
def test_a_batch_returns_one_result_per_image(client, png, api_key, stub_triton):
    response = client.post("/batch_predict", json={"images": [png, png, png]},
                           headers={"X-API-Key": api_key})
    assert response.status_code == 200, response.data
    predictions = json.loads(response.data)["predictions"]
    assert len(predictions) == 3
    assert all(p["predicted_class"] == 7 for p in predictions)


def test_an_empty_batch_is_a_400(client, api_key):
    response = client.post("/batch_predict", json={"images": []},
                           headers={"X-API-Key": api_key})
    assert response.status_code == 400


# ── health checks ─────────────────────────────────────────────────────────
def test_one_failing_check_makes_the_whole_service_unhealthy():
    checker = platform.HealthChecker()
    checker.add_check("good", lambda: True)
    checker.add_check("bad", lambda: (_ for _ in ()).throw(RuntimeError("redis is gone")))

    report = checker.run_health_checks()

    assert report["overall_status"] == "unhealthy"
    assert report["checks"]["good"]["status"] == "healthy"
    assert report["checks"]["bad"]["status"] == "unhealthy"
    assert "redis is gone" in report["checks"]["bad"]["error"]


def test_all_checks_run_even_when_an_early_one_fails():
    """A failure must not stop the others, or the report is half blank."""
    ran = []
    checker = platform.HealthChecker()
    checker.add_check("first", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    checker.add_check("second", lambda: ran.append("second") or True)

    checker.run_health_checks()
    assert ran == ["second"]
