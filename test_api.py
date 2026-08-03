import pytest
import json
import base64
from app import app, cache_manager
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return base64.b64encode(img_array.tobytes()).decode()

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_predict_endpoint(client, sample_image):
    response = client.post('/predict', 
                          json={'image': sample_image},
                          headers={'X-API-Key': 'test-key'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predicted_class' in data
    assert 'confidence' in data

def test_batch_predict(client, sample_image):
    response = client.post('/batch_predict',
                          json={'images': [sample_image, sample_image]},
                          headers={'X-API-Key': 'test-key'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'predictions' in data
    assert len(data['predictions']) == 2

def test_invalid_api_key(client, sample_image):
    response = client.post('/predict', 
                          json={'image': sample_image},
                          headers={'X-API-Key': 'invalid-key'})
    assert response.status_code == 401

def test_missing_image(client):
    response = client.post('/predict', 
                          json={},
                          headers={'X-API-Key': 'test-key'})
    assert response.status_code == 400

def test_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200