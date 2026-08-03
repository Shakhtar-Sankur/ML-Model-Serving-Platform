import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import json
import boto3
from flask import Flask, request, jsonify
import requests
import asyncio
import aiohttp
from kubernetes import client, config
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
import redis
from PIL import Image
import io
import base64

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

class ModelVersionManager:
    def __init__(self, s3_bucket):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.current_version = "v1"
        
    def deploy_new_version(self, model_path, version):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = f"models/{version}/" + os.path.relpath(local_path, model_path)
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_path)
                
    def rollback_version(self, version):
        self.current_version = version
        
    def get_model_versions(self):
        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix="models/",
            Delimiter="/"
        )
        versions = []
        for prefix in response.get('CommonPrefixes', []):
            version = prefix['Prefix'].split('/')[-2]
            versions.append(version)
        return versions

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        
    def profile_inference(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            self.metrics[func.__name__] = {
                'latency': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': time.time()
            }
            
            return result
        return wrapper
        
    def get_memory_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

class DataDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.drift_threshold = 0.05
        
    def detect_drift(self, new_data):
        from scipy import stats
        
        reference_mean = np.mean(self.reference_data, axis=0)
        new_mean = np.mean(new_data, axis=0)
        
        ks_statistic, p_value = stats.ks_2samp(
            self.reference_data.flatten(),
            new_data.flatten()
        )
        
        drift_detected = p_value < self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'ks_statistic': ks_statistic,
            'reference_mean': reference_mean.tolist(),
            'new_mean': new_mean.tolist()
        }

class ABTestManager:
    def __init__(self):
        self.experiments = {}
        
    def create_experiment(self, name, model_a, model_b, traffic_split=0.5):
        self.experiments[name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': []
        }
        
    def route_request(self, experiment_name, request_data):
        experiment = self.experiments[experiment_name]
        
        if np.random.random() < experiment['traffic_split']:
            model = experiment['model_a']
            result_list = experiment['results_a']
        else:
            model = experiment['model_b']
            result_list = experiment['results_b']
            
        result = model.predict(request_data)
        result_list.append(result)
        
        return result
        
    def analyze_experiment(self, experiment_name):
        experiment = self.experiments[experiment_name]
        
        results_a = np.array(experiment['results_a'])
        results_b = np.array(experiment['results_b'])
        
        if len(results_a) == 0 or len(results_b) == 0:
            return {"error": "Insufficient data for analysis"}
            
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        return {
            'model_a_performance': np.mean(results_a),
            'model_b_performance': np.mean(results_b),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b)
        }

class SecurityManager:
    @staticmethod
    def validate_api_key(api_key):
        valid_keys = os.getenv('VALID_API_KEYS', '').split(',')
        return api_key in valid_keys
        
    @staticmethod
    def rate_limit_check(client_id, max_requests=1000, window=3600):
        cache_key = f"rate_limit:{client_id}"
        current_count = cache_manager.redis_client.get(cache_key)
        
        if current_count is None:
            cache_manager.redis_client.setex(cache_key, window, 1)
            return True
        elif int(current_count) < max_requests:
            cache_manager.redis_client.incr(cache_key)
            return True
        else:
            return False
            
    @staticmethod
    def sanitize_input(image_data):
        try:
            decoded = base64.b64decode(image_data)
            if len(decoded) > 10 * 1024 * 1024:
                raise ValueError("Image too large")
            return True
        except Exception:
            return False

class HealthChecker:
    def __init__(self):
        self.checks = {}
        
    def add_check(self, name, check_func):
        self.checks[name] = check_func
        
    def run_health_checks(self):
        results = {}
        overall_status = "healthy"
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = {"status": "healthy", "details": result}
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
                overall_status = "unhealthy"
                
        return {"overall_status": overall_status, "checks": results}

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def create_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1000, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def save_for_triton(self, model_path):
        os.makedirs(f"{model_path}/1", exist_ok=True)
        self.model.save(f"{model_path}/1/model.savedmodel")
        
        config_pbtxt = """
name: "image_classifier"
platform: "tensorflow_savedmodel"
max_batch_size: 32
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]
output [
  {
    name: "dense_1"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
version_policy: { all { } }
"""
        
        with open(f"{model_path}/config.pbtxt", "w") as f:
            f.write(config_pbtxt)

class TritonClient:
    def __init__(self, triton_url="http://triton-server:8000"):
        self.triton_url = triton_url
        self.model_name = "image_classifier"
        
    async def predict(self, image_data):
        async with aiohttp.ClientSession() as session:
            payload = {
                "inputs": [
                    {
                        "name": "input_1",
                        "shape": [1, 224, 224, 3],
                        "datatype": "FP32",
                        "data": image_data.tolist()
                    }
                ]
            }
            
            async with session.post(
                f"{self.triton_url}/v2/models/{self.model_name}/infer",
                json=payload
            ) as response:
                result = await response.json()
                return result["outputs"][0]["data"]

class ImageProcessor:
    @staticmethod
    def preprocess_image(image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    @staticmethod
    def decode_base64_image(base64_string):
        return base64.b64decode(base64_string)

class MetricsCollector:
    def __init__(self):
        self.request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
        
    def inc_request_count(self, method, endpoint):
        self.request_count.labels(method=method, endpoint=endpoint).inc()
        
    def observe_request_duration(self, duration):
        self.request_duration.observe(duration)

class CacheManager:
    def __init__(self, redis_host='redis-service', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    def get_prediction(self, image_hash):
        return self.redis_client.get(f"prediction:{image_hash}")
        
    def set_prediction(self, image_hash, prediction, ttl=3600):
        self.redis_client.setex(f"prediction:{image_hash}", ttl, json.dumps(prediction))

circuit_breaker = CircuitBreaker()
health_checker = HealthChecker()
performance_profiler = PerformanceProfiler()
security_manager = SecurityManager()

app = Flask(__name__)
triton_client = TritonClient()
metrics = MetricsCollector()
cache_manager = CacheManager()
processor = ImageProcessor()

def check_triton_health():
    response = requests.get(f"{triton_client.triton_url}/v2/health/ready", timeout=5)
    return response.status_code == 200

def check_redis_health():
    return cache_manager.redis_client.ping()

def check_model_health():
    test_data = np.random.random((1, 224, 224, 3)).astype(np.float32)
    result = asyncio.run(triton_client.predict(test_data))
    return len(result) == 1000

health_checker.add_check("triton", check_triton_health)
health_checker.add_check("redis", check_redis_health)
health_checker.add_check("model", check_model_health)

@app.before_request
def before_request():
    if request.endpoint != 'health_check' and request.endpoint != 'metrics_endpoint':
        api_key = request.headers.get('X-API-Key')
        if not security_manager.validate_api_key(api_key):
            return jsonify({"error": "Invalid API key"}), 401
            
        client_id = request.headers.get('X-Client-ID', 'anonymous')
        if not security_manager.rate_limit_check(client_id):
            return jsonify({"error": "Rate limit exceeded"}), 429

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    return jsonify(health_checker.run_health_checks())

@app.route('/predict', methods=['POST'])
async def predict():
    start_time = time.time()
    metrics.inc_request_count('POST', '/predict')
    
    try:
        data = request.get_json()
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400
            
        image_bytes = processor.decode_base64_image(image_b64)
        image_hash = str(hash(image_bytes))
        
        cached_result = cache_manager.get_prediction(image_hash)
        if cached_result:
            return jsonify(json.loads(cached_result))
        
        processed_image = processor.preprocess_image(image_bytes)
        prediction = await triton_client.predict(processed_image)
        
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result = {
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "processing_time": time.time() - start_time
        }
        
        cache_manager.set_prediction(image_hash, result)
        
        metrics.observe_request_duration(time.time() - start_time)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    return generate_latest()

@app.route('/batch_predict', methods=['POST'])
async def batch_predict():
    start_time = time.time()
    metrics.inc_request_count('POST', '/batch_predict')
    
    try:
        data = request.get_json()
        images = data.get('images', [])
        
        if not images:
            return jsonify({"error": "No images provided"}), 400
            
        tasks = []
        for img_b64 in images:
            image_bytes = processor.decode_base64_image(img_b64)
            processed_image = processor.preprocess_image(image_bytes)
            tasks.append(triton_client.predict(processed_image))
        
        predictions = await asyncio.gather(*tasks)
        
        results = []
        for pred in predictions:
            predicted_class = np.argmax(pred)
            confidence = float(np.max(pred))
            results.append({
                "predicted_class": int(predicted_class),
                "confidence": confidence
            })
        
        response = {
            "predictions": results,
            "processing_time": time.time() - start_time
        }
        
        metrics.observe_request_duration(time.time() - start_time)
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/metrics', methods=['GET'])
def admin_metrics():
    return jsonify({
        "performance_metrics": performance_profiler.metrics,
        "circuit_breaker_state": circuit_breaker.state,
        "cache_stats": {
            "hit_rate": cache_manager.redis_client.info().get('keyspace_hits', 0),
            "miss_rate": cache_manager.redis_client.info().get('keyspace_misses', 0)
        }
    })

class KubernetesDeployer:
    def __init__(self):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        
    def create_deployment(self):
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "ml-serving-api",
                "labels": {"app": "ml-serving-api"}
            },
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "ml-serving-api"}},
                "template": {
                    "metadata": {"labels": {"app": "ml-serving-api"}},
                    "spec": {
                        "containers": [{
                            "name": "ml-api",
                            "image": "ml-serving-api:latest",
                            "ports": [{"containerPort": 5000}],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"}
                            },
                            "env": [
                                {"name": "TRITON_URL", "value": "http://triton-server:8000"},
                                {"name": "REDIS_HOST", "value": "redis-service"}
                            ],
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 5000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 5000},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        self.apps_v1.create_namespaced_deployment(
            body=deployment_manifest,
            namespace="default"
        )
        
    def create_triton_deployment(self):
        triton_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "triton-server",
                "labels": {"app": "triton-server"}
            },
            "spec": {
                "replicas": 2,
                "selector": {"matchLabels": {"app": "triton-server"}},
                "template": {
                    "metadata": {"labels": {"app": "triton-server"}},
                    "spec": {
                        "containers": [{
                            "name": "triton",
                            "image": "nvcr.io/nvidia/tritonserver:23.08-py3",
                            "args": ["tritonserver", "--model-repository=/models"],
                            "ports": [
                                {"containerPort": 8000},
                                {"containerPort": 8001},
                                {"containerPort": 8002}
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "2Gi"},
                                "limits": {"cpu": "2", "memory": "4Gi"}
                            },
                            "volumeMounts": [{
                                "name": "model-volume",
                                "mountPath": "/models"
                            }]
                        }],
                        "volumes": [{
                            "name": "model-volume",
                            "persistentVolumeClaim": {"claimName": "model-pvc"}
                        }]
                    }
                }
            }
        }
        
        self.apps_v1.create_namespaced_deployment(
            body=triton_deployment,
            namespace="default"
        )
        
    def create_services(self):
        api_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "ml-serving-api-service"},
            "spec": {
                "selector": {"app": "ml-serving-api"},
                "ports": [{"port": 80, "targetPort": 5000}],
                "type": "LoadBalancer"
            }
        }
        
        triton_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "triton-server"},
            "spec": {
                "selector": {"app": "triton-server"},
                "ports": [
                    {"name": "http", "port": 8000, "targetPort": 8000},
                    {"name": "grpc", "port": 8001, "targetPort": 8001},
                    {"name": "metrics", "port": 8002, "targetPort": 8002}
                ]
            }
        }
        
        redis_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "redis-service"},
            "spec": {
                "selector": {"app": "redis"},
                "ports": [{"port": 6379, "targetPort": 6379}]
            }
        }
        
        self.v1.create_namespaced_service(body=api_service, namespace="default")
        self.v1.create_namespaced_service(body=triton_service, namespace="default")
        self.v1.create_namespaced_service(body=redis_service, namespace="default")
        
    def create_hpa(self):
        hpa = {
            "apiVersion": "autoscaling/v1",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": "ml-serving-hpa"},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "ml-serving-api"
                },
                "minReplicas": 3,
                "maxReplicas": 20,
                "targetCPUUtilizationPercentage": 70
            }
        }
        
        self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
            body=hpa,
            namespace="default"
        )

class AWSManager:
    def __init__(self):
        self.eks_client = boto3.client('eks')
        self.s3_client = boto3.client('s3')
        self.ecr_client = boto3.client('ecr')
        
    def upload_model_to_s3(self, model_path, bucket_name):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.relpath(local_path, model_path)
                self.s3_client.upload_file(local_path, bucket_name, f"models/{s3_path}")
                
    def create_ecr_repository(self, repository_name):
        try:
            self.ecr_client.create_repository(repositoryName=repository_name)
        except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
            pass
            
    def get_eks_cluster_info(self, cluster_name):
        response = self.eks_client.describe_cluster(name=cluster_name)
        return response['cluster']

class LoadTester:
    def __init__(self, base_url):
        self.base_url = base_url
        
    async def send_request(self, session, image_data):
        async with session.post(
            f"{self.base_url}/predict",
            json={"image": image_data}
        ) as response:
            return await response.json()
            
    async def run_load_test(self, concurrent_users=100, requests_per_user=50):
        sample_image = base64.b64encode(b"fake_image_data").decode()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for user in range(concurrent_users):
                for req in range(requests_per_user):
                    tasks.append(self.send_request(session, sample_image))
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_requests = len([r for r in responses if not isinstance(r, Exception)])
            total_time = end_time - start_time
            
            print(f"Total requests: {len(tasks)}")
            print(f"Successful requests: {successful_requests}")
            print(f"Failed requests: {len(tasks) - successful_requests}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Requests per second: {len(tasks)/total_time:.2f}")

class ModelOptimizer:
    @staticmethod
    def optimize_for_inference(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        return tflite_model
        
    @staticmethod
    def quantize_model(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: [
            [np.random.random((1, 224, 224, 3)).astype(np.float32)]
            for _ in range(100)
        ]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        quantized_model = converter.convert()
        return quantized_model

if __name__ == "__main__":
    trainer = ModelTrainer()
    model = trainer.create_model()
    trainer.save_for_triton("./model_repository/image_classifier")
    
    aws_manager = AWSManager()
    aws_manager.upload_model_to_s3("./model_repository", "ml-models-bucket")
    
    deployer = KubernetesDeployer()
    deployer.create_deployment()
    deployer.create_triton_deployment()
    deployer.create_services()
    deployer.create_hpa()
    
    app.run(host='0.0.0.0', port=5000)