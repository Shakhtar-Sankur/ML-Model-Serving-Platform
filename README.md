# ML Model Serving Platform

**An image-classification serving layer with the operational parts included — versioning,
circuit breaking, drift detection, A/B routing, caching and metrics.**

Most model-serving examples stop at "load model, return prediction". This one is mostly
about everything around that call, which is where production time actually goes.

## What's here

`ml_serving_platform.py` holds the whole service. Twelve classes, each one concern:

| Class | Concern |
|---|---|
| `ModelVersionManager` | Which version serves a request; rollback |
| `ABTestManager` | Splitting traffic across versions |
| `CircuitBreaker` | Failing fast when a backend is unhealthy instead of queueing |
| `HealthChecker` | Liveness for Triton, Redis and the model itself |
| `DataDriftDetector` | Watching input distribution against the training reference |
| `PerformanceProfiler` | Per-stage timing |
| `MetricsCollector` | Prometheus counters and histograms |
| `CacheManager` | Redis-backed response cache |
| `SecurityManager` | API-key auth and rate limiting |
| `ImageProcessor` | Decode, resize, normalise |
| `ModelTrainer` | Training and export |
| `TritonClient` | Inference backend |

Supporting files: `Dockerfile`, `docker-compose.yml` for local runs, `build.sh` and
`deploy.sh` for image build and Kubernetes rollout, and `test_api.py` / `test_model.py`.

## Design targets

- ~1,200 requests per second sustained
- P95 latency low enough for interactive use
- Cache hit rates high enough to take a meaningful bite out of compute cost
- No single unhealthy backend able to stall the whole service

## On the numbers

The figures above are **design targets** that shaped the implementation — they are not
measured results. This repository ships no benchmark harness and no trained weights, so
nothing here reproduces them. They are recorded because they drove real decisions about
architecture and algorithm choice, not as claims about observed performance.

## Running it

```bash
pip install -r requirements.txt
docker compose up          # service, Redis and Triton
pytest test_api.py test_model.py
```

## Status

Complete service implementation with tests. Requires a Triton backend and a Redis
instance; no trained model is bundled.

## Licence

All rights reserved. Published for reading, not for reuse.
