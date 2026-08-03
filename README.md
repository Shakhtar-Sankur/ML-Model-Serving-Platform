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
| `DataDriftDetector` | Per-feature KS test against the training reference, Bonferroni-corrected |
| `PerformanceProfiler` | Per-stage timing |
| `MetricsCollector` | Prometheus counters and histograms |
| `CacheManager` | Redis-backed response cache |
| `SecurityManager` | API-key auth (constant-time) and an atomic fixed-window rate limit |
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

Complete service implementation with tests. Requires a Triton backend and a Redis instance;
no trained model is bundled.

### Notes from a correctness pass

Five defects fixed, one of them a security hole:

- **An empty API key authenticated.** `api_key in os.getenv('VALID_API_KEYS', '').split(',')`
  splits an unset variable to `['']`, so an empty key matched — and unset is the default
  deployment state. A trailing comma in the variable did the same. Empty entries are now
  dropped, an unconfigured service rejects everything, and the comparison is constant time.
- **The rate limiter was not atomic.** It read the counter, compared, then incremented, so
  concurrent requests all read the same value and all passed. It could also lose its expiry:
  if the key lapsed between the read and the `INCR`, the counter came back with no TTL and
  that client stayed limited forever. Now a single pipelined `INCR` plus `TTL`.
- **The circuit breaker never reset on success.** `failure_count` only cleared on the
  half-open to closed transition, so five unrelated failures spread over weeks eventually
  tripped a healthy backend. It also mutated shared state without a lock while Flask served
  requests concurrently.
- **Drift detection flattened every feature into one distribution**, so a shift in one
  feature could be cancelled by an opposite shift in another and the result was not
  interpretable even when it fired. Now one KS test per feature, with the threshold divided
  by the feature count, and the report names which features moved.
- `profile_inference` did not use `functools.wraps`, so decorating a Flask view erased its
  name and broke routing.

## Licence

All rights reserved. Published for reading, not for reuse.
