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
`deploy.sh` for image build and Kubernetes rollout, and the `tests/` suite.

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
pytest tests -q
```

## Tests

```bash
pytest tests -q            # 65 tests, no Redis, Triton or AWS required
```

Redis is replaced with an in-memory stand-in that reproduces its TTL semantics — `-2` for a
missing key, `-1` for one with no expiry — because the rate limiter branches on exactly
that. Triton is a stub, and the ResNet50 backbone is swapped for a tiny model of the same
shape so no run downloads 100 MB of ImageNet weights.

The defects listed under Status are each pinned by a test that fails against the original
code: an empty API key authenticating, the read-compare-increment rate limiter, the
breaker that never reset on success, and drift in two features cancelling out.

TensorFlow is imported at the top of the service module, so the suite skips cleanly on a
machine without it. CI installs `tensorflow-cpu` and runs it in full on Python 3.10 and
3.11.

## Status

Complete service implementation with tests. Requires a Triton backend and a Redis instance;
no trained model is bundled.

### Notes from a correctness pass

Six defects fixed, one of them a security hole:

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
- **`/predict` and `/batch_predict` are async views, and `asgiref` was not a dependency.**
  Flask refuses to run an async view without it, so both prediction endpoints raised on
  every request in a clean install. `werkzeug` was unpinned next to a pinned
  `flask==2.3.3` as well, so a fresh install resolved a 3.x release that Flask cannot use.
  Both are pinned in `requirements.txt` now.

## Licence

Licensed under the GNU Affero General Public License v3.0. See `LICENSE`.

In short: you may use, modify and redistribute this, including over a network,
provided your derivative is released under the same licence.
