"""Test environment: no Redis, no Triton, no AWS.

`ml_serving_platform` builds its Flask app, its Redis client and its Triton
client at import time, so importing it is unavoidable — but none of those
connect until they are used. What the fixtures here do is replace the one that
*is* used by the code under test (Redis) with an in-memory stand-in, and give
the API key checker something to check against.

The module imports TensorFlow at the top, which is why every test file starts
with an `importorskip`. The suite therefore runs in full on CI, where
`tensorflow-cpu` is installed, and skips cleanly on a machine without it.
"""

import os
import sys
import time
from pathlib import Path

import pytest

# The service is a single module at the repository root, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Invalid by design: if a test ever reached AWS it would fail loudly.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")


class FakeRedis:
    """Enough Redis to exercise the rate limiter, including TTL semantics.

    `ttl` returns -2 for a key that does not exist and -1 for one with no
    expiry, as real Redis does — the rate limiter branches on exactly that.
    """

    def __init__(self):
        self.store = {}
        self.expiry = {}
        self.commands = []

    # -- plain commands -------------------------------------------------
    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.expiry[key] = time.time() + ttl

    def incr(self, key):
        self.commands.append(("incr", key))
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    def ttl(self, key):
        self.commands.append(("ttl", key))
        if key not in self.store:
            return -2
        if key not in self.expiry:
            return -1
        return max(0, int(self.expiry[key] - time.time()))

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        self.expiry[key] = time.time() + seconds
        return True

    def ping(self):
        return True

    def info(self):
        return {"keyspace_hits": 0, "keyspace_misses": 0}

    # -- pipeline -------------------------------------------------------
    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    """Queues commands and runs them together, returning a list of results."""

    def __init__(self, client):
        self.client = client
        self.queued = []

    def incr(self, key):
        self.queued.append(("incr", key))
        return self

    def ttl(self, key):
        self.queued.append(("ttl", key))
        return self

    def execute(self):
        return [getattr(self.client, name)(*args) for name, *args in self.queued]


@pytest.fixture
def fake_redis(monkeypatch):
    """Point the module's cache manager at an in-memory store."""
    platform = pytest.importorskip("ml_serving_platform")
    client = FakeRedis()
    monkeypatch.setattr(platform.cache_manager, "redis_client", client)
    return client


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", "good-key,second-key")
    return "good-key"


@pytest.fixture
def client(fake_redis, api_key):
    """A Flask test client with authentication and rate limiting live."""
    platform = pytest.importorskip("ml_serving_platform")
    platform.app.config["TESTING"] = True
    with platform.app.test_client() as test_client:
        yield test_client
