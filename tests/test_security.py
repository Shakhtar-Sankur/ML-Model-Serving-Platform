"""The security hole and the rate limiter, pinned.

Two of the defects in the README's correctness pass live here, and both are the
kind that leave a service looking like it works: an unconfigured deployment that
accepts every request, and a limiter that lets a burst straight through.
"""

import pytest

pytest.importorskip("tensorflow")

from ml_serving_platform import SecurityManager


# ── API keys ──────────────────────────────────────────────────────────────
def test_a_valid_key_is_accepted(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", "alpha,beta")
    assert SecurityManager.validate_api_key("alpha")
    assert SecurityManager.validate_api_key("beta")


def test_a_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", "alpha")
    assert not SecurityManager.validate_api_key("gamma")


def test_an_empty_key_does_not_authenticate(monkeypatch):
    """The defect: `'' in ''.split(',')` is True, so an empty key matched."""
    monkeypatch.setenv("VALID_API_KEYS", "alpha")
    assert not SecurityManager.validate_api_key("")
    assert not SecurityManager.validate_api_key(None)


def test_an_unconfigured_service_rejects_everything(monkeypatch):
    """Unset is the default deployment state; it must fail closed, not open."""
    monkeypatch.delenv("VALID_API_KEYS", raising=False)
    assert not SecurityManager.validate_api_key("")
    assert not SecurityManager.validate_api_key("anything")


def test_a_trailing_comma_does_not_create_an_empty_valid_key(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", "alpha,")
    assert not SecurityManager.validate_api_key("")
    assert SecurityManager.validate_api_key("alpha"), "the real key still works"


def test_whitespace_around_a_key_is_ignored(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", " alpha , beta ")
    assert SecurityManager.validate_api_key("alpha")


def test_a_prefix_of_a_valid_key_is_not_valid(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", "supersecret")
    assert not SecurityManager.validate_api_key("super")
    assert not SecurityManager.validate_api_key("supersecretplus")


# ── rate limiting ─────────────────────────────────────────────────────────
def test_requests_under_the_limit_pass(fake_redis):
    for _ in range(5):
        assert SecurityManager.rate_limit_check("client-a", max_requests=5, window=60)


def test_the_request_over_the_limit_is_refused(fake_redis):
    for _ in range(3):
        SecurityManager.rate_limit_check("client-a", max_requests=3, window=60)
    assert not SecurityManager.rate_limit_check("client-a", max_requests=3, window=60)


def test_each_client_gets_its_own_window(fake_redis):
    for _ in range(3):
        SecurityManager.rate_limit_check("noisy", max_requests=3, window=60)
    assert not SecurityManager.rate_limit_check("noisy", max_requests=3, window=60)
    assert SecurityManager.rate_limit_check("quiet", max_requests=3, window=60), \
        "one client's traffic limited another"


def test_the_counter_is_incremented_before_it_is_read(fake_redis):
    """The defect: read, compare, then increment let concurrent requests all pass.

    INCR must be the first command, and it must be the value the decision is
    made on — that is what makes the check atomic.
    """
    SecurityManager.rate_limit_check("client-a", max_requests=10, window=60)
    assert fake_redis.commands[0][0] == "incr", \
        f"first command was {fake_redis.commands[0][0]}, not incr"


def test_the_window_gets_an_expiry_when_it_is_created(fake_redis):
    SecurityManager.rate_limit_check("client-a", max_requests=10, window=900)
    expires = [c for c in fake_redis.commands if c[0] == "expire"]
    assert expires, "the rate limit key was created without a TTL and never resets"
    assert expires[0][2] == 900


def test_the_expiry_is_not_reset_on_every_request(fake_redis):
    """Refreshing the TTL each time turns a fixed window into a rolling ban."""
    SecurityManager.rate_limit_check("client-a", max_requests=10, window=900)
    before = len([c for c in fake_redis.commands if c[0] == "expire"])
    for _ in range(4):
        SecurityManager.rate_limit_check("client-a", max_requests=10, window=900)
    after = len([c for c in fake_redis.commands if c[0] == "expire"])
    assert after == before, "the window was extended by later requests"


def test_a_key_that_lost_its_expiry_gets_one_back(fake_redis):
    """If a key survives with no TTL, that client would be limited forever."""
    fake_redis.store["rate_limit:client-a"] = 7      # exists, no expiry recorded
    SecurityManager.rate_limit_check("client-a", max_requests=100, window=60)
    assert fake_redis.ttl("rate_limit:client-a") > 0


# ── input size ────────────────────────────────────────────────────────────
def test_a_small_image_is_accepted():
    import base64
    assert SecurityManager.sanitize_input(base64.b64encode(b"x" * 1024).decode())


def test_an_oversized_image_is_refused():
    import base64
    payload = base64.b64encode(b"x" * (11 * 1024 * 1024)).decode()
    assert not SecurityManager.sanitize_input(payload), "the 10 MB cap did not hold"


def test_data_that_is_not_base64_is_refused():
    assert not SecurityManager.sanitize_input("this is not base64 !!!")
