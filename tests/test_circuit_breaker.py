"""The breaker: when it trips, when it recovers, and what it counts.

The defect the README records is that `failure_count` only cleared on the
half-open to closed transition. A healthy backend that failed five times over
five weeks — unrelated failures, each followed by thousands of successes —
eventually tripped. `test_a_success_clears_the_count` is that scenario.
"""

import threading
import time

import pytest

pytest.importorskip("tensorflow")

from ml_serving_platform import CircuitBreaker, CircuitBreakerOpen


def boom():
    raise RuntimeError("backend down")


def fine():
    return "ok"


def test_a_working_call_returns_its_value():
    assert CircuitBreaker().call(fine) == "ok"


def test_arguments_are_passed_through():
    breaker = CircuitBreaker()
    assert breaker.call(lambda a, b=0: a + b, 2, b=3) == 5


def test_the_underlying_error_is_not_swallowed():
    breaker = CircuitBreaker(failure_threshold=5)
    with pytest.raises(RuntimeError, match="backend down"):
        breaker.call(boom)
    assert breaker.state == "CLOSED", "one failure is not enough to trip"


def test_it_trips_at_the_threshold():
    breaker = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    assert breaker.state == "OPEN"


def test_an_open_breaker_does_not_attempt_the_call():
    breaker = CircuitBreaker(failure_threshold=2, timeout=60)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(boom)

    attempts = []
    with pytest.raises(CircuitBreakerOpen):
        breaker.call(lambda: attempts.append(1))
    assert not attempts, "the breaker is open but the backend was still called"


def test_a_success_clears_the_count():
    """The defect: failures accumulated forever across healthy traffic."""
    breaker = CircuitBreaker(failure_threshold=3)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    breaker.call(fine)
    assert breaker.failure_count == 0, "a success did not reset the counter"

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    assert breaker.state == "CLOSED", \
        "two failures, a success, then two more failures tripped a 3-failure breaker"


def test_it_half_opens_after_the_timeout_and_recovers():
    breaker = CircuitBreaker(failure_threshold=1, timeout=0.05)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state == "OPEN"

    time.sleep(0.06)
    assert breaker.call(fine) == "ok", "the breaker never gave the backend another chance"
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


def test_a_failure_while_half_open_opens_it_again():
    breaker = CircuitBreaker(failure_threshold=1, timeout=0.05)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    time.sleep(0.06)

    with pytest.raises(RuntimeError):
        breaker.call(boom)          # the trial call fails
    assert breaker.state == "OPEN", "a failed trial call left the breaker half-open"


def test_the_timeout_is_measured_from_the_last_failure():
    breaker = CircuitBreaker(failure_threshold=1, timeout=30)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    with pytest.raises(CircuitBreakerOpen):
        breaker.call(fine)          # well inside the timeout


def test_concurrent_failures_are_all_counted():
    """Flask serves requests from several threads; the counter is shared state."""
    breaker = CircuitBreaker(failure_threshold=10_000)   # high, so it stays closed
    threads = []

    def hammer():
        for _ in range(50):
            try:
                breaker.call(boom)
            except RuntimeError:
                pass

    for _ in range(8):
        thread = threading.Thread(target=hammer)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    assert breaker.failure_count == 400, \
        f"counted {breaker.failure_count} of 400 failures — updates were lost to a race"
