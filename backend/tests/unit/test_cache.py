import time

from app.api.cache import TTLCache


def test_hits_until_expiry() -> None:
    cache = TTLCache(ttl_seconds=1, max_entries=10)
    calls = []
    compute = lambda: calls.append(1) or len(calls)  # noqa: E731
    assert cache.get_or_compute("k", compute) == 1
    assert cache.get_or_compute("k", compute) == 1
    assert cache.hits == 1
    time.sleep(1.05)
    assert cache.get_or_compute("k", compute) == 2


def test_evicts_least_recently_used() -> None:
    cache = TTLCache(ttl_seconds=60, max_entries=2)
    cache.get_or_compute("a", lambda: 1)
    cache.get_or_compute("b", lambda: 2)
    cache.get_or_compute("a", lambda: 1)  # refresh "a"
    cache.get_or_compute("c", lambda: 3)  # evicts "b"
    assert cache.get_or_compute("a", lambda: "stale") == 1
    assert cache.get_or_compute("b", lambda: "recomputed") == "recomputed"


def test_zero_ttl_disables_caching() -> None:
    cache = TTLCache(ttl_seconds=0, max_entries=10)
    values = iter([1, 2])
    assert cache.get_or_compute("k", lambda: next(values)) == 1
    assert cache.get_or_compute("k", lambda: next(values)) == 2
