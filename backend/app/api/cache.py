"""Bounded in-process TTL cache.

Keys always include the current release id, so publishing a new release makes
every older entry unreachable without explicit invalidation.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry and entry[0] > now:
                self._data.move_to_end(key)
                self.hits += 1
                return entry[1]
        value = compute()
        with self._lock:
            self.misses += 1
            if self._ttl > 0:
                self._data[key] = (now + self._ttl, value)
                self._data.move_to_end(key)
                while len(self._data) > self._max:
                    self._data.popitem(last=False)
        return value
