"""Shared JSON-over-HTTP client with bounded retries for upstream data APIs."""

import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import HttpSettings
from app.errors import ExternalServiceError

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """Seconds the server asked us to wait (only the delta-seconds form)."""
    if isinstance(exc, httpx.HTTPStatusError):
        value = exc.response.headers.get("retry-after", "")
        if value.isdigit():
            return float(value)
    return None


class _WaitRespectingServer:
    """Use the server's Retry-After when given, otherwise jittered exponential backoff."""

    def __init__(self, backoff: float, ceiling: float) -> None:
        self._fallback = wait_exponential_jitter(multiplier=backoff, max=ceiling)
        self._ceiling = ceiling

    def __call__(self, state: RetryCallState) -> float:
        hinted = _retry_after_seconds(state.outcome.exception() if state.outcome else None)
        return min(hinted, self._ceiling) if hinted is not None else self._fallback(state)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning("retrying request", extra={"attempt": state.attempt_number, "error": repr(exc)})


class ApiClient:
    """Thin wrapper around `httpx.Client`; secrets passed as params never reach the logs."""

    def __init__(
        self, base_url: str, settings: HttpSettings, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._http = httpx.Client(
            base_url=base_url,
            timeout=settings.http_timeout_seconds,
            headers={"Accept": "application/json"},
            follow_redirects=True,
            transport=transport,
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._settings.http_max_retries + 1),
            wait=_WaitRespectingServer(
                self._settings.http_backoff_seconds, self._settings.http_max_backoff_seconds
            ),
            before_sleep=_log_retry,
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    response = self._http.get(path, params=params)
                    response.raise_for_status()
                    return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise ExternalServiceError(f"{exc.request.url.host} returned {status}") from None
        except httpx.TransportError as exc:
            raise ExternalServiceError(f"{type(exc).__name__} calling {path}") from None
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class _StalledError(Exception):
    """Throughput fell below the floor; reconnecting usually restores it."""


def download_file(
    url: str, target: Path, settings: HttpSettings, transport: httpx.BaseTransport | None = None
) -> Path:
    """Stream a large file to disk, resuming from a `.part` file after failures or stalls.

    Some CDNs throttle long-lived connections; a fresh ranged request restores
    full speed, so slow streams are dropped and resumed rather than waited out.
    """
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(settings.http_timeout_seconds, read=settings.http_timeout_seconds * 4)
    client = httpx.Client(timeout=timeout, follow_redirects=True, transport=transport)
    try:
        _download_loop(client, url, partial, settings)
    finally:
        client.close()
    partial.replace(target)
    return target


def _download_loop(client: httpx.Client, url: str, partial: Path, settings: HttpSettings) -> None:
    failures = 0
    for _ in range(settings.download_max_reconnects):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
                    break
                response.raise_for_status()
                resumed = response.status_code == httpx.codes.PARTIAL_CONTENT
                with partial.open("ab" if resumed else "wb") as sink:
                    _copy_watching_speed(response, sink, settings)
            break
        except _StalledError:
            logger.info("download throttled, reconnecting", extra={"bytes": partial.stat().st_size})
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            failures += 1
            if failures > settings.http_max_retries or not _is_retryable(exc):
                raise ExternalServiceError(f"download failed: {type(exc).__name__}") from None
            logger.warning("download interrupted, resuming", extra={"attempt": failures})
            time.sleep(settings.http_backoff_seconds * 2**failures)
    else:
        raise ExternalServiceError("download did not finish within the reconnect budget")


def _copy_watching_speed(response: httpx.Response, sink: BinaryIO, settings: HttpSettings) -> None:
    window_start, window_bytes = time.monotonic(), 0
    for chunk in response.iter_bytes(1 << 20):
        sink.write(chunk)
        window_bytes += len(chunk)
        elapsed = time.monotonic() - window_start
        if elapsed >= settings.download_stall_window_seconds:
            if window_bytes / elapsed < settings.download_min_bytes_per_second:
                raise _StalledError
            window_start, window_bytes = time.monotonic(), 0
