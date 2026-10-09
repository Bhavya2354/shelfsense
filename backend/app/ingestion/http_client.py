"""Shared JSON-over-HTTP client with bounded retries for upstream data APIs."""

import logging
from types import TracebackType
from typing import Any, Self

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


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning("retrying request", extra={"attempt": state.attempt_number, "error": repr(exc)})


class ApiClient:
    """Thin wrapper around `httpx.Client`; secrets passed as params never reach the logs."""

    def __init__(self, base_url: str, settings: HttpSettings) -> None:
        self._settings = settings
        self._http = httpx.Client(
            base_url=base_url,
            timeout=settings.http_timeout_seconds,
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._settings.http_max_retries + 1),
            wait=wait_exponential_jitter(initial=self._settings.http_backoff_seconds),
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
