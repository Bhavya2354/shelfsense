from pathlib import Path

import httpx
import pytest

from app.config import HttpSettings
from app.errors import ExternalServiceError
from app.ingestion.http_client import ApiClient, download_file

FAST = HttpSettings(http_timeout_seconds=5, http_max_retries=2, http_backoff_seconds=0.001)


def test_retries_transient_errors_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": True})

    with ApiClient("https://api.test/", FAST, transport=httpx.MockTransport(handler)) as client:
        assert client.get_json("x") == {"ok": True}
    assert calls["n"] == 3


def test_does_not_retry_client_errors_and_hides_secrets() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400)

    with (
        ApiClient("https://api.test/", FAST, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ExternalServiceError) as err,
    ):
        client.get_json("series", {"api_key": "secret-key"})
    assert calls["n"] == 1
    assert "secret-key" not in str(err.value)


def test_download_resumes_from_partial_file(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 64
    seen_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("range")
        seen_ranges.append(header)
        start = int(header.split("=")[1].rstrip("-")) if header else 0
        status = 206 if header else 200
        return httpx.Response(status, content=payload[start:])

    target = tmp_path / "archive.zip"
    (tmp_path / "archive.zip.part").write_bytes(payload[:1000])
    download_file("https://cdn.test/file", target, FAST, transport=httpx.MockTransport(handler))
    assert target.read_bytes() == payload
    assert seen_ranges == ["bytes=1000-"]
