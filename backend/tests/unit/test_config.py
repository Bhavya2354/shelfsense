import pytest

from app.config import FredSettings, StorageSettings, _load
from app.errors import ConfigurationError


def test_missing_required_settings_fail_fast_and_name_the_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env file here
    for name in ("FRED_API_KEY", "FRED_BASE_URL", "FRED_OIL_SERIES_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError) as err:
        _load(FredSettings)
    message = str(err.value)
    assert "FRED_API_KEY" in message
    assert "FRED_BASE_URL" in message


def test_secret_values_never_appear_in_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FRED_API_KEY", "super-secret-value")
    monkeypatch.setenv("FRED_BASE_URL", "not a url")
    monkeypatch.setenv("FRED_OIL_SERIES_ID", "X")
    with pytest.raises(ConfigurationError) as err:
        _load(FredSettings)
    assert "super-secret-value" not in str(err.value)


def test_storage_paths_derive_from_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", "/srv/data")
    settings = _load(StorageSettings)
    assert settings.curated_dir.as_posix().endswith("srv/data/curated")
    assert settings.artifacts_dir.name == "artifacts"
