"""Command-line entry point: `shelfsense <command>`."""

from collections.abc import Callable

import typer

from app import config
from app.errors import ConfigurationError
from app.observability import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)

_SECTIONS: dict[str, Callable[[], object]] = {
    "runtime": config.runtime_settings,
    "storage": config.storage_settings,
    "http": config.http_settings,
    "kaggle": config.kaggle_settings,
    "fred": config.fred_settings,
    "open-meteo": config.open_meteo_settings,
}


@app.callback()
def main() -> None:
    configure_logging(config.runtime_settings())


@app.command("check-config")
def check_config() -> None:
    """Validate every settings section without printing any values."""
    failed = False
    for name, load in _SECTIONS.items():
        try:
            load()
            typer.echo(f"ok      {name}")
        except ConfigurationError as exc:
            failed = True
            typer.echo(f"missing {exc}")
    if failed:
        raise typer.Exit(code=1)
