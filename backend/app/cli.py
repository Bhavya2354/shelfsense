"""Command-line entry point: `shelfsense <command>`."""

from collections.abc import Callable
from typing import Annotated

import truststore
import typer

from app import config
from app.errors import ConfigurationError
from app.observability import configure_logging
from app.pipelines.ingestion import Source

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
    # Use the OS certificate store so TLS works behind inspecting proxies and AV software.
    truststore.inject_into_ssl()


@app.command()
def ingest(
    only: Annotated[list[Source] | None, typer.Option(help="Limit to these sources.")] = None,
    force: Annotated[bool, typer.Option(help="Rebuild tables that already exist.")] = False,
) -> None:
    """Download and curate all raw data, then run the quality gate."""
    from app.pipelines.ingestion import run_ingestion

    results = run_ingestion(set(only or Source), force=force)
    for r in results:
        typer.echo(f"{'pass' if r.passed else r.severity:<8} {r.name:<28} {r.violations}")


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
