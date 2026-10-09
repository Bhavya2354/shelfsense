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
    "favorita": config.favorita_settings,
    "fred": config.fred_settings,
    "open-meteo": config.open_meteo_settings,
    "database": config.database_settings,
    "analysis": config.analysis_settings,
    "forecast": config.forecast_settings,
    "inventory": config.inventory_settings,
    "publish": config.publish_settings,
    "api": config.api_settings,
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


@app.command()
def analyze() -> None:
    """Run the statistical analyses and store the findings."""
    from app.pipelines.analysis import run_analysis

    for name, findings in run_analysis().items():
        typer.echo(f"{name:<20} {len(findings)} findings")


@app.command()
def train() -> None:
    """Backtest every model, then train the release forecast."""
    from app.pipelines.training import run_training

    report = run_training()
    typer.echo(f"best item model: {report['best_item_model']}")
    for model, scores in sorted(report["item_scores"].items(), key=lambda kv: kv[1]["nwrmsle"]):
        typer.echo(f"  {model:<18} NWRMSLE {scores['nwrmsle']:.4f}  WAPE {scores['wape']:.3f}")


@app.command()
def publish() -> None:
    """Load the latest training artifacts into Postgres as the current release."""
    from app.pipelines.publishing import run_publishing

    result = run_publishing()
    typer.echo(f"release {result['release_id']}")
    for table, rows in result["rows"].items():
        typer.echo(f"  {table:<22} {rows:>10,} rows")


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
