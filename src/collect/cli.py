"""
cli.py – Typer CLI for the Azure OpenAI cost dashboard collector.

Commands:
  collect  – Collect last month's cost data across all subscriptions
  validate – Inspect and validate a collected parquet file
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.common.logging import configure_logging
from src.collect.cost_details import download_and_parse, generate_cost_details_report
from src.collect.html_report import generate_html_report
from src.collect.normalize import normalize_dataframe, parse_month_arg
from src.collect.retail_prices import fetch_retail_prices
from src.collect.subscriptions import list_subscriptions

app = typer.Typer(
    name="azure-cost",
    help="Collect Azure OpenAI cost data and validate the output.",
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def collect(
    month: str = typer.Option(
        "last",
        help="Month to collect.  'last' = previous calendar month; 'YYYY-MM' for specific month.",
    ),
    out: Path = typer.Option(
        Path("data/normalized/openai_cost_last_month.parquet"),
        help="Output parquet file path.",
    ),
    subscription: Optional[str] = typer.Option(
        None,
        help="Comma-separated subscription IDs to limit collection to.  Omit for all.",
    ),
    metric: str = typer.Option(
        "ActualCost",
        help="Cost metric: 'ActualCost' or 'AmortizedCost'.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Collect Azure OpenAI cost details for the specified month.

    Iterates over all accessible subscriptions (or the provided subset),
    downloads the Cost Details Report from Azure Cost Management, filters to
    Azure OpenAI rows, and writes a normalised parquet file.
    """
    configure_logging("DEBUG" if verbose else "INFO")

    # Parse month argument
    try:
        start_date, end_date = parse_month_arg(month)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]Collecting Azure OpenAI costs[/bold] for "
        f"[cyan]{start_date}[/cyan] → [cyan]{end_date}[/cyan]"
    )

    # Resolve subscriptions
    if subscription:
        from src.collect.subscriptions import Subscription

        sub_ids = [s.strip() for s in subscription.split(",") if s.strip()]
        subscriptions = [
            Subscription(subscription_id=s, display_name=s, state="Enabled")
            for s in sub_ids
        ]
        console.print(f"Using {len(subscriptions)} explicitly provided subscription(s)")
    else:
        console.print("Enumerating all accessible subscriptions …")
        try:
            subscriptions = list_subscriptions()
        except Exception as exc:
            console.print(f"[red]Failed to list subscriptions:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    if not subscriptions:
        console.print("[yellow]No accessible subscriptions found.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"Found [bold]{len(subscriptions)}[/bold] subscription(s)")

    # Optionally fetch retail prices for discount computation fallback
    console.print("Fetching retail prices baseline …")
    try:
        retail_prices = fetch_retail_prices(use_cache=True)
        console.print(f"  → {len(retail_prices)} retail price entries loaded")
    except Exception as exc:
        logger.warning("Could not load retail prices: %s", exc)
        retail_prices = {}

    all_frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for sub in subscriptions:
        sid = sub.subscription_id
        console.print(
            f"\n[bold]Processing:[/bold] {sub.display_name} ([dim]{sid}[/dim])"
        )
        try:
            manifest = generate_cost_details_report(
                subscription_id=sid,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                metric=metric,
            )
            raw_df = download_and_parse(subscription_id=sid, manifest=manifest)
            console.print(f"  → Downloaded {len(raw_df):,} raw rows")

            if raw_df.empty:
                console.print("  [yellow]→ No rows found[/yellow]")
                continue

            norm_df = normalize_dataframe(raw_df, retail_prices=retail_prices)
            console.print(
                f"  → {len(norm_df):,} Azure OpenAI rows after normalisation"
            )

            if not norm_df.empty:
                all_frames.append(norm_df)

        except Exception as exc:
            logger.error("[%s] Failed: %s", sid, exc, exc_info=verbose)
            failures.append((sid, str(exc)))
            console.print(f"  [red]→ FAILED:[/red] {exc}")
            continue

    if failures:
        console.print(
            f"\n[yellow]⚠ {len(failures)} subscription(s) failed:[/yellow]"
        )
        for sid, msg in failures:
            console.print(f"  [dim]{sid}[/dim]: {msg}")

    if not all_frames:
        console.print(
            "\n[yellow]No Azure OpenAI data found across all subscriptions.[/yellow]"
        )
        raise typer.Exit(code=0)

    # Combine and write parquet
    combined = pd.concat(all_frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)

    console.print(
        f"\n[green]✓ Wrote {len(combined):,} rows to[/green] [bold]{out}[/bold]"
    )

    _print_summary(combined)


@app.command()
def validate(
    file: Path = typer.Option(
        Path("data/normalized/openai_cost_last_month.parquet"),
        help="Parquet file to inspect.",
    ),
) -> None:
    """Validate and display a summary of the collected parquet file."""
    configure_logging("INFO")

    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(file)
    console.print(f"\n[bold]{file}[/bold]  ({len(df):,} rows, {len(df.columns)} columns)\n")

    _print_summary(df)

    console.print("\n[bold]Columns:[/bold]")
    for col in df.columns:
        console.print(f"  • {col}")


@app.command("export-html")
def export_html(
    file: Path = typer.Option(
        Path("data/normalized/openai_cost_last_month.parquet"),
        help="Parquet file to read.",
    ),
    out: Path = typer.Option(
        Path("data/reports/openai_cost_report.html"),
        help="Output self-contained HTML file path.",
    ),
    title: str = typer.Option(
        "Azure OpenAI Cost Report",
        help="HTML document title.",
    ),
) -> None:
    """Export a self-contained HTML report with dashboard metrics and filters."""
    configure_logging("INFO")

    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=1)

    try:
        input_rows, output_rows = generate_html_report(
            input_file=file,
            output_file=out,
            title=title,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except PermissionError as exc:
        console.print(f"[red]Permission denied while writing report:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]File system error while generating report:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Failed to generate HTML report")
        console.print("[red]Failed to generate HTML report. Check logs for details.[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]✓ Wrote HTML report:[/green] [bold]{out}[/bold]\n"
        f"Rows processed: [bold]{output_rows:,}[/bold] (input: {input_rows:,})"
    )


def _print_summary(df: pd.DataFrame) -> None:
    """Print a Rich summary table for the collected DataFrame."""
    if df.empty:
        console.print("[yellow]Dataset is empty.[/yellow]")
        return

    table = Table(title="Azure OpenAI Cost Summary", show_header=True)
    table.add_column("Subscription", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Cached Input", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Avg Eff. Price / 1M", justify="right")

    qty_col = "total_quantity" if "total_quantity" in df.columns else "quantity"
    cost_col = "total_cost" if "total_cost" in df.columns else "cost"
    eff_col = (
        "effective_price_per_1m"
        if "effective_price_per_1m" in df.columns
        else "avg_effective_price"
    )
    sub_col = "subscription_id" if "subscription_id" in df.columns else None

    def _tokens_for_type(grp: pd.DataFrame, token_type: str) -> float:
        if "token_type" not in grp.columns or qty_col not in grp.columns:
            return 0.0
        return float(grp.loc[grp["token_type"] == token_type, qty_col].sum())

    if sub_col:
        for sub_id, grp in df.groupby(sub_col):
            total_qty = (
                grp[qty_col].sum() if qty_col in grp.columns else 0
            )
            total_cost = (
                grp[cost_col].sum() if cost_col in grp.columns else 0
            )
            avg_eff = (
                grp[eff_col].mean() if eff_col in grp.columns else 0
            )
            currency = (
                grp["currency"].iloc[0]
                if "currency" in grp.columns and not grp.empty
                else ""
            )
            table.add_row(
                str(sub_id)[:40],
                f"{len(grp):,}",
                f"{total_qty:,.0f}",
                f"{_tokens_for_type(grp, 'Cached Input Tokens'):,.0f}",
                f"{_tokens_for_type(grp, 'Input Tokens'):,.0f}",
                f"{_tokens_for_type(grp, 'Output Tokens'):,.0f}",
                f"{total_cost:,.4f} {currency}",
                f"{avg_eff:,.6f}",
            )
    else:
        table.add_row(
            "all",
            f"{len(df):,}",
            f"{df[qty_col].sum():,.0f}" if qty_col in df.columns else "n/a",
            f"{_tokens_for_type(df, 'Cached Input Tokens'):,.0f}",
            f"{_tokens_for_type(df, 'Input Tokens'):,.0f}",
            f"{_tokens_for_type(df, 'Output Tokens'):,.0f}",
            f"{df[cost_col].sum():,.4f}" if cost_col in df.columns else "n/a",
            "n/a",
        )

    console.print(table)


if __name__ == "__main__":
    app()
