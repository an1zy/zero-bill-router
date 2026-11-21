import asyncio
import shutil
import time
from pathlib import Path
from typing import Any
import httpx
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zero_bill.config import AppConfig, load_config
from zero_bill.server import create_app
from zero_bill.tracker import UsageTracker

app = typer.Typer(
    name="zero-bill",
    help="Zero-Bill Router: Local OpenAI-compatible proxy routing between local models and free cloud providers.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing zero-bill.yaml if it already exists",
    ),
) -> None:
    """Initialize a zero-bill.yaml configuration file from config.example.yaml."""
    target_path = Path("zero-bill.yaml")
    if target_path.exists() and not force:
        console.print(
            f"[yellow]File '{target_path}' already exists.[/yellow] Use [bold]--force[/bold] to overwrite."
        )
        raise typer.Exit(code=1)

    example_locations = [
        Path("config.example.yaml"),
        Path(__file__).resolve().parent.parent.parent / "config.example.yaml",
    ]
    source_path = next((p for p in example_locations if p.is_file()), None)

    if not source_path:
        console.print("[red]Error: config.example.yaml template not found![/red]")
        raise typer.Exit(code=1)

    shutil.copyfile(source_path, target_path)
    console.print(
        f"[green]Successfully created configuration file: [bold]{target_path}[/bold][/green]"
    )


async def _ping_provider(
    client: httpx.AsyncClient,
    name: str,
    target_type: str,
    base_url: str,
    api_key: str | None,
    priority: int | str,
) -> dict[str, str]:
    if target_type == "provider" and not api_key:
        return {
            "name": name,
            "type": target_type,
            "status": "[yellow]NO KEY[/yellow]",
            "latency": "-",
            "priority": str(priority),
        }

    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[:-len("/chat/completions")]
    url = f"{cleaned}/models"

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.perf_counter()
    try:
        response = await client.get(url, headers=headers, timeout=5.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if response.status_code == 200:
            status = "[green]OK[/green]"
        elif response.status_code in (401, 403):
            status = f"[red]ERROR ({response.status_code} Auth)[/red]"
        else:
            status = f"[red]ERROR ({response.status_code})[/red]"
        latency_str = f"{elapsed_ms:.1f} ms"
    except Exception:
        status = "[red]ERROR[/red]"
        latency_str = "timeout/fail"

    return {
        "name": name,
        "type": target_type,
        "status": status,
        "latency": latency_str,
        "priority": str(priority),
    }


async def _run_checks(config: AppConfig) -> list[dict[str, str]]:
    tasks = []
    async with httpx.AsyncClient() as client:
        if config.ollama.enabled:
            tasks.append(
                _ping_provider(
                    client=client,
                    name=f"ollama ({config.ollama.model})",
                    target_type="local",
                    base_url=config.ollama.base_url,
                    api_key=None,
                    priority=0,
                )
            )

        for provider in config.providers:
            tasks.append(
                _ping_provider(
                    client=client,
                    name=provider.name,
                    target_type="provider",
                    base_url=provider.base_url,
                    api_key=provider.api_key,
                    priority=provider.priority,
                )
            )

        results = await asyncio.gather(*tasks)
    return results


@app.command()
def check(
    config: Path = typer.Option(
        Path("zero-bill.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
) -> None:
    """Asynchronously check health and ping Ollama and all configured providers."""
    if not config.exists():
        if Path("config.example.yaml").exists():
            console.print(
                f"[yellow]Notice: '{config}' not found, reading from 'config.example.yaml'[/yellow]"
            )
            config = Path("config.example.yaml")
        else:
            console.print(f"[red]Error: Config file '{config}' not found.[/red]")
            raise typer.Exit(code=1)

    app_config = load_config(config)
    results = asyncio.run(_run_checks(app_config))

    table = Table(title="Zero-Bill Router: Provider Health Check", border_style="cyan")
    table.add_column("Target", style="bold")
    table.add_column("Type")
    table.add_column("Priority", justify="right")
    table.add_column("Status")
    table.add_column("Latency", justify="right")

    for item in results:
        table.add_row(
            item["name"],
            item["type"],
            item["priority"],
            item["status"],
            item["latency"],
        )

    console.print(table)


@app.command()
def serve(
    config: Path = typer.Option(
        Path("zero-bill.yaml"),
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    host: str | None = typer.Option(None, "--host", "-h", help="Override server host"),
    port: int | None = typer.Option(None, "--port", "-p", help="Override server port"),
) -> None:
    """Start the OpenAI-compatible Zero-Bill Router proxy server."""
    if not config.exists():
        if Path("config.example.yaml").exists():
            console.print(
                f"[yellow]Notice: '{config}' not found, reading from 'config.example.yaml'[/yellow]"
            )
            config = Path("config.example.yaml")
        else:
            console.print(f"[red]Error: Config file '{config}' not found.[/red]")
            raise typer.Exit(code=1)

    app_config = load_config(config)
    server_host = host or app_config.server.host
    server_port = port or app_config.server.port

    tracker = UsageTracker()
    fastapi_app = create_app(app_config, tracker=tracker)

    active_providers = [
        p.name
        for p in app_config.providers
        if p.active and p.is_active and bool(p.api_key)
    ]
    ollama_info = (
        f"Enabled ({app_config.ollama.model})"
        if app_config.ollama.enabled
        else "Disabled"
    )
    active_str = (
        ", ".join(active_providers)
        if active_providers
        else "None (Ollama only / all in cooldown)"
    )

    panel_content = (
        f"[bold cyan]Zero-Bill Router is running![/bold cyan]\n\n"
        f"[bold white]OpenAI Base URL:[/bold white] [bold green]http://{server_host}:{server_port}/v1[/bold green]\n"
        f"[bold white]Cursor / Claude Code / Cline configuration:[/bold white]\n"
        f"  [yellow]OPENAI_BASE_URL=http://{server_host}:{server_port}/v1[/yellow]\n"
        f"  [yellow]OPENAI_API_KEY=zero-bill[/yellow]\n\n"
        f"[bold]Ollama:[/bold] {ollama_info}\n"
        f"[bold]Active Cloud Providers:[/bold] {active_str}"
    )
    console.print(
        Panel(panel_content, title="zero-bill-router", border_style="cyan")
    )

    uvicorn.run(fastapi_app, host=server_host, port=server_port, log_level="info")


@app.command()
def stats(
    log_path: Path = typer.Option(
        Path.home() / ".zero-bill" / "usage.jsonl",
        "--log-path",
        help="Path to usage.jsonl file",
    ),
) -> None:
    """Read usage log and display aggregated statistics with dollar savings."""
    tracker = UsageTracker(log_path=log_path)
    summary = tracker.summary()

    if summary["total_requests"] == 0:
        console.print(f"[yellow]No requests logged yet in {log_path}[/yellow]")
        return

    table = Table(title="Zero-Bill Router: Usage by Provider", border_style="blue")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Actual Spent ($)", justify="right")
    table.add_column("Est. Saved ($)", justify="right", style="green")

    for prov, p_stats in summary["by_provider"].items():
        table.add_row(
            prov,
            str(p_stats["requests"]),
            f"{p_stats['prompt_tokens']:,}",
            f"{p_stats['completion_tokens']:,}",
            f"{p_stats['total_tokens']:,}",
            f"${p_stats['actual_spent_usd']:.4f}",
            f"${p_stats['saved_usd']:.4f}",
        )

    console.print(table)

    summary_panel = (
        f"[bold]Total Requests:[/bold] {summary['total_requests']} "
        f"([cyan]Local: {summary['local_requests']}[/cyan], [cyan]Free Cloud: {summary['free_api_requests']}[/cyan])\n"
        f"[bold]Total Tokens Processed:[/bold] {summary['total_tokens']:,}\n"
        f"[bold]Actual Spent:[/bold] [yellow]${summary['actual_spent_usd']:.4f}[/yellow]\n"
        f"[bold green]ESTIMATED TOTAL SAVED: ${summary['estimated_saved_usd']:.4f} USD[/bold green]"
    )
    console.print(Panel(summary_panel, title="Savings Summary", border_style="green"))


if __name__ == "__main__":
    app()
