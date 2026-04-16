import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Initialize Rich Console
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

logger = logging.getLogger("kafi-agent")

# Reduce noise from dependency loggers (Hugging Face hub / http clients).
for _name in ("httpx", "huggingface_hub", "transformers", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def log_tool_call(tool_name: str, args: dict, result: str):
    """Logs a tool call in a nice table format."""
    table = Table(title=f"Tool: {tool_name}", show_header=False, border_style="cyan")
    table.add_row("Args", str(args))
    table.add_row("Result", result[:200] + "..." if len(result) > 200 else result)
    console.print(table)


def log_agent_response(agent_name: str, response: str):
    """Logs the agent's response in a colored panel."""
    panel = Panel(
        Text(response, style="green"),
        title=f"Assistant: {agent_name}",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def log_user_input(question: str, conversation_id: str = "N/A"):
    """Logs the user input in a yellow panel."""
    panel = Panel(
        Text(question, style="bold white"),
        title=f"User Input ({conversation_id})",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(panel)


def log_delegation(from_agent: str, to_agent: str, question: str):
    """Logs a delegation between agents."""
    console.print(
        f"\n[bold yellow]{from_agent} -> {to_agent}[/bold yellow]: {question[:80]}...\n"
    )


def log_llm_metrics(
    *,
    model_id: str,
    ttft_s: float | None,
    total_s: float | None,
    output_tokens: int | None,
    output_tokens_per_s: float | None,
    aborted: bool = False,
):
    """Logs per-response LLM performance metrics (terminal only)."""
    table = Table(title="LLM Metrics", show_header=False, border_style="magenta")
    table.add_row("Model", str(model_id))
    table.add_row("TTFT", f"{ttft_s:.3f}s" if ttft_s is not None else "N/A")
    table.add_row("Total", f"{total_s:.3f}s" if total_s is not None else "N/A")
    table.add_row("Output tokens", str(output_tokens) if output_tokens is not None else "N/A")
    table.add_row(
        "Tokens/s",
        f"{output_tokens_per_s:.2f}" if output_tokens_per_s is not None else "N/A",
    )
    table.add_row("Aborted", "yes" if aborted else "no")
    console.print(table)
