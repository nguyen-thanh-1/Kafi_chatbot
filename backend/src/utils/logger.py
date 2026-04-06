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

def log_tool_call(tool_name: str, args: dict, result: str):
    """Logs a tool call in a nice table format."""
    table = Table(title=f"🛠️ Tool: {tool_name}", show_header=False, border_style="cyan")
    table.add_row("Args", str(args))
    table.add_row("Result", result[:200] + "..." if len(result) > 200 else result)
    console.print(table)

def log_agent_response(agent_name: str, response: str):
    """Logs the agent's response in a colored panel."""
    panel = Panel(
        Text(response, style="green"), 
        title=f"🤖 {agent_name}", 
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)

def log_user_input(question: str, conversation_id: str = "N/A"):
    """Logs the user input in a yellow panel."""
    panel = Panel(
        Text(question, style="bold white"),
        title=f"👤 User Input ({conversation_id})",
        border_style="yellow",
        padding=(1, 2)
    )
    console.print(panel)

def log_delegation(from_agent: str, to_agent: str, question: str):
    """Logs a delegation between agents."""
    console.print(f"\n[bold yellow]🔄 {from_agent} -> {to_agent}[/bold yellow]: {question[:80]}...\n")
