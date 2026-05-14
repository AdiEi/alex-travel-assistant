from rich.console import Console
from rich.panel import Panel
from assistant import TravelAssistant

console = Console()

BANNER = "  ✈  Alex — Your Personal Travel Assistant\n  Plan trips, discover destinations, pack smart."

COMMANDS = {
    "quit", "exit", "bye",
    "reset",
}


def print_welcome() -> None:
    console.print(Panel(BANNER, style="bold blue", expand=False))
    console.print(
        "Commands: [bold]reset[/bold] to start over, [bold]quit[/bold] to exit.\n"
    )


def main() -> None:
    print_welcome()
    assistant = TravelAssistant()

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[bold blue]Alex:[/bold blue] Safe travels! ✈\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            console.print("\n[bold blue]Alex:[/bold blue] Safe travels! ✈\n")
            break

        if user_input.lower() == "reset":
            assistant.reset()
            console.print("[dim]Conversation cleared — fresh start![/dim]\n")
            continue

        with console.status("[dim]Alex is thinking...[/dim]", spinner="dots"):
            reply = assistant.chat(user_input)

        console.print(f"\n[bold blue]Alex:[/bold blue] {reply}\n")


if __name__ == "__main__":
    main()
