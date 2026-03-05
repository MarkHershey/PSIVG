import os

from rich import print


def print_section(text):
    try:
        console_width = os.get_terminal_size().columns
    except:
        console_width = 100
    print("\n")
    print("[green]" + "=" * console_width)
    print(f"🚀 [bold cyan]{text}[/bold cyan]")
    print("[green]" + "=" * console_width)
