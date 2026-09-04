"""draftreasoner: a tool-augmented VLM agent for mechanical drawing understanding."""

__version__ = "0.1.0"
__all__ = ["__version__", "main"]


def main() -> None:
    from draftreasoner.cli.app import app

    app()
