import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure one simple, process-wide logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )