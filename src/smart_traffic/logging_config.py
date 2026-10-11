"""Structured logging that writes to both console and a rotating log file.

Streamlit swallows plain ``stdout`` in some contexts, so the file handler is the
reliable place to keep technical detail. The UI layer reads nothing from here --
it shows curated messages and lets operators find the trace in ``logs/``.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False
LOGGER_NAME = "smart_traffic"


class _RedactingFilter(logging.Filter):
    """Strip anything that looks like an absolute user path from log records.

    Log files may be shared or committed by accident; home-directory paths are
    not useful there and can leak a username.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        message = record.getMessage()
        home = str(Path.home())
        if home and home != "/" and home in message:
            record.msg = message.replace(home, "~")
            record.args = ()
        return True


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Configure and return the package logger.

    Args:
        level: Minimum level for the console handler.
        log_file: Optional path for a rotating file handler. Parent directories
            are created if needed. Failures to open the file are tolerated --
            logging must never be the reason the app fails to start.

    Returns:
        The configured ``smart_traffic`` logger.
    """
    global _CONFIGURED

    logger = logging.getLogger(LOGGER_NAME)
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = _RedactingFilter()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(redactor)
    logger.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(redactor)
            logger.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - environment dependent
            logger.warning("File logging disabled (%s)", exc)

    _CONFIGURED = True
    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return a child of the package logger.

    Args:
        suffix: Optional module name, e.g. ``"ingestion"``.

    Returns:
        A ``logging.Logger`` scoped to this module.
    """
    base = setup_logging()
    return base.getChild(suffix) if suffix else base
