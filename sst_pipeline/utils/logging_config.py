"""
logging_config.py
-----------------
Centralised logging setup. Every pipeline module calls get_logger(__name__)
instead of print(), so all output is timestamped, levelled, and routable
to both console and a session log file.
"""

import logging
import os
import sys
from datetime import datetime, timezone


def setup_logging(output_dir: str = ".", level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logger with console + file handlers.
    Call once from pipeline.py at startup; all child loggers inherit it.
    """
    log_path = os.path.join(output_dir, f"pipeline_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.log")
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-35s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        h.close()
    root.handlers.clear()

    # Console — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File — DEBUG and above
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    for noisy in ("matplotlib", "PIL", "astroquery", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root.info("Pipeline log started → %s", log_path)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Use as: log = get_logger(__name__)"""
    return logging.getLogger(name)
