"""Logging conventions.

Library modules log through ``logging.getLogger(__name__)`` (all under the ``perseus``
logger, which carries a NullHandler) and never print. Diagnostics are INFO, anything
a user should act on is WARNING. Nothing is shown unless the host application
configures logging — the standard library contract — except that Python's last-resort
handler still surfaces WARNING and above on stderr.

    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")   # see everything

The console scripts call ``configure_cli_logging()`` to format their own output.
"""
import logging
import sys


def configure_cli_logging(level=logging.INFO, stream=None):
    """Attach a plain ``%(message)s`` handler to the ``perseus`` logger for a CLI run
    (idempotent: a second call only adjusts the level)."""
    logger = logging.getLogger("perseus")
    stream = stream or sys.stdout
    for h in logger.handlers:
        if getattr(h, "_perseus_cli", False):
            logger.setLevel(level)
            return logger
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._perseus_cli = True
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
