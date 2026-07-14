import inspect
import logging


def log_for_0(msg, *args, level=logging.INFO):
    """Log a message using the caller's module logger."""
    caller_module = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller_module).log(level, msg, *args)
