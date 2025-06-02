import functools
import logging
import inspect

def log_method_calls(method):
    """Decorator to log method entry, exit, and exceptions at DEBUG/ERROR level, including all argument names/values and return values."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Use class-based logger if available, else fallback to class/module logger
        try:
            logger = getattr(self, 'logger')
        except AttributeError:
            logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        # Only use class name once in the log message
        method_name = method.__name__
        sig = inspect.signature(method)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arg_str = ', '.join(f"{k}={v!r}" for k, v in list(bound.arguments.items())[1:])  # skip 'self'
        # Only log args if present
        entry_msg = f"[ENTRY] " + (f" args: {arg_str}" if arg_str else "args: None")
        logger.debug(f"[{method_name}] {entry_msg}")
        try:
            result = method(self, *args, **kwargs)
            logger.debug(f"[{method_name}] [EXIT] returned: {repr(result)}")
            return result
        except Exception as e:
            logger.error(f"[{method_name}] [EXCEPTION] Error: {e}")
            raise
    return wrapper
