import logging
import os

class LoggerMixin:
    def __init__(self):
        # Use the actual filename if run as __main__, else use module path
        module = self.__module__
        if module == "__main__":
            # Get the filename without extension
            import sys
            filename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            logger_name = f"{filename}.{self.__class__.__name__}"
        else:
            logger_name = f"{module}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)
