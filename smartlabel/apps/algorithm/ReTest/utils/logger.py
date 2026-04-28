import logging
import colorlog # type: ignore


def setup_color_logger() -> logging.Logger:
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG) # Set the logging level to INFO if you are not debugging

    if not logger.handlers:
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
