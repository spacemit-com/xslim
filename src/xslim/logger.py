import sys

from loguru import logger

logger.remove()
logger.add(
    sink=sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | <lvl>{name:24}</> | <lvl>{level:8}</> | <lvl>{message}</>",
    colorize=True,
    level="INFO",
)


def xslim_error(info: str):
    logger.error(info)


def xslim_warning(info: str):
    logger.warning(info)


def xslim_info(info: str):
    logger.info(info)


def xslim_debug(info: str):
    logger.debug(info)


def xslim_trace(info: str):
    logger.trace(info)
