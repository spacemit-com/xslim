import sys
from loguru import logger

logger.remove()
logger.add(
    sink=sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | <lvl>{name:13}</> | <lvl>{level:8}</> | <lvl>{message}</>",
    colorize=True,
    level="INFO",
)


def xquant_error(info: str):
    logger.error(info)


def xquant_warning(info: str):
    logger.warning(info)


def xquant_info(info: str):
    logger.info(info)


def xquant_debug(info: str):
    logger.debug(info)


def xquant_trace(info: str):
    logger.trace(info)
