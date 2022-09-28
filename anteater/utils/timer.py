import time
from functools import wraps

from anteater.utils.log import logger


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        clock = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{args[0].__class__.__name__}.{func.__name__} spends {time.time() - clock} seconds!")
        return result
    return wrapper
