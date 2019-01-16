"""Used to make pytest functions available globally"""
import functools
import time

test_length = 0


# test_length = 10


def test_choice(*args):
    if test_length == 0:
        return args[0]
    elif test_length == 10:
        return args[1]


def retry(exception, tries=4, delay=0.001, backoff=1):
    """Call the decorated function catching `exception` for `tries` time.

    Args:
        exception (Exception):
        tries (int): The number of tries to try
        delay (float): the delay in seconds between the  function calls
        backoff (float): Factor to increase/decrease the delay time each call
    """
    if not tries >= 1:
        raise ValueError("Number of tries has to be >= 1")

    def deco_retry(func):

        @functools.wraps(func)
        def func_retry(*args, **kwargs):
            delay_time = delay
            for i in range(start=tries, stop=0, step=-1):
                try:
                    return func(*args, **kwargs)
                except exception as error:
                    time.sleep(delay_time)

                    delay_time *= backoff
            else:
                raise error

        return func_retry

    return deco_retry
