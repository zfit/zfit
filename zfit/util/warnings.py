#  Copyright (c) 2020 zfit
import functools
import warnings


def warn_experimental_feature(func):
    warned = False

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        nonlocal warned
        if not warned:
            warnings.warn(f"The function {func} is EXPERIMENTAL and likely to break in the future!"
                          f" Use it with caution and feedback (Gitter, e-mail, "
                          f"https://github.com/zfit/zfit/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)"
                          f" is very welcome!")
            warned = True

        return func(*args, **kwargs)

    return wrapped_func
