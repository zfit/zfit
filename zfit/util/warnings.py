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
                          f"https://github.com/zfit/zfit/issues)"
                          f" is very welcome!", stacklevel=2)
            warned = True

        return func(*args, **kwargs)

    return wrapped_func


class AdvancedFeatureWarning(UserWarning):
    pass


warned = set()


def warn_advanced_feature(message, identifier):
    from .. import settings

    if settings.advanced_warnings[identifier] and settings.advanced_warnings.all and identifier not in warned:
        warned.add(identifier)
        warnings.warn(
            f"Either you're using an advanced feature OR causing unwanted behavior. "
            f"To turn this warning off, use `zfit.settings.advanced_warnings.{identifier}` = False` "
            f" or 'all' (dangerous) with `zfit.settings.advanced_warnings.all = False\n"
            + message, category=AdvancedFeatureWarning, stacklevel=2)
