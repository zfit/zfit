#  Copyright (c) 2019 zfit

class NotSpecified:
    _singleton_instance = None

    def __new__(cls, *args, **kwargs):
        instance = cls._singleton_instance
        if instance is None:
            instance = super().__new__(cls)
            cls._singleton_instance = instance

        return instance

    def __bool__(self):
        return False

    def __str__(self):
        return "A not specified state"


NOT_SPECIFIED = NotSpecified()


class ZfitNotImplemented:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Cannot create an instance of it, meant to be used as a single object reference.")
