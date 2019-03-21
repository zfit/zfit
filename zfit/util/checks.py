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
