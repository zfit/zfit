import abc

import pep487


class ZfitObject(pep487.ABC):

    @abc.abstractmethod
    def get_dependents(self, only_floating=False):
        raise NotImplementedError
