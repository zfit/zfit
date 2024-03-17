#  Copyright (c) 2023 zfit
import tensorflow_probability as tfp
from zfit_interface.data import ZfitData


@tfp.experimental.auto_composite_tensor()
class UnbinnedData(tfp.experimental.AutoCompositeTensor, ZfitData):
    def __init__(self, data, space=None, weights=None):
        self._data = data
        self._space = space
        self._weights = weights

    @property
    def is_binned(self):
        return False

    @property
    def is_unbinned(self):
        return True

    @property
    def space(self):
        return self._space

    @property
    def data(self):
        return self._data

    @property
    def weights(self):
        return self._weights

    def values(self):
        return self.data

    def __getitem__(self, item):
        if not isinstance(item, str):
            return super().__getitem__(item)
        for index, axis in enumerate(self.axes):
            if axis.name == item:
                break
        else:
            raise KeyError(f"{item} not in {self.axes}")
        return self.data[..., index]

    @property
    def has_weights(self):
        return self._weights is not None


# tensorlike.register_tensor_conversion(UnbinnedData, name='UnbinnedData', overload_operators=True)
