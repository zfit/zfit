#  Copyright (c) 2021 zfit
import tensorflow_probability as tfp
from zfit_interface.data import ZfitData

import zfit.core.tensorlike as tensorlike
import zfit.serialization as serialization


@tfp.experimental.auto_composite_tensor()
@serialization.register(uid="UnbinnedData")
class UnbinnedData(tfp.experimental.AutoCompositeTensor,
                   serialization.serializer.Serializable, ZfitData):

    def __init__(self, data, axes=None, weights=None):
        self._data = data
        self._axes = axes
        self._weights = weights

    @property
    def is_binned(self):
        return False

    @property
    def is_unbinned(self):
        return True

    @property
    def axes(self):
        return self._axes

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

    def _obj_to_repr(self):
        return {'data': self.data, 'axes': self.axes}

# tensorlike.register_tensor_conversion(UnbinnedData, name='UnbinnedData', overload_operators=True)
