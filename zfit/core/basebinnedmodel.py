#  Copyright (c) 2019 zfit
from typing import Union, Dict

from ..settings import ztypes
from ..util import ztyping
from .interfaces import ZfitBinnedModel, ZfitParameter
from .dimension import BaseDimensional
from ..util.cache import Cachable
from .baseobject import BaseNumeric


class BaseBinnedModel(BaseNumeric, Cachable, BaseDimensional, ZfitBinnedModel):

    def __init__(self, obs: ztyping.ObsTypeInput, params: Union[Dict[str, ZfitParameter], None] = None,
                 name: str = "BaseModel", dtype=ztypes.float,
                 **kwargs):
        """The base model to inherit from and overwrite `_unnormalized_pdf`.

        Args:
            dtype (DType): the dtype of the model
            name (str): the name of the model
            params (Dict(str, :py:class:`~zfit.Parameter`)): A dictionary with the internal name of the parameter and
                the parameters itself the model depends on
        """
        super().__init__(name=name, dtype=dtype, params=params, **kwargs)
        self._check_set_space(obs)


if __name__ == '__main__':
    import zfit

    obs = zfit.Space('asdf', (-3, 5))
    binned_model = BaseBinnedModel()
