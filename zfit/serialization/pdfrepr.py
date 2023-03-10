#  Copyright (c) 2023 zfit
from __future__ import annotations

from typing import Union

try:
    from typing import Literal
except ImportError:  # TODO(3.8): remove
    from typing_extensions import Literal
import pydantic
from pydantic import Field, root_validator

from .serializer import BaseRepr, Serializer


class BasePDFRepr(BaseRepr):
    _implementation = None
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["BasePDF"] = Field("BasePDF", alias="type")
    extended: Union[bool, None, Serializer.types.ParamTypeDiscriminated] = None

    @root_validator(pre=True)
    def convert_params(cls, values):
        if cls.orm_mode(values):
            values = dict(values)
            values.update(**values.pop("params"))
            values["x"] = values.pop("space")
        return values
