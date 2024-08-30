#  Copyright (c) 2024 zfit
from __future__ import annotations

from typing import Literal, Optional, Union

import pydantic.v1 as pydantic
from pydantic.v1 import Field, root_validator

from .serializer import BaseRepr, Serializer


class BasePDFRepr(BaseRepr):
    _implementation = None
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["BasePDF"] = Field("BasePDF", alias="type")
    extended: Union[bool, None, Serializer.types.ParamTypeDiscriminated] = None
    # TODO: add norm?
    name: Optional[str] = None

    @root_validator(pre=True)
    def convert_params(cls, values):
        if cls.orm_mode(values):
            values = dict(values)
            values.update(**values.pop("params"))
            values["x"] = values.pop("space")
        return values

    def _to_orm(self, init):
        if "x" in init:  # in case it was already popped downstreams
            init["obs"] = init.pop("x")
        return super()._to_orm(init)
