#  Copyright (c) 2022 zfit

from __future__ import annotations

from typing import Literal, Optional

import pydantic
from pydantic import Field, validator

from .serializer import BaseRepr, Serializer
from ..core.parameter import Parameter


def param_constructor(name, **kwargs):
    if name in Parameter._existing_params:
        return Parameter._existing_params[name]
    else:
        return Parameter(name=name, **kwargs)


class ParameterRepr(BaseRepr):
    _implementation = Parameter
    _constructor = pydantic.PrivateAttr(param_constructor)
    hs3_type: Literal["Parameter"] = Field("Parameter", alias="type")
    name: str
    value: float
    lower: float | None = Field(None, alias="min")
    upper: float | None = Field(None, alias="max")
    step_size: float | None = None
    floating: bool | None = None

    @validator("value", pre=True)
    def _validate_value(cls, v):
        if cls.orm_mode(v):
            v = v()
        return v
