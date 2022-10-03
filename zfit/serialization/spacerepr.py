#  Copyright (c) 2022 zfit
from __future__ import annotations

from typing import List, Tuple, Union, Optional, Mapping
from typing_extensions import Literal

from pydantic import Field, validator, root_validator

from .serializer import BaseRepr
from ..core.space import Space
from ..util.exception import WorkInProgressError

NumericTyped = Union[float, int]

NameObsTyped = Optional[Union[Tuple[str], str]]


class SpaceRepr(BaseRepr):
    _implementation = Space
    hs3_type: Literal["Space"] = Field("Space", alias="type")
    name: str
    lower: NumericTyped = Field(alias="min")
    upper: NumericTyped = Field(alias="max")
    binning: Optional[float] = None

    @root_validator(pre=True)
    def _validate_pre(cls, values):
        if cls.orm_mode(values):
            values = dict(values)
            values["name"] = values.pop("obs")[0]

        return values

    @validator("lower", pre=True)
    def _validate_lower(cls, v):
        if cls.orm_mode(v):
            v = v[0, 0]
        return v

    @validator("upper", pre=True)
    def _validate_upper(cls, v):
        if cls.orm_mode(v):
            v = v[0, 0]
        return v

    @validator("binning", pre=True, allow_reuse=True)
    def validate_binning(cls, v):
        if v is not None:
            raise WorkInProgressError("Binning is not implemented yet")
        return v

    def _to_orm(self, init) -> SpaceRepr._implementation:
        init["limits"] = init.pop("lower"), init.pop("upper")
        init["obs"] = init.pop("name")
        init = super()._to_orm(init)
        return init
