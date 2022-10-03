#  Copyright (c) 2022 zfit
from __future__ import annotations

from typing import Optional, Union, Any, List

from typing_extensions import Literal, Annotated
import pydantic
from pydantic import Field, root_validator, validator

from .spacerepr import SpaceRepr
from ..core.parameter import ParameterRepr
from .serializer import BaseRepr, Serializer
from ..core.interfaces import ZfitPDF


class BasePDFRepr(BaseRepr):
    _implementation = None
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["BasePDF"] = Field("BasePDF", alias="type")
    extended: Union[bool, None, Serializer.types.ParamTypeDiscriminated] = None
