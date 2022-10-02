#  Copyright (c) 2022 zfit
from typing import Optional, Union, Any

from typing_extensions import Literal
import pydantic
from pydantic import Field, root_validator, validator

from .spacerepr import SpaceRepr
from .paramrepr import ParameterRepr
from .serializer import BaseRepr
from ..core.interfaces import ZfitPDF

ParamsTypeDiscriminated = ParameterRepr


# Annotated[Union[ParameterRepr], Field(discriminator='hs3_type')]


class BasePDFRepr(BaseRepr):
    _implementation = None
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["BasePDF"] = Field("BasePDF", alias="type")
    extended: Optional[Union[bool, ParameterRepr]]
