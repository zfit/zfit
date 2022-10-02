#  Copyright (c) 2022 zfit

from typing_extensions import Literal
import pydantic
from pydantic import Field, root_validator, validator

from .spacerepr import SpaceRepr
from .paramrepr import ParameterRepr
from .serializer import BaseRepr

ParamsTypeDiscriminated = ParameterRepr


# Annotated[Union[ParameterRepr], Field(discriminator='hs3_type')]


class BasePDFRepr(BaseRepr):
    _implementation = None
    hs3_type: Literal["BasePDF"] = Field("BasePDF", alias="type")
