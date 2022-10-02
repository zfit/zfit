#  Copyright (c) 2022 zfit
from __future__ import annotations
import contextlib
import copy
import functools
from enum import Enum
from typing import Type, Any, Optional, Union, Mapping, Iterable, Dict, List
from typing_extensions import Literal

from frozendict import frozendict
from pydantic.utils import GetterDict

import pydantic
from pydantic import constr, Field, validator

import pydantic

from zfit.util.exception import WorkInProgressError


class Serializer:
    constructor_repr = {}
    type_repr = {}
    _deserializing = False

    @classmethod
    def register(own_cls, repr):
        cls = repr._implementation

        if cls not in own_cls.constructor_repr:
            own_cls.constructor_repr[cls] = repr
        else:
            raise ValueError(f"Class {cls} already registered")

        hs3_type = repr.__fields__["hs3_type"].default
        cls.hs3_type = hs3_type
        cls.__annotations__["hs3_type"] = Literal[hs3_type]
        if hs3_type not in own_cls.type_repr:
            own_cls.type_repr[hs3_type] = repr
        else:
            raise ValueError(f"Type {hs3_type} already registered")

    @classmethod
    def to_hs3(cls, obj):
        serial_kwargs = {"exclude_none": True, "by_alias": True}
        if not isinstance(obj, (list, tuple)):
            pdfs = [obj]
        from zfit.core.interfaces import ZfitPDF

        if not all(isinstance(ob, ZfitPDF) for ob in pdfs):
            raise WorkInProgressError("Only PDFs can be serialized currently")
        from zfit.core.serialmixin import ZfitSerializable

        if not all(isinstance(pdf, ZfitSerializable) for pdf in pdfs):
            raise TypeError("All pdfs must be ZfitSerializable")
        import zfit

        out = {
            "metadata": {
                "HS3": {"version": "experimental"},
                "serializer": {"lib": "zfit", "version": zfit.__version__},
            },
            "pdfs": {},
            "variables": {},
        }
        pdf_number = range(len(pdfs))
        for pdf in pdfs:
            name = pdf.name
            if name in out["pdfs"]:
                name = f"{name}_{pdf_number}"
            out["pdfs"][name] = pdf.get_repr().from_orm(pdf).dict(**serial_kwargs)

            for param in pdf.get_params(floating=None):
                if param.name not in out["variables"]:
                    paramdict = param.get_repr().from_orm(param).dict(**serial_kwargs)
                    del paramdict["type"]
                    out["variables"][param.name] = paramdict

            for ob in pdf.obs:
                if ob not in out["variables"]:
                    space = pdf.space.with_obs(ob)
                    spacedict = space.get_repr().from_orm(space).dict(**serial_kwargs)
                    del spacedict["type"]
                    out["variables"][ob] = spacedict

        out = cls.post_serialize(out)

        return out

    @classmethod
    def from_hs3(cls, load):
        for param, paramdict in load["variables"].items():
            if 'value' in paramdict:
                paramdict['type'] = 'Parameter'
            else:
                paramdict['type'] = 'Space'

        load = cls.pre_deserialize(load)

        out = {'pdfs': {}, 'variables': {}}
        for name, pdf in load['pdfs'].items():
            repr = Serializer.type_repr[pdf['type']]
            out['pdfs'][name] = repr(**pdf).to_orm()
        for name, param in load['variables'].items():
            repr = Serializer.type_repr[param['type']]
            out['variables'][name] = repr(**param).to_orm()
        return out

    @classmethod
    @contextlib.contextmanager
    def deserializing(cls):
        cls._deserializing = True
        yield
        cls._deserializing = False

    @classmethod
    def post_serialize(cls, out):
        parameter = frozendict({'name': None, 'min': None, 'max': None})
        replace_forward = {parameter: lambda x: x['name']}
        out['pdfs'] = replace_matching(out['pdfs'], replace_forward)
        hs3_variables = frozendict({'name': None, 'type': None})
        return out

    @classmethod
    def pre_deserialize(cls, out):
        out = copy.deepcopy(out)
        replace_backward = {k: lambda x=k: out['variables'][x] for k in out['variables'].keys()}
        out['pdfs'] = replace_matching(out['pdfs'], replace_backward)
        return out


TYPENAME = "hs3_type"


def elements_match(mapping, replace):
    found = False
    for match, replacement in replace.items():
        if isinstance(mapping, Mapping) and isinstance(match, Mapping):
            for k, v in match.items():
                if k not in mapping:
                    break
                if v is None:
                    continue  # fine so far, a "free field"
                val = mapping.get(k)

                if val == v:
                    continue
                break
            else:
                found = True
                break
        if mapping == match:
            found = True
        if found:
            break
    else:
        return False, None
    return True, replacement(mapping)


def replace_matching(mapping, replace):
    # we need to test in the very beginning, it could be that the structure is a match
    is_match, new_map = elements_match(mapping, replace)
    if is_match:
        return new_map

    mapping = copy.copy(mapping)
    if isinstance(mapping, Mapping):
        for k, v in mapping.items():
            mapping[k] = replace_matching(v, replace)
    elif not isinstance(mapping, str) and isinstance(mapping, Iterable):
        replaced_list = [replace_matching(v, replace) for v in mapping]
        mapping = type(mapping)(replaced_list)
    return mapping


def convert_to_orm(init):
    if isinstance(init, Mapping):
        for k, v in init.items():

            if not isinstance(v, (Iterable, Mapping)):
                continue
            elif TYPENAME in v:
                type_ = v[TYPENAME]
                init[k] = Serializer.type_repr[type_](**v).to_orm()
            else:
                init[k] = convert_to_orm(v)

    elif isinstance(init, (list, tuple)):
        init = type(init)([convert_to_orm(v) for v in init])
    return init


def to_mro_init(func):
    @functools.wraps(func)
    def wrapper(self, init, **kwargs):
        init = convert_to_orm(init)
        return func(self, init, **kwargs)

    return wrapper


class MODES(Enum):
    orm = "orm"
    repr = "repr"


class BaseRepr(pydantic.BaseModel):
    _implementation = pydantic.PrivateAttr()
    _context = pydantic.PrivateAttr(None)
    _constructor = pydantic.PrivateAttr(None)
    dictionary: Optional[Dict] = Field(alias="dict")
    tags: Optional[List[str]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            cls._implementation is not None
        ):  # TODO: better way to catch if constructor is set vs BaseClass?
            Serializer.register(cls)

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @classmethod
    def orm_mode(cls, v):
        if cls._context is None:
            raise ValueError("No context set!")
        return cls._context == MODES.orm
        # return isinstance(v, (GetterDict, cls._implementation))

    @classmethod
    def from_orm(cls: pydantic.BaseModel, obj: Any) -> "BaseRepr":
        old_mode = cls._context
        cls._context = MODES.orm
        out = super().from_orm(obj)
        cls._context = old_mode
        return out

    def to_orm(self):
        print("Starting to_orm")
        old_mode = type(self)._context
        type(self)._context = MODES.repr
        # raise NotImplementedError("LOGIC?? TODO")
        if self._implementation is None:
            raise ValueError("No implementation registered!")
        init = self.dict(exclude_none=True)
        print(init)
        type_ = init.pop("hs3_type")
        # assert type_ == self.hs3_type
        out = self._to_orm(init)
        type(self)._context = old_mode
        print(f"Finished to_orm, mode = {type(self)._context}")
        return out

    @to_mro_init
    def _to_orm(self, init):
        constructor = self._constructor
        if constructor is None:
            constructor = self._implementation
        return constructor(**init)
