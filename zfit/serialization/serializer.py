#  Copyright (c) 2023 zfit
from __future__ import annotations

import collections
import contextlib
import copy
import functools
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Union,
    Mapping,
    Iterable,
    Dict,
    List,
    TypeVar,
    Optional,
    Tuple,
    Literal,
)

import numpy as np
import pydantic
import tensorflow as tf
from frozendict import frozendict
from pydantic import Field

from zfit.util.container import convert_to_container

try:
    from typing import Annotated
except ImportError:  # TODO(3.9): remove
    from typing_extensions import Annotated

from zfit.core.interfaces import (
    ZfitParameter,
    ZfitPDF,
    ZfitData,
    ZfitBinnedData,
    ZfitConstraint,
    ZfitLoss,
)
from zfit.core.serialmixin import ZfitSerializable
from zfit.util.warnings import warn_experimental_feature


@dataclass
class Aliases:
    hs3_type: str = "type"


alias1 = Aliases(hs3_type="type")


class Types:
    def __init__(self):
        """Hold all types that are used in the serialization, automatically collects parameters and PDFs.

        This holding of types in this way is needed to delay the evaluation of the annotations, which is necessary since
        pydantic will remember (statically) the types and not evaluate them again. Therefore, we delay the evaluation of
        the types until the first time the serializer is used by artificially blocking the forward references
        evaluation.
        """
        self._constraint_repr = []
        self._data_repr = []
        self._pdf_repr = []
        self._param_repr = []
        self.block_forward_refs = True
        self.alias = alias1
        self.DUMMYTYPE = TypeVar("DUMMYTYPE")

    def one_or_many(self, repr):
        """Returns either a single or a list of the given repr correctly annotated.

        Args:
            repr: The repr to be annotated.

        Returns:
            The annotated repr.
        """
        if self.block_forward_refs:
            raise NameError(
                "Internal error, should always be caught! If you see this, most likely the annotation"
                " evaluation was not postponed. To fix this, add a `from __future__ import annotations`"
                " and make sure to use Python 3.8+"
            )
        if len(repr) == 0:
            return None
        elif len(repr) == 1:
            return repr[0]
        else:
            return Union[
                Annotated[Union[tuple(repr)], Field(discriminator="hs3_type")],
                self.DUMMYTYPE,
            ]

    @property
    def PDFTypeDiscriminated(self):
        return self.one_or_many(self._pdf_repr)

    @property
    def DataTypeDiscriminated(self):
        return self.one_or_many(self._data_repr)

    @property
    def ConstraintTypeDiscriminated(self):
        return self.one_or_many(self._constraint_repr)

    @property
    def ParamTypeDiscriminated(self):
        return self.one_or_many(self._param_repr)

    @property
    def ListParamTypeDiscriminated(self):
        return List[self.ParamTypeDiscriminated]

    @property
    def ParamInputTypeDiscriminated(self):
        return Union[self.ParamTypeDiscriminated, float, int]

    @property
    def ListParamInputTypeDiscriminated(self):
        return List[self.ParamInputTypeDiscriminated]

    def register_repr(self, repr: Union[ZfitPDF, ZfitParameter]) -> None:
        """Register a repr to be used in the serialization such as PDF or Parameter.

        This is needed to make sure that objects which use any of these types can be recursively
        serialized. They are autamtically added to the correct type.
        For example, a `SumPDF` can be a sum of arbitrary PDFs, so it needs to be able to serialize
        any PDF. This is done by registering the PDFs to the serializer and assigning the datafield
        of `PDFTypeDiscriminated` to the `pdfs` field of `SumPDF`.

        Discriminated refers to the fact that they're not arbitrary types, but only the types that
        are registered and that exactly those will be used.

        Args:
            repr: The repr to be registered.
        """
        cls = repr._implementation

        if issubclass(cls, ZfitPDF):
            self._pdf_repr.append(repr)
        elif issubclass(cls, ZfitParameter):
            self._param_repr.append(repr)
        elif issubclass(cls, (ZfitData, ZfitBinnedData)):
            self._data_repr.append(repr)
        elif issubclass(cls, ZfitConstraint):
            self._constraint_repr.append(repr)


class SerializationTypeError(TypeError):
    pass


class Serializer:
    """Main serializer, to be used as a class only."""

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "Serializer should be used as a class, no instances are allowed"
        )

    types = Types()
    is_initialized = False

    constructor_repr = {}
    type_repr = {}
    _deserializing = False

    @classmethod
    def register(own_cls, repr: ZfitSerializable) -> None:
        """Register a repr to be used in the HS3 serialization.

        Args:
            repr: The repr to be registered.
        """
        cls = repr._implementation
        if not issubclass(cls, ZfitSerializable):
            raise TypeError(
                f"{cls} is not a subclass of ZfitSerializable. Possible solution: inherit from "
                f"the SerializableMixin"
            )

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

        own_cls.types.register_repr(repr)

    @classmethod
    def initialize(cls) -> None:
        """Initialize the serializer by evaluating all the forward references.

        This is a necessary implementation trick to work properly with pydantics caching as we cannot modify an existing
        Type after it has been created. This is necessary to register new types as possible options (say a new PDF is
        registered and can be a possibility in a SumPDF) on the fly.
        """
        if not cls.is_initialized:
            cls.types.block_forward_refs = False
            for repr in cls.constructor_repr.values():
                repr.update_forward_refs(
                    **{"Union": Union, "List": List, "Literal": Literal}
                )
            cls.is_initialized = True

    @classmethod
    @warn_experimental_feature
    def to_hs3(
        cls,
        obj: Union[
            Union[List[ZfitPDF], Tuple[ZfitPDF], ZfitPDF],
            Union[List[ZfitLoss], Tuple[ZfitLoss], ZfitLoss],
        ],
    ) -> Mapping[str, Any]:
        """Serialize a PDF or a list of PDFs to a JSON string according to the HS3 standard.

        .. warning::
            This is an experimental feature and the API might change in the future. DO NOT RELY ON THE OUTPUT FOR
            ANYTHING ELSE THAN TESTING.

        THIS FUNCTION DOESN'T YET ADHERE TO HS3 (but just as a proxy).

        |@doc:hs3.explain| HS3 is the `HEP Statistics Serialization Standard <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_.
                   It is a JSON/YAML-based serialization that is a
                   coordinated effort of the HEP community to standardize the serialization of statistical models. The standard
                   is still in development and is not yet finalized. This function is experimental and may change in the future. |@docend:hs3.explain|

        Args:
            obj: The PDF or list of PDFs to be serialized.

        Returns:
            mapping: The serialized objects as a mapping.
                     |@doc:hs3.layout.explain| The keys in the HS3 format are:
                    - 'distributions': list of PDFs
                    - 'variables': list of variables, i.e. ``zfit.Space`` and ``zfit.Parameter`` (or more generally parameters)
                    - 'loss': list of losses
                    - 'data': list of data
                    - 'metadata': contains the version of the HS3 format and the
                      zfit version used to create the file |@docend:hs3.layout.explain|
        """

        cls.initialize()

        serial_kwargs = {"exclude_none": True, "by_alias": True}
        # check if already HS3 format
        if isinstance(obj, collections.abc.Mapping):
            if (
                "distributions" in obj
                and isinstance(obj["distributions"], collections.abc.Mapping)
                and "variables" in obj
                and "metadata" in obj
            ):
                raise ValueError(
                    "Object seems to be already in HS3 format. If it contains PDFs, use `obj['distributions'].values()` instead of `obj` to get a valid conversion"
                )
            else:
                raise ValueError(
                    "Mappings are currently not supported. Use a PDF or a list of PDFs instead."
                )
        else:
            obj = convert_to_container(obj)
        from zfit.core.interfaces import ZfitPDF

        all_pdfs = all(isinstance(ob, ZfitPDF) for ob in obj)
        all_losses = all(isinstance(ob, ZfitLoss) for ob in obj)
        if not all_pdfs and not all_losses:
            raise TypeError("Only PDFs or losses can be serialized.")
        from zfit.core.serialmixin import ZfitSerializable

        if not all(isinstance(pdf, ZfitSerializable) for pdf in obj):
            raise SerializationTypeError("All distributions must be ZfitSerializable")
        import zfit

        out = {
            "metadata": {
                "HS3": {"version": "experimental"},
                "serializer": {"lib": "zfit", "version": zfit.__version__},
            },
            "distributions": {},
            "variables": {},
            "loss": {},
            "data": {},
            "constraints": {},
        }
        loss_number = range(len(obj))

        all_objs = {"data": [], "distributions": [], "constraints": [], "loss": []}
        if all_pdfs:
            all_objs["distributions"] = obj
        else:
            for loss in obj:
                all_objs["distributions"].extend(loss.model)
                all_objs["constraints"].extend(loss.constraints)
                all_objs["data"].extend(loss.data)
                all_objs["loss"].append(loss)
        all_objs = {key: set(val) for key, val in all_objs.items()}
        all_objs_cleaned = {key: {} for key in all_objs.keys()}
        # give all of the objects unique names
        for key, val in all_objs.items():
            for ob in val:
                name = ob.name
                if name in all_objs_cleaned[key]:
                    name = f"{name}_{iter(loss_number)}"
                all_objs_cleaned[key][name] = ob

        for name, pdf in all_objs_cleaned["distributions"].items():
            assert name not in out["distributions"], "Name should have been uniqueified"
            pdf_repr = pdf.get_repr().from_orm(pdf)
            out["distributions"][name] = pdf_repr.dict(**serial_kwargs)

            for param in pdf.get_params(
                floating=None, extract_independent=None
            ):  # TODO: this is not ideal, we should take the serialized params?
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

        for name, loss in all_objs_cleaned["loss"].items():
            out["loss"][name] = loss.get_repr().from_orm(loss).dict(**serial_kwargs)

        for name, data in all_objs_cleaned["data"].items():
            out["data"][name] = data.get_repr().from_orm(data).dict(**serial_kwargs)

        for name, constraint in all_objs_cleaned["constraints"].items():
            out["constraints"][name] = (
                constraint.get_repr().from_orm(constraint).dict(**serial_kwargs)
            )

        out = cls.post_serialize(out)

        return out

    @classmethod
    @warn_experimental_feature
    def from_hs3(
        cls, load: Mapping[str, Mapping]
    ) -> Mapping[str, Union[ZfitPDF, ZfitParameter]]:
        """Load a PDF or a list of PDFs from a JSON string according to the HS3 standard.

        .. warning::
            This is an experimental feature and the API might change in the future. DO NOT RELY ON THE OUTPUT FOR
            ANYTHING ELSE THAN TESTING.

        THIS FUNCTION DOESN'T YET ADHERE TO HS3 (but just as a proxy).

        |@doc:hs3.explain| HS3 is the `HEP Statistics Serialization Standard <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_.
                   It is a JSON/YAML-based serialization that is a
                   coordinated effort of the HEP community to standardize the serialization of statistical models. The standard
                   is still in development and is not yet finalized. This function is experimental and may change in the future. |@docend:hs3.explain|

        Args:
            load: The serialized objects as a mapping. |@doc:hs3.layout.explain| The keys in the HS3 format are:
                    - 'distributions': list of PDFs
                    - 'variables': list of variables, i.e. ``zfit.Space`` and ``zfit.Parameter`` (or more generally parameters)
                    - 'loss': list of losses
                    - 'data': list of data
                    - 'metadata': contains the version of the HS3 format and the
                      zfit version used to create the file |@docend:hs3.layout.explain|

        Returns:
            mapping: The PDFs and variables as a mapping to the original keys.
        """
        cls.initialize()
        # sanity checks, TODO
        if "variables" not in load:
            pass
        if "distributions" not in load:
            pass
        if "metadata" not in load:
            pass
        for param, paramdict in load["variables"].items():
            if "value" in paramdict:
                if paramdict.get("floating", True) is False:
                    paramdict["type"] = "ConstantParameter"
                else:
                    paramdict["type"] = "Parameter"
            elif "value_fn" in paramdict:
                paramdict["type"] = "ComposedParameter"
            else:
                paramdict["type"] = "Space"

        load = cls.pre_deserialize(load)

        out = {
            "variables": {},  # order matters! variables first
            "data": {},
            "constraints": {},
            "distributions": {},
            "loss": {},
        }
        assert (
            list(out)[0] == "variables"
        ), "Order changed, has to deserealize variables first"
        for kind, kindout in out.items():
            for name, obj_dict in load[kind].items():
                repr = Serializer.type_repr[obj_dict["type"]]
                repr_inst = repr(**obj_dict)
                kindout[name] = repr_inst.to_orm()
        out["metadata"] = load["metadata"].copy()

        out = cls.post_deserialize(out)
        return out

    @classmethod
    @contextlib.contextmanager
    def deserializing(cls):
        cls._deserializing = True
        yield
        cls._deserializing = False

    @classmethod
    def post_serialize(cls, out):
        # This is not very stable as it allows only one pass and cannot be applied multiple times (i.e. the replacement back of the params:
        # name is replaced by the dict, that's fine for *once*, but fails if done twice (as the "name" field will be replaced by the dict)
        for what in ["distributions", "loss", "data", "constraints"]:
            # replace constant parameters with their name
            const_params = frozendict(
                {"name": None, "type": "ConstantParameter", "floating": False}
            )
            replace_forward_const_param = {const_params: lambda x: x["name"]}
            out[what] = replace_matching(out[what], replace_forward_const_param)

            # replace composed parameters with their name
            composed_params = frozendict(
                {"name": None, "type": "ComposedParameter", "value_fn": None}
            )
            replace_forward_composed_param = {composed_params: lambda x: x["name"]}
            out[what] = replace_matching(out[what], replace_forward_composed_param)

            # replace parameters and spaces with their name
            parameter = frozendict({"name": None, "min": None, "max": None})
            replace_forward_param = {parameter: lambda x: x["name"]}
            out[what] = replace_matching(out[what], replace_forward_param)
        for parname, param in out["variables"].items():
            if "value_fn" in param:
                out["variables"][parname]["params"] = replace_matching(
                    out["variables"][parname]["params"], replace_forward_const_param
                )
                out["variables"][parname]["params"] = replace_matching(
                    out["variables"][parname]["params"], replace_forward_param
                )
        return out

    @classmethod
    def pre_deserialize(cls, out):
        out = copy.deepcopy(out)
        replace_backward = {
            k: lambda x=k: out["variables"][x] for k in out["variables"].keys()
        }
        for parname, param in out["variables"].items():
            if param["type"] == "ComposedParameter":
                param["params"] = replace_matching(param["params"], replace_backward)
        for what in ["distributions", "loss", "data", "constraints"]:
            out[what] = replace_matching(out[what], replace_backward)
        return out

    @classmethod
    def post_deserialize(cls, out):
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
                    continue  # also fine
                break  # we're not fine, so let's stop here
            else:
                found = True
                break
        try:
            direct_hit = bool(mapping == match)
        except ValueError as error:
            if (
                "The truth value of an array with more than one element is ambiguous"
                in str(error)
            ):
                direct_hit = bool((mapping == match).all())
            else:
                raise
        except TypeError:
            continue
        found = direct_hit or found
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
    elif isinstance(mapping, Iterable) and not (
        isinstance(mapping, (str, (ZfitParameter, np.ndarray))) or tf.is_tensor(mapping)
    ):
        replaced_list = [replace_matching(v, replace) for v in mapping]
        mapping = type(mapping)(replaced_list)
    return mapping


def convert_to_orm(init):
    if isinstance(init, Mapping):
        for k, v in init.items():
            from zfit.core.interfaces import ZfitParameter, ZfitSpace

            from zfit.core.data import LightDataset

            if (
                not isinstance(v, (Iterable, Mapping))
                or isinstance(
                    v,
                    (
                        ZfitParameter,
                        ZfitSpace,
                        ZfitData,
                        ZfitBinnedData,
                        LightDataset,
                        np.ndarray,
                        str,
                        bytes,
                        bool,
                        int,
                        float,
                    ),
                )  # skip to not trigger the "in"
                or tf.is_tensor(v)
            ):
                continue
            elif TYPENAME in v:
                type_ = v[TYPENAME]
                init[k] = Serializer.type_repr[type_](**v).to_orm()
            else:
                init[k] = convert_to_orm(v)

        if TYPENAME in init:  # dicts can be the raw data we want.
            cls = Serializer.type_repr[init[TYPENAME]]
            return cls(**init).to_orm()

    elif isinstance(init, (list, tuple)):
        init = type(init)([convert_to_orm(v) for v in init])
    return init


def to_orm_init(func):
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
    tags: Optional[List[str]] = None

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
        smart_union = True

    @classmethod
    def orm_mode(cls, v):
        if cls._context is None:
            raise ValueError("No context set!")
        return cls._context == MODES.orm

    @classmethod
    def from_orm(cls: pydantic.BaseModel, obj: Any) -> BaseRepr:
        old_mode = cls._context
        try:
            cls._context = MODES.orm
            out = super().from_orm(obj)
        finally:
            cls._context = old_mode
        return out

    def to_orm(self):
        old_mode = type(self)._context
        try:
            type(self)._context = MODES.repr
            if self._implementation is None:
                raise ValueError("No implementation registered!")
            init = self.dict(exclude_none=True)
            type_ = init.pop("hs3_type")
            # assert type_ == self.hs3_type
            out = self._to_orm(init)
        finally:
            type(self)._context = old_mode
        return out

    @to_orm_init
    def _to_orm(self, init):
        constructor = self._constructor
        if constructor is None:
            constructor = self._implementation
        return constructor(**init)
