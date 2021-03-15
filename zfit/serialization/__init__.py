#  Copyright (c) 2020 zfit
from zfit.serialization.interfaces import ZfitSerializable, ZfitRepr, ZfitArranger

__all__ = [ZfitSerializable, ZfitRepr, ZfitArranger]


"""Serialize and deserialize objects.

The zfit serialization process is split into the following steps:

.. graphviz::

   digraph serialization_order {
      "ZfitSerializable" -> "ZfitRepr";
      "ZfitRepr" -> "ZfitArranger";
      "ZfitArranger" -> "ZfitSerializer";
   }
"""
