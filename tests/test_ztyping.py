#  Copyright (c) 2026 zfit
from __future__ import annotations

import ast
import collections
import inspect

from zfit.util import ztyping

# Assigned twice, the first definition is used for ``LimitsTypeInputV1``, see issue 721.
KNOWN_REDEFINITIONS = {"NumericalType"}


def test_type_aliases_defined_once():
    # a second assignment silently replaces the first one, so a type that was meant to be
    # used is not
    tree = ast.parse(inspect.getsource(ztyping))
    lines = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lines[target.id].append(node.lineno)
    redefined = {name: nums for name, nums in lines.items() if len(nums) > 1 and name not in KNOWN_REDEFINITIONS}
    assert not redefined, f"type aliases assigned more than once (name: lines): {redefined}"
