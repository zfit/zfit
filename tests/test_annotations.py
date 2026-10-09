#  Copyright (c) 2026 zfit
from __future__ import annotations

import ast
import pathlib

import zfit


def _annotations(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def test_no_builtin_callable_in_annotations():
    # `callable` is a function, not a type: `callable | None` raises a TypeError as soon as the
    # annotation is evaluated (e.g. by `typing.get_type_hints`), use `collections.abc.Callable`
    src_dir = pathlib.Path(zfit.__file__).parent
    wrong = []
    for path in src_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for annotation in _annotations(tree):
            for node in ast.walk(annotation):
                if isinstance(node, ast.Name) and node.id == "callable":
                    wrong.append(f"{path.relative_to(src_dir)}:{node.lineno}")
    assert not wrong, f"builtin `callable` used as a type annotation: {wrong}"
