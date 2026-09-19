#!/usr/bin/env python3
"""Catch the "assigned in a nested function, read from the enclosing one" bug.

Python decides a name is local to a function if it is assigned *anywhere* in
that function body.  Rebinding a name that the enclosing function owns - very
easy to do while backtracking, e.g. `delta *= 0.5` inside a nested closure -
therefore turns every earlier read of that name into an UnboundLocalError,
even on paths where the assignment never runs.

Running a real training step used to be the only way to notice.  This check is
static and takes milliseconds, so it can gate every commit.

Usage:
  python scripts/check_scoping.py src main.py
Exit status 1 if any shadowing assignment is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def assigned_names(node: ast.AST) -> set[str]:
    """Names bound directly by this function body (not inside nested defs)."""
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, n):  # noqa: N802
            names.add(n.name)  # the def itself, not its body

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, n):  # noqa: N802
            return

        def visit_ClassDef(self, n):  # noqa: N802
            names.add(n.name)

        def visit_Name(self, n):  # noqa: N802
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                names.add(n.id)

        def visit_Global(self, n):  # noqa: N802
            names.update(n.names)

        def visit_Nonlocal(self, n):  # noqa: N802
            names.update(n.names)

    if isinstance(node, ast.Lambda):
        return set()
    body = getattr(node, "body", [])
    if not isinstance(body, list):
        return set()
    for statement in body:
        Visitor().visit(statement)
    args = node.args
    for arg in (
        *args.posonlyargs, *args.args, *args.kwonlyargs,
        args.vararg, args.kwarg,
    ):
        if arg is not None:
            names.discard(arg.arg)
    return names


def enclosing_bindings(func: ast.AST) -> set[str]:
    """Everything the enclosing function could have bound."""
    names = set()
    args = func.args
    for arg in (
        *args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg,
    ):
        if arg is not None:
            names.add(arg.arg)
    names |= assigned_names(func)
    return names


def walk(functions, path: Path, out: list[str]) -> None:
    for func in functions:
        owned = enclosing_bindings(func)
        for child in ast.walk(func):
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                continue
            if child is func:
                continue
            # only direct/nested defs inside this function
            shadowed = assigned_names(child) & owned
            for name in sorted(shadowed):
                out.append(
                    f"{path}:{child.lineno}: '{name}' is bound in the enclosing "
                    f"'{getattr(func, 'name', '<lambda>')}' and reassigned in a "
                    f"nested function -> earlier reads become UnboundLocalError"
                )
        walk(
            [n for n in func.body if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef))],
            path, out,
        )


def check(path: Path, out: list[str]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    walk([n for n in tree.body if isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef))], path, out)


def main() -> int:
    targets: list[Path] = []
    for arg in sys.argv[1:] or ["src", "main.py"]:
        p = Path(arg)
        targets.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    findings: list[str] = []
    for path in targets:
        if "__pycache__" in path.parts:
            continue
        check(path, findings)
    if findings:
        print("\n".join(findings))
        print(f"\n{len(findings)} shadowing assignment(s) found")
        return 1
    print(f"scoping OK ({len(targets)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
