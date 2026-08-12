"""Test package.

Present so that `tests` is importable as a package regardless of how pytest is invoked.
Without it, bare `pytest` puts only `tests/` on sys.path -- not the repository root -- and
any cross-module import fails, while `python -m pytest` silently works because it prepends
the working directory. CI runs the bare form deliberately, so the difference cannot hide.
"""
