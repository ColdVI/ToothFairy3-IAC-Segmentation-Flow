"""Shared sys.path setup so tests run both under pytest and as `python tests/x.py`."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "data"), os.path.join(ROOT, "flow")):
    if p not in sys.path:
        sys.path.insert(0, p)


def run_module(mod):
    """Run every test_* function in a module as a mini test runner."""
    fns = [v for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"[{mod.__name__}] {len(fns)} passed")
