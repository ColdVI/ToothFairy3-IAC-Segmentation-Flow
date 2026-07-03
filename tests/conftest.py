import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "data"), os.path.join(ROOT, "flow")):
    if p not in sys.path:
        sys.path.insert(0, p)
