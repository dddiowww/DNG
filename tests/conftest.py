import sys
from pathlib import Path

# Ensure the project root is on sys.path so flat-module imports work.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
