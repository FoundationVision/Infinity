"""Configure sys.path for Infinity project tests.

The cai-framework package installs a .pth file that adds cai/tools/ to
sys.path. To avoid import conflicts, we insert the Infinity project root
at position 0 and invalidate any cached 'tools' module.
"""

import importlib
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure project root is first in sys.path
if sys.path[0] != _project_root:
    while _project_root in sys.path:
        sys.path.remove(_project_root)
    sys.path.insert(0, _project_root)

# Evict stale 'tools' module from cai-framework so our tools/ wins
for key in list(sys.modules):
    if key == "tools" or key.startswith("tools."):
        del sys.modules[key]

# Force-create a proper package reference for our tools/ directory
import types
tools_pkg = types.ModuleType("tools")
tools_pkg.__path__ = [os.path.join(_project_root, "tools")]
tools_pkg.__file__ = os.path.join(_project_root, "tools", "__init__.py")
tools_pkg.__package__ = "tools"
sys.modules["tools"] = tools_pkg
