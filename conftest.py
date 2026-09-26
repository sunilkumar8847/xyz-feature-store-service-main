"""Root conftest — adds libs/xyz_security to sys.path for pytest."""
import os
import sys

_libs = os.path.join(os.path.dirname(__file__), "..", "..", "libs", "xyz_security")
if os.path.isdir(_libs):
    sys.path.insert(0, os.path.abspath(_libs))
