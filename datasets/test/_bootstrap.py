"""Add policy root to sys.path for running tests as scripts."""

from __future__ import annotations

import os
import sys

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ensure_policy_root() -> str:
    if _POLICY_ROOT not in sys.path:
        sys.path.insert(0, _POLICY_ROOT)
    return _POLICY_ROOT
