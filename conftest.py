"""Pytest configuration for resume-jd-matcher.

Ensures the repository root is on sys.path so modules and packages
can be imported consistently across environments.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
