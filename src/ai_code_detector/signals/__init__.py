"""Signals: small, versioned pieces of evidence for or against AI authorship."""

from ai_code_detector.signals.base import FileSignal, RepoContext, RepoSignal
from ai_code_detector.signals.python_file import FILE_SIGNALS
from ai_code_detector.signals.repo import REPO_SIGNALS

__all__ = ["FILE_SIGNALS", "REPO_SIGNALS", "FileSignal", "RepoContext", "RepoSignal"]
