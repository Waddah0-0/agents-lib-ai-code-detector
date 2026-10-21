"""AI Code Detector: estimates how likely a codebase was written with AI assistance."""

from ai_code_detector.agent import AICodeDetector
from ai_code_detector.models import DetectorInput, DetectorOutput, FileScore, SignalHit

__version__ = "0.2.1"

__all__ = [
    "AICodeDetector",
    "DetectorInput",
    "DetectorOutput",
    "FileScore",
    "SignalHit",
    "__version__",
]
