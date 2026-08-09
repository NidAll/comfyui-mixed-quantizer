"""Converter exception hierarchy."""

from __future__ import annotations

class QuantizerError(Exception):
    """Base class for all converter errors."""

class UsageError(QuantizerError):
    """Bad CLI usage."""

class InputError(QuantizerError):
    """Unreadable / unsafe / malformed input."""

class PickleInputError(InputError):
    """Pickle-based input without explicit --trust-pickle."""

class UnknownArchitectureError(QuantizerError):
    """Architecture could not be identified unambiguously."""

class UnsupportedArchitectureError(QuantizerError):
    """Architecture is known but has no safe conversion policy."""

class PolicyError(QuantizerError):
    """Architecture policy violation."""

class QualityGateError(QuantizerError):
    """Mixed planner: the global quality gate could not be met (hard failure)."""

class CompressionGateError(QuantizerError):
    """Mixed planner: the compression gate could not be met (hard failure)."""

class RuntimeCompatibilityError(QuantizerError):
    """Mixed mode: a selected format has no verified runtime path."""

class CalibrationError(QuantizerError):
    """Calibration data problem."""

class ValidationError(QuantizerError):
    """Output failed standalone validation."""

class OutputError(QuantizerError):
    """Output path / serialization problem."""

class SelfTestFailure(QuantizerError):
    """Embedded self-test failure."""
