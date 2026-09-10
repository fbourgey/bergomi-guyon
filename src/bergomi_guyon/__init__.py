"""Formal Bergomi--Guyon coefficients and exact verification."""

from .recursion import GenerationResult, generate_coefficients
from .verify import verify

__all__ = ["GenerationResult", "generate_coefficients", "verify"]
