"""Error handling — halt-and-escalate default."""

from .errors import DrawboreError, SanitizationError, halt_reason_for, HALT_CODES

__all__ = ["DrawboreError", "SanitizationError", "halt_reason_for", "HALT_CODES"]
