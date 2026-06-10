"""Error handling — halt-and-escalate default."""

from .errors import DrawboreError, ResumeDriftError, SanitizationError, halt_reason_for, HALT_CODES

__all__ = ["DrawboreError", "ResumeDriftError", "SanitizationError", "halt_reason_for", "HALT_CODES"]
