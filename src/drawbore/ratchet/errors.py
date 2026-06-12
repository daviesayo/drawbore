"""Ratchet error types.

``RatchetError`` covers misuse of the corpus/gate APIs (blank sponsor, malformed
bundle); ``CorpusIntegrityError`` is the fail-closed signal that the hash chain
or the pinned replay inputs do not verify.
"""

from __future__ import annotations


class RatchetError(Exception):
    """Invalid use of a ratchet API (fail closed, legible message)."""


class CorpusIntegrityError(RatchetError):
    """The corpus hash chain or a pinned replay input failed verification."""
