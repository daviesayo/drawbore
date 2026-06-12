"""Improvement under a safety ratchet.

An untrusted optimizer may propose a new pipeline manifest as JSON; nothing it
can express reaches the verifier code in this package. ``admit()`` is the only
sanctioned path from a candidate manifest to a blessed ``Pipeline``.
"""

from .errors import CorpusIntegrityError, RatchetError
from .sink import InMemoryRatchetSink, RatchetSink
from .verdict import RatchetVerdict

__all__ = [
    "CorpusIntegrityError",
    "RatchetError",
    "InMemoryRatchetSink",
    "RatchetSink",
    "RatchetVerdict",
]
