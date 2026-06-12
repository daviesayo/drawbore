"""Improvement under a safety ratchet.

An untrusted optimizer may propose a new pipeline manifest as JSON; nothing it
can express reaches the verifier code in this package. ``admit()`` is the only
sanctioned path from a candidate manifest to a blessed ``Pipeline``.
"""

from .corpus import (
    GENESIS,
    InMemoryRegressionCorpus,
    RegressionCase,
    RegressionCorpus,
    SafetyProperty,
    derive_cases,
    mocks_fingerprint,
    seal_case,
)
from .capability import authority_delta
from .errors import CorpusIntegrityError, RatchetError
from .gate import admit
from .sink import InMemoryRatchetSink, RatchetSink
from .verdict import RatchetVerdict

__all__ = [
    "CorpusIntegrityError",
    "admit",
    "authority_delta",
    "GENESIS",
    "InMemoryRatchetSink",
    "InMemoryRegressionCorpus",
    "RatchetError",
    "RatchetSink",
    "RatchetVerdict",
    "RegressionCase",
    "RegressionCorpus",
    "SafetyProperty",
    "derive_cases",
    "mocks_fingerprint",
    "seal_case",
]
