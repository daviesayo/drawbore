"""Confinement receipts: deterministic, offline-verifiable run attestations.

A ``ConfinementReceipt`` is minted at the end of every run, binding the run's
declared authority footprint to its observed tool-call enforcement. ``verify``
re-checks the confinement invariants and the artifact's integrity offline — a
verifier never has to trust the runtime, only re-derive the verdict from the
evidence. The receipt is fingerprinted (SHA-256), not signed.
"""

from .receipt import (
    ConfinementReceipt,
    ConfinementVerdict,
    ObservedCall,
    mint_receipt,
    verify,
)

__all__ = [
    "ConfinementReceipt",
    "ConfinementVerdict",
    "ObservedCall",
    "mint_receipt",
    "verify",
]
