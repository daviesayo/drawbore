import pathlib
import re

import drawbore.tools.taint  # noqa: F401  (ensure importable)
from drawbore.tools.taint import TrustLabel, join, TaintLedger

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

# Patterns that must NOT appear in a stdlib-only leaf module.
_DRAWBORE = re.compile(r"^\s*(?:import|from)\s+drawbore\b", re.MULTILINE)
_THIRD_PARTY = re.compile(
    r"^\s*(?:import|from)\s+(?:pydantic|google|litellm|opentelemetry|mcp)\b",
    re.MULTILINE,
)


def test_taint_is_stdlib_only():
    """taint.py is a pure leaf: stdlib only, no drawbore imports, no third-party."""
    src = (_SRC / "tools" / "taint.py").read_text()
    drawbore_hits = _DRAWBORE.findall(src)
    third_party_hits = _THIRD_PARTY.findall(src)
    assert drawbore_hits == [], f"taint.py must not import drawbore: {drawbore_hits}"
    assert third_party_hits == [], f"taint.py must not import third-party libs: {third_party_hits}"


def test_join_laws():
    T, U = TrustLabel.TRUSTED, TrustLabel.UNTRUSTED
    assert join() is T                       # identity / empty
    assert join(T, T) is T
    assert join(T, U) is U and join(U, T) is U   # commutative, absorbing
    assert join(U, U) is U                   # idempotent on U
    assert join(T, T, T) is T


def test_ledger_seed_and_scope():
    led = TaintLedger()
    assert led.scope("r", 0) is TrustLabel.TRUSTED      # unseeded, unmanaged -> trusted (inert)
    led.seed("r", 0, TrustLabel.UNTRUSTED)
    assert led.scope("r", 0) is TrustLabel.UNTRUSTED


def test_observe_is_monotone():
    led = TaintLedger()
    led.seed("r", 1, TrustLabel.TRUSTED)
    led.observe("r", 1, TrustLabel.TRUSTED)
    assert led.scope("r", 1) is TrustLabel.TRUSTED
    led.observe("r", 1, TrustLabel.UNTRUSTED)
    assert led.scope("r", 1) is TrustLabel.UNTRUSTED
    led.observe("r", 1, TrustLabel.TRUSTED)             # cannot fall back
    assert led.scope("r", 1) is TrustLabel.UNTRUSTED


def test_managed_unseeded_scope_fails_closed():
    bare = TaintLedger()
    assert bare.scope("r", 9) is TrustLabel.TRUSTED      # standalone proxy stays inert
    managed = TaintLedger(initial_trust=TrustLabel.TRUSTED, managed=True)
    assert managed.scope("r", 9) is TrustLabel.UNTRUSTED  # unseeded-but-executed -> fail closed
    managed.seed("r", 0, TrustLabel.TRUSTED)
    assert managed.scope("r", 0) is TrustLabel.TRUSTED    # seed overrides the managed default for that key


def test_seed_cannot_lower_observed_taint():
    # Once a step is tainted UNTRUSTED, a subsequent seed(TRUSTED) must not reset it.
    led = TaintLedger()
    led.seed("r", 0, TrustLabel.TRUSTED)
    led.observe("r", 0, TrustLabel.UNTRUSTED)
    assert led.scope("r", 0) is TrustLabel.UNTRUSTED     # observe tainted it
    led.seed("r", 0, TrustLabel.TRUSTED)                 # should NOT lower the scope
    assert led.scope("r", 0) is TrustLabel.UNTRUSTED     # must stay UNTRUSTED

    # Fresh-key behavior must be unchanged: seed on an absent key sets the label as-is.
    led2 = TaintLedger()
    led2.seed("r", 1, TrustLabel.TRUSTED)
    assert led2.scope("r", 1) is TrustLabel.TRUSTED


def test_scopes_are_isolated_per_run_and_step():
    # Tainting one (run_id, step) must not affect any other key.
    led = TaintLedger()                                   # unmanaged: unseeded keys -> TRUSTED
    led.observe("r1", 0, TrustLabel.UNTRUSTED)
    assert led.scope("r1", 0) is TrustLabel.UNTRUSTED    # the tainted slot
    assert led.scope("r1", 1) is TrustLabel.TRUSTED      # different step, same run
    assert led.scope("r2", 0) is TrustLabel.TRUSTED      # different run, same step index
