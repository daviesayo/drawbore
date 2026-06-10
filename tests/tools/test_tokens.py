import pickle

import pytest
from drawbore.tools.tokens import CapabilityToken, TokenIssuer
from drawbore.tools.errors import TokenError


def test_issue_then_consume_once_ok():
    issuer = TokenIssuer()
    tok = issuer.issue("db.read", run_id="r1")
    issuer.consume(tok, "db.read", "r1")  # no raise


def test_second_consume_raises_single_use():
    issuer = TokenIssuer()
    tok = issuer.issue("db.read", run_id="r1")
    issuer.consume(tok, "db.read", "r1")
    with pytest.raises(TokenError):
        issuer.consume(tok, "db.read", "r1")


def test_scope_mismatch_raises():
    issuer = TokenIssuer()
    tok = issuer.issue("db.read", run_id="r1")
    with pytest.raises(TokenError):
        issuer.consume(tok, "db.write", "r1")  # wrong tool
    tok2 = issuer.issue("db.read", run_id="r1")
    with pytest.raises(TokenError):
        issuer.consume(tok2, "db.read", "r2")  # wrong run


def test_expiry_raises():
    clock = {"t": 100.0}
    issuer = TokenIssuer(ttl_seconds=5.0, clock=lambda: clock["t"])
    tok = issuer.issue("db.read", run_id="r1")
    clock["t"] = 106.0  # past expiry
    with pytest.raises(TokenError):
        issuer.consume(tok, "db.read", "r1")


def test_token_is_not_serializable():
    tok = TokenIssuer().issue("db.read", run_id="r1")
    with pytest.raises(TypeError):
        pickle.dumps(tok)


def test_repr_hides_secret():
    issuer = TokenIssuer()
    tok = issuer.issue("db.read", run_id="r1")
    assert "db.read" in repr(tok)
    # the internal secret must not leak through repr
    assert tok._secret not in repr(tok)


def test_consume_rejects_operation_scope_mismatch():
    import pytest
    from drawbore.tools import TokenIssuer
    from drawbore.tools.errors import TokenError

    issuer = TokenIssuer()
    token = issuer.issue("db.read", "run-1", operation="read")
    with pytest.raises(TokenError):
        issuer.consume(token, "db.read", "run-1", operation="write")


def test_consume_accepts_matching_operation():
    from drawbore.tools import TokenIssuer

    issuer = TokenIssuer()
    token = issuer.issue("db.read", "run-1", operation="read")
    issuer.consume(token, "db.read", "run-1", operation="read")  # no raise
