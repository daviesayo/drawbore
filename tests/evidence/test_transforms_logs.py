from drawbore.evidence import EvidencePolicy
from drawbore.evidence.transforms import get_transform, logs


def _policy(**kw):
    return EvidencePolicy(enabled=True, **kw)


def _logtext(n_info=400):
    lines = [f"INFO step {i} ok" for i in range(n_info)]
    lines.insert(200, "ERROR connection refused")
    lines.insert(201, "Traceback (most recent call last):")
    lines.insert(202, "  File 'x.py', line 1, in <module>")
    lines.append("WARN slow response")
    return {"log": "\n".join(lines)}


def test_registry_resolves_logs_by_name():
    assert get_transform("logs") is logs


def test_compress_is_deterministic():
    value = _logtext()
    out1, w1 = logs.compress(value, _policy())
    out2, w2 = logs.compress(value, _policy())
    assert out1 == out2 and w1 == w2


def test_errors_warnings_and_traceback_are_preserved():
    out, _ = logs.compress(_logtext(), _policy())
    body = out["log"]
    assert "ERROR connection refused" in body
    assert "Traceback (most recent call last):" in body
    assert "WARN slow response" in body


def test_compression_shrinks_and_summarizes_drops():
    value = _logtext(2000)
    out, warnings = logs.compress(value, _policy())
    assert len(out["log"]) < len(value["log"])
    assert any("dropped" in w.lower() for w in warnings)


def test_adversarial_benign_error_word_is_not_a_false_error_line():
    value = {"log": "\n".join(
        ["INFO no error occurred during startup"] * 500 + ["INFO done"]
    )}
    out, _ = logs.compress(value, _policy())
    assert out["log"].count("INFO no error occurred during startup") < 500


def test_adversarial_every_line_is_an_error_is_capped():
    value = {"log": "\n".join([f"ERROR failure {i}" for i in range(2000)])}
    out, warnings = logs.compress(value, _policy(max_output_tokens=2000))
    assert out["log"].count("ERROR failure") < 2000
    assert any("cap" in w.lower() or "drop" in w.lower() for w in warnings)


def test_adversarial_indented_benign_log_still_compresses():
    # Pretty-printed / indented benign lines must NOT be treated as severe just
    # for their indentation, or compression is defeated on common log formats.
    value = {"log": "\n".join(
        ["    \"event\": \"heartbeat ok\"," for _ in range(1000)] + ["INFO done"]
    )}
    out, _ = logs.compress(value, _policy())
    assert out["log"].count("heartbeat ok") < 1000  # the indented middle is dropped


def test_newline_free_string_is_left_untouched():
    # A single newline-free blob has no line structure to compress deterministically;
    # the logs transform leaves it alone (out of scope, documented).
    value = {"blob": "x" * 5000}
    out, _ = logs.compress(value, _policy())
    assert out == value


def test_non_log_payload_is_unchanged():
    value = {"summary": "short", "n": 1}
    out, _ = logs.compress(value, _policy())
    assert out == value
