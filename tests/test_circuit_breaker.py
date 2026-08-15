from unittest.mock import patch, MagicMock
from resilience.circuit_breaker import CircuitBreaker, FAILURE_THRESHOLD


def _fake_response(text):
    msg = MagicMock()
    msg.message.content = text
    resp = MagicMock()
    resp.choices = [msg]
    return resp


def test_successful_call_returns_content_and_stays_closed():
    breaker = CircuitBreaker()
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("ok")

    result = breaker.call(client, "some-model", [{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


def test_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker()
    breaker._fallback_client = MagicMock()
    breaker._fallback_client.chat.completions.create.return_value = _fake_response("fallback")

    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("upstream down")

    for _ in range(FAILURE_THRESHOLD):
        result = breaker.call(client, "some-model", [{"role": "user", "content": "hi"}])
        assert result == "fallback"

    assert breaker.state == "OPEN"
    assert breaker.failure_count == FAILURE_THRESHOLD


def test_open_circuit_routes_directly_to_fallback_without_calling_primary():
    breaker = CircuitBreaker()
    breaker.state = "OPEN"
    breaker.last_failure_time = __import__("time").time()
    breaker._fallback_client = MagicMock()
    breaker._fallback_client.chat.completions.create.return_value = _fake_response("fallback")

    client = MagicMock()
    result = breaker.call(client, "some-model", [{"role": "user", "content": "hi"}])

    assert result == "fallback"
    client.chat.completions.create.assert_not_called()
