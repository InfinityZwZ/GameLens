import httpx
import pytest

from gamelens.core import llm as llm_client
from gamelens.core.llm import OpenAICompatibleClient, LLMResponseError
from gamelens.storage.telemetry import TelemetryStore, summarize


def setup_client(monkeypatch, tmp_path, handler):
    monkeypatch.setenv("LLM_INPUT_PRICE_PER_MILLION", "2")
    monkeypatch.setenv("LLM_OUTPUT_PRICE_PER_MILLION", "4")
    monkeypatch.setenv("LLM_PRICE_CURRENCY", "CNY")
    factory = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_client.httpx, "Client", lambda **kwargs: factory(transport=transport, **kwargs))
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    store = TelemetryStore(tmp_path / "telemetry.sqlite")
    client = OpenAICompatibleClient("test-model", api_key="not-a-real-key", max_retries=1,
                                    telemetry_sink=lambda event: store.record("run-1", event))
    return client, store


def response(content='{"ok": true}', usage=True):
    result = {"choices": [{"message": {"content": content}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 100, "completion_tokens": 50}
    return result


def test_retry_usage_cost_and_durable_records(monkeypatch, tmp_path):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json={}) if len(attempts) == 1 else httpx.Response(200, json=response())

    client, store = setup_client(monkeypatch, tmp_path, handler)
    assert client.generate_json([{ "role": "user", "content": "private content" }]) == {"ok": True}
    events = TelemetryStore(store.path).events("run-1")
    metrics = summarize(events)
    assert metrics["http_requests"] == 2
    assert metrics["model_calls"] == 1
    assert metrics["retries"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["prompt_tokens"] == 100
    assert metrics["known_cost_subtotal"] == pytest.approx(0.0004)
    assert metrics["estimated_cost"] is None  # Failed response has no usage; no invented zero.
    assert "private content" not in str(events)
    assert "not-a-real-key" not in str(events)
    assert len(events) == 3  # Initial request records are updated rather than duplicated.


def test_invalid_json_is_failed_call_but_usage_is_still_counted(monkeypatch, tmp_path):
    client, store = setup_client(monkeypatch, tmp_path, lambda _: httpx.Response(200, json=response("invalid")))
    with pytest.raises(LLMResponseError):
        client.generate_json([])
    metrics = summarize(store.events("run-1"))
    assert metrics["failed_calls"] == 1
    assert metrics["failed_requests"] == 0
    assert metrics["estimated_cost"] == pytest.approx(0.0004)


def test_missing_usage_remains_unknown(monkeypatch, tmp_path):
    client, store = setup_client(monkeypatch, tmp_path, lambda _: httpx.Response(200, json=response(usage=False)))
    client.generate_json([])
    metrics = summarize(store.events("run-1"))
    assert metrics["prompt_tokens"] is None
    assert metrics["estimated_cost"] is None
    assert metrics["usage_complete"] is False


def test_fallback_and_transport_failure(monkeypatch, tmp_path):
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(400, json={})
        raise httpx.ReadTimeout("timeout", request=request)

    client, store = setup_client(monkeypatch, tmp_path, handler)
    with pytest.raises(httpx.ReadTimeout):
        client.generate_json([])
    metrics = summarize(store.events("run-1"))
    assert metrics["http_requests"] == 3
    assert metrics["fallbacks"] == 1
    assert metrics["retries"] == 1
    assert metrics["failed_calls"] == 1
    assert metrics["pending_requests"] == 0


def test_optional_output_budget_and_context_size_are_content_free(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "512")
    captured = []
    def handler(request):
        import json
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    client, store = setup_client(monkeypatch, tmp_path, handler)
    token = llm_client.OUTPUT_BUDGET.set(128)
    try:
        client.generate_json([{"role": "user", "content": "private"}])
    finally:
        llm_client.OUTPUT_BUDGET.reset(token)
    assert captured[0]["max_tokens"] == 128
    metrics = summarize(store.events("run-1"))
    assert metrics["prompt_chars_total"] == 7
    assert "private" not in str(store.events("run-1"))


def test_finish_reason_length_is_not_treated_as_complete(monkeypatch, tmp_path):
    body = response()
    body["choices"][0]["finish_reason"] = "length"
    client, store = setup_client(monkeypatch, tmp_path, lambda _: httpx.Response(200, json=body))
    with pytest.raises(LLMResponseError, match="截断"):
        client.generate_json([])
    assert summarize(store.events("run-1"))["failed_calls"] == 1
