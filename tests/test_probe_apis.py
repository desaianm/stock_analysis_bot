import pytest

from scripts.probe_apis import (
    ProbeResult,
    _from_response,
    classify_response,
    report_exit_code,
    run_probes,
    safe_report,
    validate_payload,
)


class SuccessfulResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.text = ""
        self._payload = payload

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        payloads = {
            "https://api.openai.com/v1/responses": {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "error": None,
                "output": [{
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                }],
            },
            "https://api.financialdatasets.ai/insider-trades/": {"insider_trades": []},
            "https://api.tavily.com/search": {"results": []},
            "https://api.exa.ai/search": {"results": []},
            "https://discord.com/api/v10/users/@me": {"id": "1", "username": "bot"},
            "https://www.reddit.com/api/v1/access_token": {
                "access_token": "oauth-token", "token_type": "bearer", "expires_in": 3600,
            },
            "https://oauth.reddit.com/r/stocks/new": {"kind": "Listing", "data": {"children": []}},
            "https://query1.finance.yahoo.com/v8/finance/chart/AAPL": {
                "chart": {"result": [{"meta": {"symbol": "AAPL"}}], "error": None},
            },
        }
        return SuccessfulResponse(payloads[url])

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)


def test_response_classification_is_deterministic():
    assert classify_response(200, "ok") == "working"
    assert classify_response(401, "bad token") == "auth_or_permission"
    assert classify_response(403, "forbidden") == "auth_or_permission"
    assert classify_response(429, "rate limit") == "rate_limit"
    assert classify_response(429, "insufficient_quota") == "quota_or_billing"
    assert classify_response(402, "billing") == "quota_or_billing"
    assert classify_response(400, "insufficient_quota") == "quota_or_billing"
    assert classify_response(500, "oops") == "network_or_endpoint_failure"


def test_safe_report_contains_secret_names_but_never_values():
    secret = "super-secret-value"
    results = [ProbeResult("OpenAI", True, True, "working", 200, "OPENAI_API_KEY")]

    report = safe_report(results, configured_values={"OPENAI_API_KEY": secret})

    assert report["services"][0]["secret_name"] == "OPENAI_API_KEY"
    assert secret not in str(report)


def test_exit_nonzero_for_failed_or_missing_required_service():
    optional_failure = ProbeResult("Reddit", False, True, "auth_or_permission", 401,
                                   "REDDIT_CLIENT_ID,REDDIT_CLIENT_SECRET")
    missing_required = ProbeResult("OpenAI", True, False, "not_configured", None, "OPENAI_API_KEY")
    failed_required = ProbeResult("Yahoo Finance", True, True, "network_or_endpoint_failure", None, None)

    assert report_exit_code([optional_failure, missing_required]) == 1
    assert report_exit_code([failed_required]) == 1


@pytest.mark.parametrize(
    ("service", "payload"),
    [
        ("OpenAI", {"data": []}),
        ("Financial Datasets", {"status": "ok"}),
        ("Tavily", {"answer": "AAPL"}),
        ("EXA", {"requestId": "x"}),
        ("Discord", {"message": "ok"}),
        ("Reddit token", {"access_token": "x"}),
        ("Reddit listing", {"data": {}}),
        ("Yahoo Finance", {"chart": {"result": None, "error": None}}),
    ],
)
def test_arbitrary_2xx_payload_is_not_healthy(service, payload):
    assert validate_payload(service, payload) is False


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "resp_empty", "status": "completed", "error": None, "output": []},
        {
            "id": "resp_incomplete",
            "status": "incomplete",
            "error": None,
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            }],
        },
        {
            "id": "resp_refusal",
            "status": "completed",
            "error": None,
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "Cannot comply"}],
            }],
        },
        {
            "id": "resp_error",
            "status": "completed",
            "error": {"code": "server_error", "message": "Failed"},
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            }],
        },
    ],
)
def test_openai_malformed_2xx_payload_is_invalid_payload(payload):
    result = _from_response(
        "OpenAI", True, True, "OPENAI_API_KEY", SuccessfulResponse(payload)
    )

    assert result.status == "invalid_payload"


def test_openai_probe_uses_tiny_completion_request():
    session = RecordingSession()

    run_probes({"OPENAI_API_KEY": "secret"}, session=session)

    call = next(call for call in session.calls if call[1].startswith("https://api.openai.com"))
    assert call[0] == "POST"
    assert call[1] == "https://api.openai.com/v1/responses"
    assert call[2]["json"]["max_output_tokens"] == 16


def test_probe_requests_never_put_secrets_in_urls_or_bodies():
    secret_values = {name: f"secret-{name}" for name in (
        "OPENAI_API_KEY", "FINANCIAL_DATASETS_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY",
        "DISCORD_TOKEN", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "POLYGON_API_KEY",
    )}
    session = RecordingSession()

    results = run_probes(secret_values, session=session)

    serialized_unsafe_fields = str([
        (url, kwargs.get("params"), kwargs.get("json"), kwargs.get("data"))
        for _, url, kwargs in session.calls
    ])
    assert all(value not in serialized_unsafe_fields for value in secret_values.values())
    assert {result.service for result in results} == {
        "OpenAI", "Financial Datasets", "Tavily", "EXA", "Discord", "Reddit",
        "Yahoo Finance", "Polygon (optional/unused)",
    }
