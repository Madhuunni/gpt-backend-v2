import json

import httpx
import pytest

from app.graph import build_graph, evaluate
from app.extensions import EXTRA_TASKS
from http_task.task import HTTPTaskError, call_http_api


class Classifier:
    def __init__(self, context):
        self.context = context
    def predict(self, query):
        return self.context


class Extractor:
    def predict(self, query):
        return {"employee_id": "00123"}


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.setenv("DOWNSTREAM_API_URL", "https://example.test/details")
    monkeypatch.setenv("DOWNSTREAM_API_TOKEN", "test-token")


def test_http_response_in_graph(configure, monkeypatch):
    def post(url, **kwargs):
        assert url == "https://example.test/details"
        assert kwargs["json"] == {"query": "original query", "context": "case_details", "identifiers": {"employee_id": "00123"}}
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"
        assert kwargs["timeout"] == 15
        assert kwargs["follow_redirects"] is False
        return httpx.Response(200, json=[{"amount": 100}])
    monkeypatch.setattr(httpx, "post", post)
    result = evaluate(build_graph(Classifier("case_details"), Extractor(), EXTRA_TASKS), "original query")
    assert result["results"]["call_http_api"] == {"status_code": 200, "data": [{"amount": 100}]}


def test_unknown_skips_http(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP must not run for unknown context")
    monkeypatch.setattr(httpx, "post", forbidden)
    assert evaluate(build_graph(Classifier(""), Extractor(), EXTRA_TASKS), "unknown") == {"context": "", "identifiers": {}}


@pytest.mark.parametrize("response,expected", [(httpx.Response(503),502), (httpx.Response(200,text="not json"),502)])
def test_invalid_response(configure, monkeypatch, response, expected):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: response)
    with pytest.raises(HTTPTaskError) as exc:
        call_http_api({"query":"q","context":"c","identifiers":{}})
    assert exc.value.status_code == expected


def test_timeout(configure, monkeypatch):
    def post(*args, **kwargs):
        raise httpx.ReadTimeout("timeout")
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(HTTPTaskError) as exc:
        call_http_api({"query":"q","context":"c","identifiers":{}})
    assert exc.value.status_code == 504


def test_empty_ids_still_calls_http(configure, monkeypatch):
    def post(*args, **kwargs):
        assert kwargs["json"]["identifiers"] == {}
        return httpx.Response(204)
    monkeypatch.setattr(httpx, "post", post)
    assert call_http_api({"query":"q","context":"c","identifiers":{}}) == {"status_code":204,"data":None}


def test_missing_configuration(monkeypatch):
    monkeypatch.delenv("DOWNSTREAM_API_URL", raising=False)
    with pytest.raises(HTTPTaskError, match="not configured"):
        call_http_api({})
