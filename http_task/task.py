"""Third graph task: POST classified query and return the downstream response."""
import math
import os

import httpx


class HTTPTaskError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def call_http_api(state):
    url = os.getenv("DOWNSTREAM_API_URL", "").strip()
    if not url:
        raise HTTPTaskError("DOWNSTREAM_API_URL is not configured", 500)
    timeout = float(os.getenv("DOWNSTREAM_TIMEOUT_SECONDS", "15"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise HTTPTaskError("DOWNSTREAM_TIMEOUT_SECONDS must be positive and finite", 500)
    payload = {
        "query": state["query"],
        "context": state["context"],
        "identifiers": state["identifiers"],
    }
    headers = {"Accept": "application/json"}
    token = os.getenv("DOWNSTREAM_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        # No automatic retries: the remote endpoint may have side effects.
        response = httpx.post(url, json=payload, headers=headers,
                              timeout=timeout, follow_redirects=False)
        if not 200 <= response.status_code < 300:
            raise HTTPTaskError("Downstream API returned an unsuccessful status")
        if response.status_code == 204:
            data = None
        else:
            try:
                data = response.json()
            except ValueError as exc:
                raise HTTPTaskError("Downstream API did not return valid JSON") from exc
        return {"status_code": response.status_code, "data": data}
    except httpx.TimeoutException as exc:
        raise HTTPTaskError("Downstream API timed out", 504) from exc
    except httpx.RequestError as exc:
        raise HTTPTaskError("Downstream API connection failed") from exc
