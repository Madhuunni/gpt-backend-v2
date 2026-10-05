from typing import Any, TypedDict


class QueryState(TypedDict, total=False):
    query: str
    context: str
    identifiers: dict[str, str]
    results: dict[str, Any]
