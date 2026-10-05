"""Conditional two-task graph with optional sequential extension tasks."""
import copy
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from .state import QueryState


@dataclass(frozen=True)
class ExtraTask:
    name: str
    run: Callable[[QueryState], dict]
    require_identifiers: bool = False


def build_graph(classifier, extractor, extra_tasks: Sequence[ExtraTask] = ()):
    graph = StateGraph(QueryState)

    def classify(state):
        return {"context": classifier.predict(state["query"])}

    def extract(state):
        return {"identifiers": extractor.predict(state["query"])}

    graph.add_node("intent_classification", classify)
    graph.add_node("extract_identifiers", extract)
    graph.add_edge(START, "intent_classification")
    graph.add_conditional_edges("intent_classification", lambda s: "recognized" if s["context"] else "unknown",
                                {"recognized": "extract_identifiers", "unknown": END})
    previous = "extract_identifiers"
    reserved = {START, END, "intent_classification", "extract_identifiers", "query", "context", "identifiers", "results"}
    for task in extra_tasks:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", task.name) or task.name in reserved:
            raise ValueError("Invalid or duplicate extra task name: " + task.name)
        reserved.add(task.name)
        def run_extra(state, task=task):
            if task.require_identifiers and not state["identifiers"]:
                return {}
            # Extension code cannot mutate input state or overwrite core outputs.
            result = task.run(copy.deepcopy(state))
            if not isinstance(result, dict):
                raise TypeError("Extra tasks must return a JSON object")
            json.dumps(result, allow_nan=False)
            return {"results": {**state.get("results", {}), task.name: result}}
        graph.add_node(task.name, run_extra)
        graph.add_edge(previous, task.name)
        previous = task.name
    graph.add_edge(previous, END)
    return graph.compile()


def evaluate(graph, query):
    # Fresh state for every HTTP request; no cross-request memory/checkpointer.
    state = graph.invoke({"query": query, "context": "", "identifiers": {}, "results": {}})
    response = {"context": state["context"], "identifiers": state["identifiers"]}
    if state.get("results"):
        response["results"] = state["results"]
    return response
