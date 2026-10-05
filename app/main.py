import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from identifier_extraction.predictor import IdentifierExtractor
from intent_classification.predictor import IntentClassifier
from http_task.task import HTTPTaskError
from .extensions import EXTRA_TASKS
from .graph import build_graph, evaluate

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


class QueryResponse(BaseModel):
    context: str
    identifiers: dict[str, str]
    results: dict[str, Any] | None = None


def create_app(artifact_dir=None, extra_tasks=None):
    configured = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))
    directory = configured if configured.is_absolute() else ROOT / configured
    tasks = EXTRA_TASKS if extra_tasks is None else extra_tasks

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        # Fail at startup for missing/incompatible files. Do not turn a broken
        # deployment into a misleading empty intent response.
        classifier = IntentClassifier(directory / "intent_classification")
        extractor = IdentifierExtractor(directory / "identifier_extraction")
        app.state.graph = build_graph(classifier, extractor, tasks)
        yield
        del app.state.graph

    app = FastAPI(title="Intent and identifier query pipeline", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.post("/evaluate", response_model=QueryResponse, response_model_exclude_none=True)
    def evaluate_query(body: QueryRequest):
        try:
            return evaluate(app.state.graph, body.query)
        except HTTPTaskError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except Exception as exc:
            log.exception("Query pipeline execution failed")
            raise HTTPException(500, "Query evaluation failed") from exc

    return app


app = create_app()
