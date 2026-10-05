# FastAPI + LangGraph with ContextClassifier and IDTagger

Uses the original trained models from `pytorch_context_classifier.zip` and
`pytorch_id_extractor.zip`. No external LLM or model downloads are used at inference.
The original weights, vocabularies, normalization, label mappings and decoding are
preserved. Python 3.10 / PyTorch 2.2.1 are the dependency baseline.

## Run

Extract the ZIP, enter `query_pipeline`, then run in your existing environment:

```bash
conda activate stock_pred
python -m pip install -r requirements.txt
python check_environment.py
export DOWNSTREAM_API_URL="https://your-service.example/api/details"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API.
Both trained artifact bundles are included; no retraining is required.
If `import torch` still fails with `iJIT_NotifyEvent`, repair the native environment
first; changing the FastAPI code cannot resolve that loader error.

## Request and response

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"query":"Show overpayment details for employee ID 0012345"}'
```

```json
{"context":"case_details","identifiers":{"case_id":"","employee_id":"0012345","sr_no":""}}
```

Execution order:

1. `ContextClassifier` evaluates the query using its original TF-IDF features.
2. If no features match or the score is below `INTENT_THRESHOLD`, return
   `{"context":"","identifiers":{}}` without invoking extraction.
3. Otherwise, pass the exact original query to `IDExtractor`, which uses `IDTagger`.
4. Run `call_http_api` and include its response under `results.call_http_api`.
   Return the context and identifiers. If all identifier values are empty,
   return `"identifiers": {}` while retaining the recognized context.

The included classifier labels are `case_details`, `installment_details`, and
`payment_history`. Labels come from metadata, not from hard-coded API routing.
The API cutoff defaults to 0.7; `export INTENT_THRESHOLD=0.0` matches the original
classifier CLI default. Tune the cutoff on your validation data. A softmax cutoff
cannot guarantee rejection of every out-of-domain query.

The original IDTagger supports numeric IDs only, preserves leading zeros, and
keeps the first accepted mention of each field. Its saved confidence cutoff is
0.95. Unsupported alphanumeric IDs are not partially extracted. The original
extractor returns three empty string fields when nothing is found; the API
adapter converts that result to `{}`. Over-512-token extraction inputs return
`{}`. Partial valid extraction retains all three keys with missing fields empty.

Normal prediction outcomes return HTTP 200; invalid request bodies (including
blank or over-4000-character queries) return 422. Unexpected runtime errors
return 500. Missing or incompatible artifacts fail startup.

## Project layout

| Path | Role |
|---|---|
| app/main.py | FastAPI startup, request validation and JSON response |
| app/graph.py | Conditional LangGraph and extension registration |
| app/extensions.py | Add tasks after extraction |
| intent_classification/classifier.py | Original ContextClassifier, TF-IDF loader and training CLI |
| intent_classification/model.py | Exports the original ContextClassifier |
| intent_classification/predictor.py | In-memory classifier adapter returning context or empty string |
| identifier_extraction/model.py | Original bidirectional LSTM IDTagger |
| identifier_extraction/extractor.py | Original IDExtractor loader and prediction logic |
| identifier_extraction/common.py | Original tokenization and decoding |
| identifier_extraction/predictor.py | API adapter converting no IDs to empty JSON |
| artifacts/intent_classification/ | Original weights.pt and metadata.json |
| artifacts/identifier_extraction/ | Original model.pt and metadata.json |
| data/ | Original datasets, separated by task |
| evaluate.py | CLI running the same adapters and graph |

Artifacts are loaded once per worker at startup, on CPU. `ARTIFACT_DIR` can point
to an alternate root with these same two artifact subdirectories. Relative paths
resolve against the project directory. Replace each complete weight/metadata
pair together and restart workers. `TORCH_NUM_THREADS` defaults to 2.

## Add tasks

Edit `app/extensions.py`:

```python
from .graph import ExtraTask


def get_employee_information(state):
    employee_id = state["identifiers"].get("employee_id")
    if not employee_id:
        return {"found": False}
    return {"employee_id": employee_id}  # Replace with your service call.


EXTRA_TASKS = [
    ExtraTask("get_employee_information", get_employee_information,
              require_identifiers=True),
]
```

Tasks run in registration order after extraction and return JSON objects, stored
under `results[task_name]`. Each receives a copy of query, context, identifiers
and earlier results. Unknown context skips all later tasks. An empty identifier
object skips tasks marked `require_identifiers=True`; check specific fields in
the task itself. Exceptions abort the request; side effects are not retried.
Edit `build_graph` for additional branching. See `examples/extended_app.py`.
This is a fixed conditional workflow, without an autonomous LLM planner.

## Train and evaluate

The original training code is included, with package-relative imports for the
extractor. Train to a separate location before deploying new artifacts:

```bash
python -m intent_classification.classifier train \
  --data data/intent_classification/demo_train.json \
  --output candidate_artifacts/intent_classification
python -m identifier_extraction.train \
  --data data/identifier_extraction \
  --output candidate_artifacts/identifier_extraction
```

Classifier data uses `{"intents":[{"context":"...","query_patterns":["..."]}]}`.
Extractor JSONL rows use `text` and `entities` with `start`, `end` (exclusive),
and `label` matching `case_id`, `employee_id`, or `sr_no`. Original datasets
illustrate both formats. Included original evaluation reports describe the
source models; they do not establish production accuracy for the combined API.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python evaluate.py "Show overpayment details for case ID 9001"
```

## Third task: HTTP API call

The default graph now runs `intent_classification`, then `extract_identifiers`,
then `call_http_api`. Unknown context still ends the graph immediately.
Configure the actual downstream service before making a recognized request:

```bash
export DOWNSTREAM_API_URL='https://your-service.example/api/details'
export DOWNSTREAM_TIMEOUT_SECONDS=15
# Optional, if the service uses bearer authentication:
export DOWNSTREAM_API_TOKEN='your-token'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The HTTP task sends POST JSON containing `query`, `context`, and `identifiers`.
Edit `http_task/task.py` if your service expects a different payload or method.
The response body is available to subsequent tasks at
`state["results"]["call_http_api"]["data"]` and appears in the HTTP response:

```json
{
  "context": "case_details",
  "identifiers": {"case_id": "", "employee_id": "12345", "sr_no": ""},
  "results": {
    "call_http_api": {
      "status_code": 200,
      "data": {"employee_name": "Example", "amount": 100}
    }
  }
}
```

Any JSON value is supported, including arrays. HTTP 204 becomes `data: null`.
The task runs even if extraction returns `{}`. Set `require_identifiers=True`
on its `ExtraTask` registration to skip it for empty identifiers.
Append additional tasks to `EXTRA_TASKS` to process its response.

Timeouts return HTTP 504; network errors, unsuccessful HTTP status codes and
non-JSON bodies return HTTP 502. Missing URL configuration returns HTTP 500 on
recognized requests. The timeout is an HTTPX network-operation timeout, not a
whole-workflow deadline. Redirects and automatic retries are disabled.
No live downstream call was made while building this project; the HTTP task is
tested using mocked responses. Configure your service URL and verify its contract.

The standalone `evaluate.py` CLI still runs only the two model tasks; the
FastAPI default app registers the HTTP task through `app/extensions.py`.
