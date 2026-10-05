# Compatibility and verification

This integration was tested with Python 3.10.21, PyTorch 2.2.1+cpu,
NumPy 1.26.4, scikit-learn 1.4.2, FastAPI 0.115.8 and LangGraph 0.2.76.
All 24 tests passed. Two dependency deprecation warnings did not affect results.
The user's specific Conda CUDA/MKL environment was not available for testing.

The original ContextClassifier is a TF-IDF-fed neural classifier; IDTagger is a
bidirectional LSTM token classifier. Both original weight files load directly
with torch.load(weights_only=True) under PyTorch 2.2.1. They were copied unchanged,
not retrained or converted. Their metadata records the original training runtime;
that field does not impose an inference-runtime version requirement.

Changes from the preceding query_pipeline release:

- Replaced the GRU models and artifacts with the original ContextClassifier and
  IDTagger architectures, trained weights and preprocessing.
- Added scikit-learn for the original classifier's TF-IDF feature reconstruction.
- Preserved the three original context labels and numeric-only ID decoding.
- Added a configurable classifier confidence cutoff (API default 0.7).
- Converted all-empty extraction results to {}, preserving the HTTP contract.
- Kept conditional LangGraph routing, extension tasks, startup loading and JSON API.

Your installed CUDA-enabled torch 2.2.1 satisfies the torch pin; CPU-only wheels
are not required. Inference runs on CPU and does not need torchvision/torchaudio.
If your environment still raises iJIT_NotifyEvent during import torch, resolve the
native PyTorch/MKL dependency mismatch first. Python application changes cannot
repair an import-time shared-library symbol error. check_environment.py provides
the relevant diagnostic. Existing environment repair guidance still applies.

HTTP task update: added HTTPX 0.28.1 and a default POST task. The earlier
24-test result above describes the model integration release. HTTP task checks
use mocked HTTP responses; no real downstream service was supplied.

All 7 new HTTP task tests passed with Python 3.12 and HTTPX 0.28.1.
The full PyTorch model suite was not rerun for this HTTP-only update.
