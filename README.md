# Training on Google Colab

This repo fine-tunes a small instruct model (default: **Qwen2.5-0.5B-Instruct**) with **LoRA** to emit structured JSON for syllabus documents. On Colab you typically: clone → install deps → open or sync the pipeline notebook → run.

## 1. Create a notebook or open the bundled one

**Option A — Full pipeline (CSV → fetch → label → SFT → train)**  
After cloning (step 2), upload or open:

- `notebooks/run_full_syllabus_pipeline.ipynb`

**Option B — Train only** (you already have a JSONL with document text):  

- `notebooks/train_syllabus_extractor_colab.ipynb`

## 2. Enable GPU

Runtime → **Change runtime type** → **GPU** (T4 or better recommended for training).

## 3. Clone and enter the repo

Run in a cell:

```python
!git clone https://github.com/madch3m/training-pipeline.git /content/training_pipeline
%cd /content/training_pipeline
```

To update an existing clone:

```bash
%cd /content/training_pipeline
!git pull origin main
```

## 4. Install dependencies

**Training stack** (Transformers, TRL, PEFT, datasets, MLflow, PyMuPDF, etc.):

```bash
%pip install -q -e ".[train]"
```

**PyTorch with CUDA** (pick an index that matches Colab’s CUDA; if `%pip install -e ".[train]"` already pulled a working GPU torch, you can skip this):

```bash
# Example only — adjust cuXXX to match your Colab stack if needed:
# %pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Hugging Face Hub** (fewer rate limits, required for some models):

- Colab: **Secrets** → add `HF_TOKEN`, then in a cell:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Or paste a token for the session only: `os.environ["HF_TOKEN"] = "hf_..."` (avoid committing it).

## 5. Full pipeline notebook (recommended first run)

Open `notebooks/run_full_syllabus_pipeline.ipynb`. The first cells set:

- `REPO_ROOT = Path("/content/training_pipeline")`
- `os.chdir` and `sys.path`

Then they call **`syllabus_torch_compat.apply_torch_ao_compat_patches()`** before training to reduce Torch/TorchAO friction on Colab.

**Smoke test (few URLs, faster):**

```python
cfg.ingest_max_rows = 3   # or None for all rows in the URL JSONL
cfg.ingest_sleep_seconds = 1.0
```

**Skip training** while debugging data:

```python
STEPS = ("csv", "ingest", "process", "build")  # no "train"
```

**Allow empty train/valid** if you need the runner to finish without examples (optional):

```python
cfg.allow_empty_finetune = True
```

Artifacts (default paths under the repo):

| Stage | Output |
|--------|--------|
| URL list | `data/ingested/us_freshman_core_syllabi_urls.jsonl` |
| Fetched text | `data/ingested/us_freshman_core_syllabi_with_text.jsonl` |
| Regex/heuristic labels | `data/labeled/syllabus_entities.jsonl` |
| SFT splits | `data/finetune/train.jsonl`, `data/finetune/valid.jsonl` |
| LoRA + tokenizer | `artifacts/hf_syllabus_extractor/` |
| MLflow (default) | `mlruns/` (local to the clone) |

Errors are logged beside outputs as `*_errors.jsonl` / `*_csv_errors.jsonl` when stages run in tolerant mode (default).

## 6. MLflow on Colab

By default the trainer logs to **local** `mlruns` under the repo. To point elsewhere:

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:///content/training_pipeline/mlruns"
```

Or disable MLflow in code/config used by `train_hf_structured_extractor.py` (`--disable-mlflow` if you invoke the CLI directly).

## 7. Evaluation helpers (after you have data)

From `/content/training_pipeline`:

```bash
!python generate_pipeline_data_report.py --output-json reports/data_report.json
!python evaluate_structured_json.py --sft-jsonl data/finetune/train.jsonl --output-json reports/sft_json_eval.json
```

**After training** — run the adapter on held-out chat examples and score JSON validity plus per-field overlap (install `".[train]"` first, GPU recommended):

```bash
!python evaluate_trained_extractor.py \
  --adapter-path artifacts/hf_syllabus_extractor \
  --sft-jsonl data/finetune/valid.jsonl \
  --max-samples 50 \
  --predictions-jsonl reports/extractor_predictions.jsonl \
  --report-json reports/extractor_eval.json
```

On CPU-only machines pass `--device-map cpu` (slower; uses `float32`).

See `docs/ML_EVAL_RUBRIC.md`, `docs/TRAIN_REPORT_TEMPLATE.md`, and `docs/RISK_REGISTER.md` for checklists and reporting.

## 8. Zip and download artifacts

```python
from google.colab import files
!cd /content/training_pipeline && zip -rq bundle.zip artifacts/hf_syllabus_extractor data/finetune mlruns 2>/dev/null; true
files.download("/content/training_pipeline/bundle.zip")
```

## 9. Edge training and release gates

From `/content/training_pipeline` (or local repo root):

```bash
# 1) Pre-training data gates (parse/schema/split integrity)
python validate_training_readiness.py \
  --train-jsonl data/finetune/train.jsonl \
  --valid-jsonl data/finetune/valid.jsonl \
  --strict \
  --output-json reports/training_readiness.json

# 2) Frontier training (0.5B vs 1.5B) + winner report
python train_frontier.py \
  --train-jsonl data/finetune/train.jsonl \
  --valid-jsonl data/finetune/valid.jsonl \
  --report-json artifacts/frontier/frontier_report.json

# 3) Export edge artifacts (ONNX/GGUF/CoreML manifest)
python edge/export_quantized_model.py \
  --adapter-path artifacts/hf_syllabus_extractor \
  --output-dir artifacts/edge \
  --manifest-json artifacts/edge/quantization_manifest.json

# 4) Benchmark edge inference (latency, memory, quality)
python edge/benchmark_edge_inference.py \
  --adapter-path artifacts/hf_syllabus_extractor \
  --sft-jsonl data/finetune/valid.jsonl \
  --report-json reports/edge_benchmark.json

# 5) Go / no-go release gates + privacy-safe telemetry aggregate
python edge/release_gates.py --benchmark-json reports/edge_benchmark.json --strict
python edge/privacy_telemetry.py \
  --benchmark-json reports/edge_benchmark.json \
  --output-json reports/edge_telemetry_aggregate.json
```

Deployment thresholds and device matrix are documented in `docs/EDGE_DEPLOYMENT_PLAN.md`.

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| Torch / `torchao` / import errors | Pull latest `main`; rerun the cell that calls `apply_torch_ao_compat_patches()`; align `torch` + CUDA wheels with Colab. |
| Empty `text` after ingest | Many rows skipped — check `data/ingested/*_input_errors.jsonl` and rows with `ingest_fetch_status` / empty extraction. |
| `eval_loss` is `nan` | Very small validation set or few steps — normal for smoke tests; use more data or tune `eval_steps`. |
| Out of memory | Lower `--max-length`, batch size, or use a smaller base model. |

## License / usage

Only fetch and use syllabi you are **allowed** to access. See `docs/RISK_REGISTER.md` for PII and compliance notes.
