# ML evaluation rubric: syllabus training pipeline

Use **Pass** / **Partial** / **Fail** per row. Add evidence (log path, commit SHA, screenshot, metric value) in the last column.

Evidence paths refer to this repository: `training_pipeline/` root.

---

## 1. Problem formulation and label quality

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| Task definition | Structured JSON extraction is explicit in prompts and schema keys | | [`build_finetune_dataset.py`](../build_finetune_dataset.py) |
| Silver vs gold | Team documents heuristic limits and known failure modes | | [`process_syllabi_jsonl.py`](../process_syllabi_jsonl.py) |
| JSON validity (training file) | 100% assistant messages parse as JSON (or documented exception count) | | Run `python evaluate_structured_json.py --sft-jsonl data/finetune/train.jsonl` |
| Schema stability | Assistant objects have expected keys (or tracked drift rate) | | Same command reports `schema_issues` |

---

## 2. Data pipeline: coverage, bias, leakage

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| Ingest representativeness | Domain / school distribution recorded vs index | | `python generate_pipeline_data_report.py --output-json reports/data.json` |
| Train/val split | No unintended overlap of `document_id` across splits | | `train_valid_ids.overlap_count` in data report |
| Temporal leakage | `download_timestamp` (if present) not leaked into model input unintentionally | | Inspect CSV vs user prompt |
| Truncation | Policy for `max_text_chars` vs trainer `max_length` documented | | [`build_finetune_dataset.py`](../build_finetune_dataset.py), [`train_hf_structured_extractor.py`](../train_hf_structured_extractor.py) |
| Empty / bad rows | Skips logged; pipeline completes with partial data | | `*_errors.jsonl`, tolerant defaults in process/build |

---

## 3. Training configuration (TRL / LoRA)

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| LoRA appropriateness | Target modules validated for base model or documented as default | | MLflow / training script |
| LR / schedule | Learning rate and warmup justified or swept once | | [`train_hf_structured_extractor.py`](../train_hf_structured_extractor.py) |
| Batch / memory | GPU OOM not observed at chosen settings | | Training log |
| Precision | bf16/fp16 matches hardware | | CLI flags |
| Eval frequency | Eval steps sane for dataset size (no chronic `nan` without explanation) | | MLflow metrics |
| Checkpoints | Save policy documented | | `save_steps`, `save_total_limit` |

---

## 4. Model evaluation (beyond loss)

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| JSON EM / key F1 | Held-out task metric defined and reported | | Custom eval or `evaluate_structured_json.py` on predictions |
| Constraint violations | Rate of invalid JSON from **model** tracked | | `--predictions-jsonl` on `evaluate_structured_json.py` |
| Generation settings | Inference doc_temperature / max_new_tokens fixed for eval | | Notebook / serving doc |
| Error analysis | Stratified slice (PDF vs HTML, domain) | | Notes |

---

## 5. Reproducibility and dependency hygiene

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| Version pins | Lockfile or `pip freeze` archived per release | | Artifact in MLflow or repo |
| Seeds | Documented seed(s); known GPU nondeterminism called out | | |
| Data lineage | Input CSV hash + git SHA captured | | `generate_pipeline_data_report.py` output |

---

## 6. MLOps and observability

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| MLflow params | Core hparams logged | | |
| Artifacts | Model artifact size acceptable | | |
| Custom metrics | JSON validity or parse rate logged for runs | | |
| Tracking URI | Remote vs local decision documented | | |

---

## 7. Security, compliance, operations

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| Fetch policy | Legal / ToS stance documented | | [`docs/RISK_REGISTER.md`](RISK_REGISTER.md) |
| PII | Email handling in syllabi acknowledged | | Risk register |
| Rate limiting | `sleep_seconds` and host policy | | [`ingest_jsonl_from_urls.py`](../ingest_jsonl_from_urls.py) |
| Failure modes | Structured error logs exist | | `pipeline_errors.py`, `*_errors.jsonl` |

---

## 8. Testing and CI

| Item | Pass criteria | Your rating | Evidence |
|------|----------------|-------------|----------|
| Unit coverage | Ingest, process, build, runner tests pass | | `pytest tests/` |
| ML smoke | Optional nightly GPU smoke | | CI config / manual log |

---

## Fault tolerance (implementation checklist)

| Layer | Pass criteria | Your rating | Evidence |
|-------|----------------|-------------|----------|
| CSV | Bad / empty rows logged; run continues | | `*_csv_errors.jsonl` |
| URL JSONL load | Bad lines skipped and logged | | `*_input_errors.jsonl` (ingest) |
| Process | Bad lines + per-record errors logged; empty output allowed when tolerant | | `syllabus_entities_errors.jsonl` |
| Build | Per-record failures logged; optional empty outputs | | `*_build_errors.jsonl` |
