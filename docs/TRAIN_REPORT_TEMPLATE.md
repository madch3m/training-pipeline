# Training run report template

Copy this section into a Notion page, ticket, or MLflow run description.

## Run metadata

- **Date**:
- **Git commit** (`git rev-parse HEAD`):
- **Host** (Colab / local / cloud):
- **GPU** (`nvidia-smi` summary if applicable):
- **MLflow experiment**:
- **MLflow run ID**:
- **Tracking URI**:

## Data snapshot

- **Index CSV path / hash**: (from `generate_pipeline_data_report.py` → `artifact_hashes.index_csv_sha256`)
- **Rows**: URL JSONL / text JSONL / entities / train / valid line counts
- **Fault tolerance**: attach or link `*_csv_errors.jsonl`, `*_input_errors.jsonl`, `*_errors.jsonl`, `*_build_errors.jsonl` if non-empty

## Hyperparameters

- **Model**: (e.g. `Qwen/Qwen2.5-0.5B-Instruct`)
- **Epochs**, **learning rate**, **batch size**, **grad accumulation**, **max_length**
- **LoRA**: r, alpha, dropout, full fine-tune flag
- **bf16/fp16**:

## Metrics

- **Train loss** (final / best):
- **Eval loss** (note if `nan` on tiny val — explain):
- **Train runtime**, steps, samples/sec
- **JSON validity (teacher labels)**: output of `evaluate_structured_json.py --sft-jsonl data/finetune/train.jsonl`

## Observations

- **Leakage / overlap**: `train_valid_ids` from data report
- **Truncation**: any long documents cut by `max_text_chars` or `max_length`
- **Incidents**: OOM, Hub rate limits, torch/torchao warnings

## Artifacts

- **Output directory** (adapter + tokenizer path):
- **Archive** (zip or Hub repo ID):

## Follow-ups

- [ ] Held-out JSON F1 / EM on predictions
- [ ] Grouped split by domain
- [ ] Lock dependencies (`pip freeze` > `constraints.txt`)
