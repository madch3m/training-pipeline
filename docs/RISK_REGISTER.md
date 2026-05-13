# Risk register: syllabus training pipeline

Audience: engineers and stakeholders using [`pipeline_runner.py`](../pipeline_runner.py) and public syllabus sources.

## Data acquisition

- **Unauthenticated HTTP fetching** ([`ingest_jsonl_from_urls.py`](../ingest_jsonl_from_urls.py)) may violate site terms, robots.txt, or campus policies. **Mitigation:** only crawl hosts and paths you are authorized to use; set `--allowed-host-suffixes` and delays; keep audit logs (`*_errors.jsonl`).
- **Login / SSO pages** may be fetched if URLs are misclassified; content may be low quality or sensitive. **Mitigation:** manual URL curation where possible.

## Privacy / PII

- Syllabus PDFs and HTML often contain **names, emails, office hours, and student-facing policies**. Stored artifacts under `data/` are gitignored but may exist on disk or in Colab volumes. **Mitigation:** treat outputs as confidential; redact before external sharing; document retention.

## Label quality (silver supervision)

- Training targets come from **regex and heuristics** ([`process_syllabi_jsonl.py`](../process_syllabi_jsonl.py)). The model can inherit **systematic omissions** (e.g. course codes not matching the pattern) or **wrong labels** (`GRADING_WEIGHT` vs `PERCENT`). **Mitigation:** human or LLM-judged spot checks; separate **gold** eval set before production claims.

## Operational

- **Empty extraction** after ingest yields rows with no usable text; tolerant stages log and skip. Training with **zero examples** still raises unless `allow_empty_outputs` / pipeline flag is set—avoid silent bad runs.
- **Dependency drift**: `pyproject.toml` uses lower bounds for training stacks; reproduce with a **frozen** environment for serious runs.

## Legal / compliance (non-legal advice)

- This README does not provide legal advice. Consult institutional policy before bulk downloading or redistributing third-party syllabi.
