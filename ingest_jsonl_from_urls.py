"""Fetch syllabus documents from URLs listed in JSONL and emit JSONL with plain `text`.

End-to-end pipeline (after this step, downstream expects embedded text):

1. ``python syllabi_index_csv_to_jsonl.py`` (defaults: ``us_freshman_core_syllabi_index.csv`` → ``data/ingested/us_freshman_core_syllabi_urls.jsonl``)
2. ``python ingest_jsonl_from_urls.py --input data/ingested/us_freshman_core_syllabi_urls.jsonl --output data/ingested/with_text.jsonl``
3. ``python process_syllabi_jsonl.py --input-jsonl data/ingested/with_text.jsonl --output-jsonl data/labeled/syllabus_entities.jsonl``
4. ``python build_finetune_dataset.py --input-jsonl data/ingested/with_text.jsonl --train-output data/finetune/train.jsonl --valid-output data/finetune/valid.jsonl``
5. ``python train_hf_structured_extractor.py --train-jsonl data/finetune/train.jsonl --valid-jsonl data/finetune/valid.jsonl``

Only fetch URLs you are permitted to retrieve. Use ``--allowed-host-suffixes`` and
``--allow-http`` deliberately; defaults require HTTPS and impose no host filter.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests

from pipeline_errors import log_pipeline_error
from process_syllabi_jsonl import load_jsonl, normalize_record_id

DEFAULT_TIMEOUT_SECONDS = 25.0
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SyllabusJsonlIngest/1.0; "
        "research and internal dataset preparation)"
    ),
}


def extension_from_response(url: str, content_type: str) -> str:
    lowered_url = url.lower()
    lowered_content_type = content_type.lower()
    if lowered_url.endswith(".pdf") or "application/pdf" in lowered_content_type:
        return "pdf"
    if "text/html" in lowered_content_type or "application/xhtml+xml" in lowered_content_type:
        return "html"
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return (guessed or ".bin").lstrip(".")


def sniff_is_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def extract_html_text(content: bytes, encoding: str | None) -> str:
    codec = encoding or "utf-8"
    return content.decode(codec, errors="ignore")


def resolve_document_format(url: str, content_type: str, content: bytes) -> str:
    fmt = extension_from_response(url, content_type)
    if fmt == "bin" and sniff_is_pdf(content):
        return "pdf"
    return fmt


def parse_host_suffixes(raw: str | None) -> list[str] | None:
    if not raw or not raw.strip():
        return None
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def host_matches_suffixes(hostname: str, suffixes: list[str]) -> bool:
    host = hostname.lower()
    return any(host == suffix or host.endswith(suffix) for suffix in suffixes)


def url_is_allowed(
    url: str,
    *,
    require_https: bool,
    allowed_host_suffixes: list[str] | None,
) -> tuple[bool, str | None]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "invalid_url"

    if require_https and parsed.scheme != "https":
        return False, "https_required"
    if not require_https and parsed.scheme not in {"http", "https"}:
        return False, "unsupported_scheme"

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "missing_host"

    if allowed_host_suffixes is not None and not host_matches_suffixes(hostname, allowed_host_suffixes):
        return False, "host_not_allowed"

    return True, None


def resolve_record_url(record: dict, url_field: str | None) -> str | None:
    if url_field:
        value = record.get(url_field)
        return str(value).strip() if isinstance(value, str) and value.strip() else None

    for key in ("source_url", "url", "href"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_url_key(url: str) -> str:
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""
    return f"{parsed.scheme}://{netloc}{path}"


def load_successful_request_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if row.get("ingest_fetch_status") != "ok":
            continue
        request_url = row.get("ingest_request_url")
        if isinstance(request_url, str) and request_url:
            seen.add(normalize_url_key(request_url))
    return seen


def extract_text_from_bytes(url: str, content_type: str, content: bytes) -> tuple[str, str]:
    fmt = resolve_document_format(url, content_type, content)
    if fmt == "pdf":
        return extract_pdf_text(content), fmt
    if fmt == "html":
        return extract_html_text(content, None), fmt
    raise ValueError(f"Unsupported document format: {fmt}")


def fetch_url(
    url: str,
    *,
    session: requests.Session,
    timeout: float,
) -> tuple[bytes, str, str]:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    final_url = response.url
    content_type = response.headers.get("Content-Type", "")
    return response.content, final_url, content_type


def build_output_row(
    record: dict,
    index: int,
    *,
    request_url: str,
    text: str,
    final_url: str,
    content_type: str,
    doc_format: str,
    status: str,
    error: str | None,
) -> dict:
    row = dict(record)
    row["text"] = text
    row["ingest_request_url"] = request_url
    row["ingest_final_url"] = final_url
    row["ingest_content_type"] = content_type
    row["ingest_format"] = doc_format
    row["ingest_fetch_status"] = status
    if error:
        row["ingest_error"] = error
    elif "ingest_error" in row:
        del row["ingest_error"]
    row.setdefault("id", normalize_record_id(record, index))
    return row


def default_ingest_input_error_log(output_path: str | Path) -> Path:
    out = Path(output_path)
    return out.parent / f"{out.stem}_input_errors.jsonl"


def ingest_jsonl_from_urls(
    input_path: str | Path,
    output_path: str | Path,
    *,
    url_field: str | None = None,
    allowed_host_suffixes: list[str] | None = None,
    require_https: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    dedupe_urls: bool = False,
    resume: bool = False,
    include_failures: bool = False,
    max_rows: int | None = None,
    session: requests.Session | None = None,
) -> tuple[int, int, int]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_error_log = default_ingest_input_error_log(output_path)
    if not resume:
        input_error_log.parent.mkdir(parents=True, exist_ok=True)
        input_error_log.write_text("", encoding="utf-8")

    rows = load_jsonl(
        input_path,
        strict=False,
        error_log_path=input_error_log,
        stage="ingest_load_input",
    )
    if max_rows is not None:
        rows = rows[:max_rows]

    http = session or requests.Session()
    http.headers.update(DEFAULT_HEADERS)

    written_ok = 0
    skipped = 0
    failed = 0

    seen_urls: set[str] = set()
    completed_urls: set[str] = set()
    if resume:
        completed_urls = load_successful_request_urls(output)

    mode = "a" if resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(rows, start=1):
            if not isinstance(record, dict):
                failed += 1
                log_pipeline_error(
                    input_error_log,
                    stage="ingest_row",
                    message="record is not a JSON object",
                    index=index,
                    path=str(input_path),
                )
                print(f"[{index}] skip: invalid record type")
                continue
            request_url = resolve_record_url(record, url_field)
            if not request_url:
                failed += 1
                if include_failures:
                    row = build_output_row(
                        record,
                        index,
                        request_url="",
                        text="",
                        final_url="",
                        content_type="",
                        doc_format="",
                        status="error",
                        error="missing_url",
                    )
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                print(f"[{index}] skip: no URL field")
                continue

            allowed, reason = url_is_allowed(
                request_url,
                require_https=require_https,
                allowed_host_suffixes=allowed_host_suffixes,
            )
            if not allowed:
                failed += 1
                if include_failures:
                    row = build_output_row(
                        record,
                        index,
                        request_url=request_url,
                        text="",
                        final_url="",
                        content_type="",
                        doc_format="",
                        status="error",
                        error=reason or "blocked",
                    )
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                print(f"[{index}] skip: {reason} ({request_url})")
                continue

            url_key = normalize_url_key(request_url)
            if dedupe_urls and url_key in seen_urls:
                skipped += 1
                print(f"[{index}] skip: duplicate URL ({request_url})")
                continue
            seen_urls.add(url_key)

            if resume and url_key in completed_urls:
                skipped += 1
                print(f"[{index}] skip: resume ({request_url})")
                continue

            try:
                content, final_url, content_type = fetch_url(request_url, session=http, timeout=timeout)
                text, doc_format = extract_text_from_bytes(final_url, content_type, content)
                if not (text or "").strip():
                    raise ValueError("extracted document text is empty")
                row = build_output_row(
                    record,
                    index,
                    request_url=request_url,
                    text=text,
                    final_url=final_url,
                    content_type=content_type,
                    doc_format=doc_format,
                    status="ok",
                    error=None,
                )
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                written_ok += 1
                print(f"[{index}] ok {doc_format} ({final_url})")
            except Exception as exc:
                failed += 1
                if include_failures:
                    row = build_output_row(
                        record,
                        index,
                        request_url=request_url,
                        text="",
                        final_url="",
                        content_type="",
                        doc_format="",
                        status="error",
                        error=str(exc),
                    )
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                print(f"[{index}] error: {exc} ({request_url})")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return written_ok, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch documents from URLs in JSONL and write JSONL with extracted plain text."
    )
    parser.add_argument("--input", required=True, help="Path to URL-only JSONL input.")
    parser.add_argument("--output", required=True, help="Path for JSONL output with text field.")
    parser.add_argument(
        "--url-field",
        default=None,
        help="JSON field holding the URL. If omitted, tries source_url, url, then href.",
    )
    parser.add_argument(
        "--allowed-host-suffixes",
        default="",
        help="Comma-separated host suffixes to allow (e.g. .edu,.gov). Empty means no host filter.",
    )
    parser.add_argument(
        "--allow-http",
        action="store_true",
        help="Allow http:// URLs (default: HTTPS only).",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--dedupe-urls", action="store_true", help="Skip duplicate request URLs within the input.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output and skip URLs that already completed successfully in the output file.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Write rows for failures (empty text, ingest_fetch_status=error) instead of only logging.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most this many input JSONL rows (after load; useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffixes = parse_host_suffixes(args.allowed_host_suffixes)
    written, skipped, failed = ingest_jsonl_from_urls(
        args.input,
        args.output,
        url_field=args.url_field,
        allowed_host_suffixes=suffixes,
        require_https=not args.allow_http,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        dedupe_urls=args.dedupe_urls,
        resume=args.resume,
        include_failures=args.include_failures,
        max_rows=args.max_rows,
    )
    print(f"Written OK: {written}, skipped: {skipped}, failed: {failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
