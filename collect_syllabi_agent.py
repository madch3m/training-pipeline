from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence
from urllib.parse import urlparse

import fitz
import requests
from duckduckgo_search import DDGS

DEFAULT_TARGET_COUNT = 100
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_QUERIES = [
    'site:.edu filetype:pdf syllabus "office hours"',
    'site:.edu filetype:pdf syllabus "grading policy"',
    'site:.edu filetype:pdf syllabus "required readings"',
    'site:.edu filetype:pdf syllabus "course schedule"',
    'site:.edu filetype:pdf syllabus "learning objectives"',
    'site:.edu "course syllabus" "office hours" "grading"',
    'site:.edu "syllabus" "required textbook" "week 1"',
    'site:.edu "syllabus" "assignments" "percent"',
]
SYLLABUS_SIGNALS = (
    "syllabus",
    "office hours",
    "grading",
    "required reading",
    "required readings",
    "course schedule",
    "learning objectives",
    "assignment",
    "assignments",
    "week 1",
    "textbook",
    "attendance policy",
)
LOGIN_HINTS = (
    "login",
    "sign-in",
    "signin",
    "shibboleth",
    "sso",
    "cas",
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PublicSyllabusCollector/1.0; "
        "public educational research)"
    )
}


@dataclass(frozen=True)
class SyllabusRecord:
    id: str
    source_url: str
    domain: str
    school: str
    department: str | None
    course: str | None
    query: str
    file_path: str
    format: str
    sha256: str
    status: str


@dataclass(frozen=True)
class FetchResult:
    content: bytes
    format: str
    final_url: str


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def looks_like_syllabus_text(text: str, minimum_hits: int = 3) -> bool:
    lowered = text.lower()
    return sum(1 for signal in SYLLABUS_SIGNALS if signal in lowered) >= minimum_hits


def is_allowed_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    hostname = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if parsed.scheme not in {"http", "https"}:
        return False
    if not hostname.endswith(".edu"):
        return False
    if any(hint in hostname or hint in path for hint in LOGIN_HINTS):
        return False
    return True


def extension_from_response(url: str, content_type: str) -> str:
    lowered_url = url.lower()
    lowered_content_type = content_type.lower()
    if lowered_url.endswith(".pdf") or "application/pdf" in lowered_content_type:
        return "pdf"
    if "text/html" in lowered_content_type or "application/xhtml+xml" in lowered_content_type:
        return "html"
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return (guessed or ".bin").lstrip(".")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def infer_school_from_domain(domain: str) -> str:
    host = domain.lower()
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) >= 2 and labels[-1] == "edu":
        return labels[-2].replace("-", " ").upper()
    return labels[0].replace("-", " ").upper()


def build_output_paths(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    paths = {
        "output_dir": output_dir,
        "raw_dir": raw_dir,
        "pdf_dir": raw_dir / "pdfs",
        "html_dir": raw_dir / "html",
        "metadata_dir": output_dir / "metadata",
        "index_path": output_dir / "syllabi_index.csv",
    }
    for key in ("output_dir", "raw_dir", "pdf_dir", "html_dir", "metadata_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def load_queries(query_file: str | None) -> list[str]:
    if not query_file:
        return list(DEFAULT_QUERIES)
    query_path = Path(query_file)
    return [
        line.strip()
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def search_results(query: str, max_results: int = 75) -> Iterator[dict]:
    with DDGS() as ddgs:
        yield from ddgs.text(query, max_results=max_results)


def safe_fetch(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> FetchResult | None:
    if not is_allowed_public_url(url):
        return None

    http = session or requests
    response = http.get(
        url,
        headers=HEADERS,
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()

    final_url = response.url
    if not is_allowed_public_url(final_url):
        return None

    content_type = response.headers.get("Content-Type", "")
    file_format = extension_from_response(final_url, content_type)
    content = response.content

    if file_format == "pdf":
        if len(content) < 10_000:
            return None
        try:
            text = extract_pdf_text(content)
        except Exception:
            return None
        if not looks_like_syllabus_text(text):
            return None
    elif file_format == "html":
        text = content.decode(response.encoding or "utf-8", errors="ignore")
        if not looks_like_syllabus_text(text):
            return None
    else:
        return None

    return FetchResult(content=content, format=file_format, final_url=final_url)


def write_index(index_path: Path, records: Sequence[SyllabusRecord]) -> None:
    if not records:
        return
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def persist_record(
    record_id: str,
    fetched: FetchResult,
    query: str,
    paths: dict[str, Path],
) -> SyllabusRecord:
    parsed = urlparse(fetched.final_url)
    domain = parsed.netloc
    school = infer_school_from_domain(domain)
    if fetched.format == "pdf":
        file_path = paths["pdf_dir"] / f"{record_id}.pdf"
    else:
        file_path = paths["html_dir"] / f"{record_id}.html"

    file_path.write_bytes(fetched.content)
    record = SyllabusRecord(
        id=record_id,
        source_url=fetched.final_url,
        domain=domain,
        school=school,
        department=None,
        course=None,
        query=query,
        file_path=str(file_path),
        format=fetched.format,
        sha256=sha256_bytes(fetched.content),
        status="downloaded",
    )
    metadata_path = paths["metadata_dir"] / f"{record_id}.json"
    metadata_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return record


def collect_syllabi(
    *,
    target_count: int = DEFAULT_TARGET_COUNT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    queries: Iterable[str] | None = None,
    fetcher: Callable[[str], FetchResult | None] | None = None,
    searcher: Callable[[str], Iterable[dict]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[SyllabusRecord]:
    query_list = list(queries or DEFAULT_QUERIES)
    paths = build_output_paths(Path(output_dir))
    fetch = fetcher or (lambda url: safe_fetch(url))
    run_search = searcher or search_results

    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    records: list[SyllabusRecord] = []

    for query in query_list:
        if len(records) >= target_count:
            break

        for result in run_search(query):
            if len(records) >= target_count:
                break

            candidate_url = result.get("href") or result.get("url")
            if not candidate_url or candidate_url in seen_urls:
                continue
            seen_urls.add(candidate_url)

            fetched = None
            try:
                fetched = fetch(candidate_url)
            except Exception as exc:
                print(f"Skipped: {candidate_url} ({exc})")
            finally:
                if sleep_seconds > 0:
                    sleeper(sleep_seconds)

            if not fetched:
                continue

            digest = sha256_bytes(fetched.content)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            record_id = f"syllabus_{len(records) + 1:03d}"
            record = persist_record(record_id, fetched, query, paths)
            records.append(record)
            print(f"[{len(records)}/{target_count}] {record.source_url}")

    write_index(paths["index_path"], records)
    print(f"Done. Collected {len(records)} syllabi.")
    print(f"Index: {paths['index_path']}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public university syllabi.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--query-file",
        default=None,
        help="Path to a newline-delimited file of search queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_syllabi(
        target_count=args.target_count,
        output_dir=args.output_dir,
        sleep_seconds=args.sleep_seconds,
        queries=load_queries(args.query_file),
    )


if __name__ == "__main__":
    main()
