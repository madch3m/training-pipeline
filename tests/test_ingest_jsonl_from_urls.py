import json
from pathlib import Path

import fitz

from ingest_jsonl_from_urls import (
    host_matches_suffixes,
    ingest_jsonl_from_urls,
    parse_host_suffixes,
    resolve_document_format,
    resolve_record_url,
    url_is_allowed,
)


def minimal_pdf_bytes(text: str = "CS 101 Syllabus\nGrading: Homework 40%\n") -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    return document.tobytes()


class FakeResponse:
    def __init__(self, content: bytes, final_url: str, content_type: str) -> None:
        self.content = content
        self.url = final_url
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, mapping: dict[str, FakeResponse]) -> None:
        self.headers: dict[str, str] = {}
        self._mapping = mapping

    def get(self, url: str, timeout: float | None = None, allow_redirects: bool = True) -> FakeResponse:
        if url not in self._mapping:
            raise AssertionError(f"unexpected url: {url}")
        return self._mapping[url]


def test_resolve_record_url_priority_and_custom_field():
    assert resolve_record_url({"source_url": "https://a", "url": "https://b"}, None) == "https://a"
    assert resolve_record_url({"url": "https://b"}, None) == "https://b"
    assert resolve_record_url({"href": "https://c"}, None) == "https://c"
    assert resolve_record_url({"link": "https://x"}, "link") == "https://x"
    assert resolve_record_url({"text": "nope"}, None) is None


def test_parse_host_suffixes_empty_means_none():
    assert parse_host_suffixes("") is None
    assert parse_host_suffixes("  ") is None
    assert parse_host_suffixes(".edu,.gov") == [".edu", ".gov"]


def test_url_is_allowed_https_and_suffixes():
    ok, _ = url_is_allowed("https://example.edu/path", require_https=True, allowed_host_suffixes=None)
    assert ok

    ok, reason = url_is_allowed("http://example.edu/path", require_https=True, allowed_host_suffixes=None)
    assert not ok
    assert reason == "https_required"

    ok, reason = url_is_allowed(
        "https://evil.com/x",
        require_https=True,
        allowed_host_suffixes=[".edu"],
    )
    assert not ok
    assert reason == "host_not_allowed"


def test_host_matches_suffixes():
    assert host_matches_suffixes("course.example.edu", [".edu"])
    assert not host_matches_suffixes("example.education", [".edu"])


def test_resolve_document_format_sniffs_pdf_octet_stream():
    pdf_bytes = minimal_pdf_bytes()
    fmt = resolve_document_format(
        "https://example.edu/download",
        "application/octet-stream",
        pdf_bytes,
    )
    assert fmt == "pdf"


def test_ingest_jsonl_pdf_and_html(tmp_path: Path):
    pdf_url = "https://example.edu/syllabus.pdf"
    html_url = "https://example.org/page.html"
    pdf_bytes = minimal_pdf_bytes("MATH 201 syllabus text")
    html_body = b"<html><body><p>BIO 110 syllabus</p></body></html>"

    session = FakeSession(
        {
            pdf_url: FakeResponse(pdf_bytes, pdf_url, "application/pdf"),
            html_url: FakeResponse(html_body, html_url, "text/html; charset=utf-8"),
        }
    )

    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps({"id": "a", "source_url": pdf_url}) + "\n"
        + json.dumps({"id": "b", "url": html_url}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"

    ok, skipped, failed = ingest_jsonl_from_urls(
        input_path,
        output_path,
        session=session,
    )

    assert (ok, skipped, failed) == (2, 0, 0)
    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["ingest_fetch_status"] == "ok"
    assert "MATH 201" in lines[0]["text"]
    assert lines[0]["ingest_format"] == "pdf"
    assert lines[1]["ingest_format"] == "html"
    assert "BIO 110" in lines[1]["text"]


def test_ingest_dedupe_urls(tmp_path: Path):
    url = "https://example.edu/one.pdf"
    pdf_bytes = minimal_pdf_bytes("dup")
    session = FakeSession({url: FakeResponse(pdf_bytes, url, "application/pdf")})

    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        json.dumps({"id": "1", "source_url": url}) + "\n" + json.dumps({"id": "2", "source_url": url}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"

    ok, skipped, failed = ingest_jsonl_from_urls(
        input_path,
        output_path,
        dedupe_urls=True,
        session=session,
    )

    assert ok == 1
    assert skipped == 1
    assert failed == 0


def test_ingest_respects_max_rows(tmp_path: Path):
    pdf_bytes = minimal_pdf_bytes("x")
    urls = [f"https://example.edu/doc{i}.pdf" for i in range(5)]
    session = FakeSession({u: FakeResponse(pdf_bytes, u, "application/pdf") for u in urls})
    lines_in = "\n".join(json.dumps({"id": str(i), "source_url": u}) for i, u in enumerate(urls)) + "\n"
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(lines_in, encoding="utf-8")
    output_path = tmp_path / "out.jsonl"

    ok, skipped, failed = ingest_jsonl_from_urls(
        input_path,
        output_path,
        max_rows=2,
        session=session,
    )

    assert (ok, skipped, failed) == (2, 0, 0)
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_ingest_include_failures_for_blocked_scheme(tmp_path: Path):
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"id": "x", "source_url": "http://example.edu/a.pdf"}) + "\n", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"

    ok, skipped, failed = ingest_jsonl_from_urls(
        input_path,
        output_path,
        require_https=True,
        include_failures=True,
        session=FakeSession({}),
    )

    assert ok == 0
    assert skipped == 0
    assert failed == 1
    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["ingest_fetch_status"] == "error"
    assert row["text"] == ""


def test_ingest_resume_skips_completed_url(tmp_path: Path):
    url = "https://example.edu/resume.pdf"
    pdf_bytes = minimal_pdf_bytes("done")
    session = FakeSession({url: FakeResponse(pdf_bytes, url, "application/pdf")})

    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps({"id": "1", "source_url": url}) + "\n", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"

    ingest_jsonl_from_urls(input_path, output_path, session=session)
    first_len = len(output_path.read_text(encoding="utf-8").splitlines())

    ok, skipped, failed = ingest_jsonl_from_urls(
        input_path,
        output_path,
        resume=True,
        session=session,
    )

    assert ok == 0
    assert skipped == 1
    assert failed == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == first_len
