import json
from pathlib import Path

from collect_syllabi_agent import (
    FetchResult,
    collect_syllabi,
    looks_like_syllabus_text,
)


def test_looks_like_syllabus_text_accepts_expected_content():
    text = """
    Course Syllabus
    Office Hours: Tuesday 2pm
    Grading: Assignments 40%
    Required Readings listed weekly
    Attendance Policy applies to all students
    """
    assert looks_like_syllabus_text(text) is True


def test_looks_like_syllabus_text_rejects_non_syllabus_content():
    text = """
    Campus event announcement with a parking update and alumni dinner details.
    Please register early and review the venue logistics before arrival.
    """
    assert looks_like_syllabus_text(text) is False


def test_url_dedupes_before_fetch(tmp_path: Path):
    seen = []

    def fake_search(_query):
        return [
            {"href": "https://example.edu/syllabus-one.pdf"},
            {"href": "https://example.edu/syllabus-one.pdf"},
        ]

    def fake_fetch(url: str):
        seen.append(url)
        return FetchResult(
            content=b"pdf-a",
            format="pdf",
            final_url=url,
        )

    records = collect_syllabi(
        target_count=5,
        output_dir=tmp_path,
        sleep_seconds=0,
        queries=["query"],
        searcher=fake_search,
        fetcher=fake_fetch,
    )

    assert len(seen) == 1
    assert len(records) == 1


def test_hash_dedupes_distinct_urls(tmp_path: Path):
    def fake_search(_query):
        return [
            {"href": "https://example.edu/a.pdf"},
            {"href": "https://example.edu/b.pdf"},
        ]

    def fake_fetch(url: str):
        return FetchResult(
            content=b"same-content",
            format="pdf",
            final_url=url,
        )

    records = collect_syllabi(
        target_count=5,
        output_dir=tmp_path,
        sleep_seconds=0,
        queries=["query"],
        searcher=fake_search,
        fetcher=fake_fetch,
    )

    assert len(records) == 1
    assert Path(records[0].file_path).exists()


def test_metadata_and_index_are_created(tmp_path: Path):
    def fake_search(_query):
        return [{"href": "https://cs.mit.edu/course/syllabus.html"}]

    def fake_fetch(url: str):
        return FetchResult(
            content=b"<html><body>syllabus office hours grading assignments week 1</body></html>",
            format="html",
            final_url=url,
        )

    records = collect_syllabi(
        target_count=1,
        output_dir=tmp_path,
        sleep_seconds=0,
        queries=["query"],
        searcher=fake_search,
        fetcher=fake_fetch,
    )

    assert len(records) == 1
    metadata_path = tmp_path / "metadata" / "syllabus_001.json"
    index_path = tmp_path / "syllabi_index.csv"
    assert metadata_path.exists()
    assert index_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_url"] == "https://cs.mit.edu/course/syllabus.html"
    assert metadata["format"] == "html"
    assert metadata["school"] == "MIT"
