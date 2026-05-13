from edge.study_plan_runtime import merge_chunk_payloads, split_text_into_chunks


def test_split_text_into_chunks_fallback_without_headers():
    text = "x" * 9500
    chunks = split_text_into_chunks(text, max_chars=4000)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == len(text)


def test_merge_chunk_payloads_dedupes_and_merges_by_heading():
    p1 = {
        "document_id": "doc-1",
        "source_url": "https://example.edu/syl",
        "text_field": "text",
        "course_codes": ["CS101"],
        "instructors": [],
        "emails": [],
        "section_names": ["assignments"],
        "assignments": ["Homework 1"],
        "readings": [],
        "grading_weights": [],
        "due_dates": ["Sep 10,2026"],
        "course_dates": [],
        "concepts": [],
        "study_plan": [
            {
                "section_heading": "Assignments",
                "course_codes": [],
                "readings": [],
                "assignments": ["Homework 1"],
                "due_dates": ["Sep 10,2026"],
                "course_dates": [],
                "concepts": [],
                "grading_weights": [],
                "instructors": [],
                "emails": [],
            }
        ],
        "entities": [],
    }
    p2 = {
        **p1,
        "assignments": ["Homework 1", "Project"],
        "due_dates": ["Sep 10, 2026", "Oct 1, 2026"],
        "study_plan": [
            {
                "section_heading": "Assignments",
                "course_codes": [],
                "readings": [],
                "assignments": ["Project"],
                "due_dates": ["Oct 1, 2026"],
                "course_dates": [],
                "concepts": [],
                "grading_weights": [],
                "instructors": [],
                "emails": [],
            }
        ],
    }
    merged = merge_chunk_payloads([p1, p2])
    assert merged["document_id"] == "doc-1"
    assert merged["assignments"] == ["Homework 1", "Project"]
    assert merged["due_dates"] == ["Sep 10, 2026", "Oct 1, 2026"]
    assert len(merged["study_plan"]) == 1
    assert merged["study_plan"][0]["assignments"] == ["Homework 1", "Project"]
