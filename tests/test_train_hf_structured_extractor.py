from train_hf_structured_extractor import dataset_needs_conversion, split_messages_for_sft


def test_split_messages_for_sft_creates_prompt_completion():
    example = {
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Extract syllabus fields."},
            {"role": "assistant", "content": '{"course_codes":["CS 101"]}'},
        ]
    }

    converted = split_messages_for_sft(example)

    assert [item["role"] for item in converted["prompt"]] == ["system", "user"]
    assert converted["completion"] == [{"role": "assistant", "content": '{"course_codes":["CS 101"]}'}]


def test_dataset_needs_conversion_detects_messages_only_shape():
    assert dataset_needs_conversion(["messages"]) is True
    assert dataset_needs_conversion(["prompt", "completion"]) is False
