import pytest

from llm_cost.usage import aggregate, load_usage, parse_record


def test_parse_record_anthropic_shape_excludes_cache_from_input():
    record = parse_record(
        {
            "model": "claude-opus-5",
            "date": "2026-06-01",
            "usage": {
                "input_tokens": 18400,
                "output_tokens": 2100,
                "cache_read_input_tokens": 52000,
                "cache_creation_input_tokens": 9000,
            },
        }
    )
    assert record.input_tokens == 18400
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000


def test_parse_record_openai_shape_subtracts_cache_from_prompt_tokens():
    record = parse_record(
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 31000,
                "completion_tokens": 420,
                "prompt_tokens_details": {"cached_tokens": 24000},
            },
        }
    )
    assert record.input_tokens == 31000 - 24000
    assert record.cached_input_tokens == 24000
    assert record.cache_write_tokens == 0


def test_parse_record_keeps_extra_fields_for_grouping():
    record = parse_record(
        {
            "model": "gpt-4o-mini",
            "team": "search",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
    )
    assert record.group_key("team") == "search"
    assert record.group_key("model") == "gpt-4o-mini"
    assert "usage" not in record.fields


def test_parse_record_group_key_missing_field_raises():
    record = parse_record({"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    with pytest.raises(KeyError):
        record.group_key("team")


def test_parse_record_missing_model_raises():
    with pytest.raises(ValueError):
        parse_record({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def test_parse_record_missing_usage_raises():
    with pytest.raises(ValueError):
        parse_record({"model": "m"})


def test_parse_record_neither_shape_raises():
    with pytest.raises(ValueError):
        parse_record({"model": "m", "usage": {"tokens": 1}})


def test_parse_record_cached_exceeding_prompt_raises():
    with pytest.raises(ValueError):
        parse_record(
            {
                "model": "m",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            }
        )


def test_parse_record_negative_anthropic_input_tokens_raises():
    with pytest.raises(ValueError):
        parse_record(
            {
                "model": "m",
                "usage": {"input_tokens": -5, "output_tokens": 1},
            }
        )


def test_parse_record_negative_output_tokens_raises():
    with pytest.raises(ValueError):
        parse_record({"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": -1}})


def test_parse_record_negative_cache_write_tokens_raises():
    with pytest.raises(ValueError):
        parse_record(
            {
                "model": "m",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": -9000,
                },
            }
        )


def test_load_usage_negative_token_line_is_skipped_not_raised():
    text = "\n".join(
        [
            '{"model":"m","usage":{"prompt_tokens":10,"completion_tokens":-1}}',
            '{"model":"m","usage":{"prompt_tokens":10,"completion_tokens":1}}',
        ]
    )
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 1
    assert "output_tokens" in problems[0]


def test_load_usage_skips_malformed_lines_by_default():
    text = "\n".join(
        [
            '{"model":"m","usage":{"prompt_tokens":10,"completion_tokens":1}}',
            "not json",
            '{"usage":{"prompt_tokens":10,"completion_tokens":1}}',
            "",
        ]
    )
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 2
    assert problems[0].startswith("line 2:")
    assert problems[1].startswith("line 3:")


def test_load_usage_strict_raises_on_first_bad_line():
    text = '{"model":"m","usage":{"prompt_tokens":10,"completion_tokens":1}}\nnot json'
    with pytest.raises(ValueError):
        load_usage(text, strict=True)


def test_load_usage_ignores_blank_lines():
    records, problems = load_usage("\n\n   \n")
    assert records == []
    assert problems == []


def test_aggregate_sums_by_model_in_first_seen_order():
    records, _ = load_usage(
        "\n".join(
            [
                '{"model":"b","usage":{"prompt_tokens":10,"completion_tokens":1}}',
                '{"model":"a","usage":{"prompt_tokens":20,"completion_tokens":2}}',
                '{"model":"b","usage":{"prompt_tokens":5,"completion_tokens":1}}',
            ]
        )
    )
    grouped = aggregate(records)
    assert [key for key, _ in grouped] == ["b", "a"]
    b_totals = dict(grouped)["b"]
    assert b_totals["calls"] == 2
    assert b_totals["input_tokens"] == 15
    assert b_totals["output_tokens"] == 2


def test_aggregate_accepts_custom_key():
    records, _ = load_usage(
        '{"model":"m","team":"agents","usage":{"prompt_tokens":10,"completion_tokens":1}}'
    )
    grouped = aggregate(records, key=lambda record: record.group_key("team"))
    assert grouped == [("agents", grouped[0][1])]
