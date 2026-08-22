import pytest

from llm_cost.pricing import default_pricing
from llm_cost.report import build_report, compare_models
from llm_cost.usage import load_usage


USAGE_LOG = "\n".join(
    [
        '{"model":"claude-opus-5","team":"agents",'
        '"usage":{"input_tokens":18400,"output_tokens":2100,'
        '"cache_read_input_tokens":52000,"cache_creation_input_tokens":9000}}',
        '{"model":"claude-opus-5","team":"agents",'
        '"usage":{"input_tokens":9200,"output_tokens":1400,'
        '"cache_read_input_tokens":52000}}',
        '{"model":"gpt-4o-mini","team":"search",'
        '"usage":{"prompt_tokens":31000,"completion_tokens":420,'
        '"prompt_tokens_details":{"cached_tokens":24000}}}',
        '{"model":"internal-router-v3","team":"ops",'
        '"usage":{"prompt_tokens":1000,"completion_tokens":100}}',
    ]
)


def _records():
    records, problems = load_usage(USAGE_LOG)
    assert problems == []
    return records


def test_build_report_groups_by_model_and_skips_unpriced():
    table = default_pricing()
    report = build_report(_records(), table, group_by="model")
    keys = [group.key for group in report.groups]
    assert keys == ["claude-opus-5", "gpt-4o-mini"]
    assert report.unknown_models == {"internal-router-v3"}
    assert report.skipped == 1
    assert report.total_calls == 3


def test_build_report_groups_by_arbitrary_field():
    table = default_pricing()
    report = build_report(_records(), table, group_by="team")
    keys = [group.key for group in report.groups]
    assert keys == ["agents", "search"]
    agents = report.groups[0]
    assert agents.calls == 2
    assert agents.input_tokens == 18400 + 9200


def test_build_report_prices_each_record_at_its_own_model_rate():
    table = default_pricing()
    report = build_report(_records(), table, group_by="model")
    opus_group = report.groups[0]
    price = table.resolve("claude-opus-5")
    expected = (
        (18400 + 9200) * price.input
        + (2100 + 1400) * price.output
        + (52000 + 52000) * price.cached_input
        + 9000 * price.cache_write
    ) / 1e6
    assert opus_group.cost == pytest.approx(expected)


def test_build_report_total_cost_matches_sum_of_groups():
    table = default_pricing()
    report = build_report(_records(), table, group_by="model")
    assert report.total_cost == pytest.approx(sum(g.cost for g in report.groups))


def test_build_report_to_dict_lists_unknown_models_sorted():
    table = default_pricing()
    report = build_report(_records(), table, group_by="model")
    payload = report.to_dict()
    assert payload["unknown_models"] == ["internal-router-v3"]
    assert payload["skipped"] == 1


def test_build_report_group_by_missing_field_raises():
    table = default_pricing()
    with pytest.raises(KeyError):
        build_report(_records(), table, group_by="nonexistent-field")


def test_compare_models_ranks_cheapest_first():
    table = default_pricing()
    results = compare_models(table, input_tokens=10000, output_tokens=1000)
    costs = [result["breakdown"].total_cost for result in results]
    assert costs == sorted(costs)
    assert results[0]["vs_cheapest"] == pytest.approx(1.0)


def test_compare_models_filters_by_provider():
    table = default_pricing()
    results = compare_models(table, input_tokens=1000, output_tokens=100, provider="anthropic")
    assert results
    assert all(result["provider"] == "anthropic" for result in results)


def test_compare_models_filters_by_shortlist():
    table = default_pricing()
    results = compare_models(
        table, input_tokens=1000, output_tokens=100, models=["gpt-4o", "gpt-4o-mini"]
    )
    assert sorted(result["model"] for result in results) == ["gpt-4o", "gpt-4o-mini"]


def test_compare_models_no_match_returns_empty_list():
    table = default_pricing()
    results = compare_models(
        table, input_tokens=1000, output_tokens=100, provider="does-not-exist"
    )
    assert results == []
