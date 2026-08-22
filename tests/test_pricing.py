import pytest

from llm_cost.pricing import (
    BUILTIN_PRICING,
    ModelPrice,
    PricingTable,
    UnknownModelError,
    default_pricing,
    parse_pricing,
)


def test_model_price_defaults_cache_prices_to_input():
    price = ModelPrice("m", input=2.0, output=8.0)
    assert price.cached_input == 2.0
    assert price.cache_write == 2.0


def test_model_price_rejects_negative():
    with pytest.raises(ValueError):
        ModelPrice("m", input=-1.0, output=1.0)


def test_default_pricing_matches_builtin_table():
    table = default_pricing()
    assert table.source == "built-in"
    assert len(table) == len(BUILTIN_PRICING)
    assert "claude-opus-5" in table


def test_resolve_exact_and_case_insensitive():
    table = default_pricing()
    assert table.resolve("claude-opus-5").model == "claude-opus-5"
    assert table.resolve("CLAUDE-OPUS-5").model == "claude-opus-5"


def test_resolve_strips_provider_prefix():
    table = default_pricing()
    assert table.resolve("anthropic.claude-opus-5").model == "claude-opus-5"
    assert table.resolve("google/gemini-2.5-flash").model == "gemini-2.5-flash"


def test_resolve_tolerates_date_suffix_via_prefix_match():
    table = default_pricing()
    # No exact entry for this dated name; it should match the longest known
    # model name that the given string starts with.
    assert table.resolve("gpt-4o-mini-2026-01-31").model == "gpt-4o-mini"


def test_resolve_unknown_model_raises_without_repr_quoting():
    table = default_pricing()
    with pytest.raises(UnknownModelError) as excinfo:
        table.resolve("internal-router-v3")
    message = str(excinfo.value)
    assert "internal-router-v3" in message
    assert not message.startswith("'")


def test_resolve_empty_name_raises():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("")


def test_contains_reflects_resolve():
    table = default_pricing()
    assert "claude-opus-5" in table
    assert "internal-router-v3" not in table


def test_parse_pricing_merges_onto_builtin_by_default():
    table = parse_pricing({"models": {"internal-router-v3": {"input": 0.2, "output": 0.8}}})
    assert "claude-opus-5" in table
    assert table.resolve("internal-router-v3").input == 0.2


def test_parse_pricing_replace_starts_from_empty_table():
    table = parse_pricing(
        {"replace": True, "models": {"internal-router-v3": {"input": 0.2, "output": 0.8}}}
    )
    assert "claude-opus-5" not in table
    assert len(table) == 1


def test_parse_pricing_bare_model_map_shape():
    table = parse_pricing({"internal-router-v3": {"input": 0.2, "output": 0.8}})
    assert table.as_of == "unspecified"
    assert table.resolve("internal-router-v3").output == 0.8


def test_parse_pricing_missing_field_raises():
    with pytest.raises(ValueError):
        parse_pricing({"models": {"m": {"input": 1.0}}})


def test_parse_pricing_non_numeric_field_raises():
    with pytest.raises(ValueError):
        parse_pricing({"models": {"m": {"input": 1.0, "output": "a lot"}}})


def test_parse_pricing_empty_models_raises():
    with pytest.raises(ValueError):
        parse_pricing({"models": {}})


def test_parse_pricing_non_dict_raises():
    with pytest.raises(ValueError):
        parse_pricing([1, 2, 3])


def test_pricing_table_models_is_sorted():
    table = PricingTable({"b": ModelPrice("b", 1, 1), "a": ModelPrice("a", 1, 1)})
    assert table.models() == ["a", "b"]
