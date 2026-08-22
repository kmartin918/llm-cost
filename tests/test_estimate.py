import pytest

from llm_cost.estimate import estimate_cost
from llm_cost.pricing import ModelPrice


PRICE = ModelPrice("m", input=5.0, output=25.0, cached_input=0.5, cache_write=6.25)


def test_estimate_prices_each_token_class_separately():
    breakdown = estimate_cost(
        PRICE,
        input_tokens=12000,
        output_tokens=800,
        cached_input_tokens=52000,
        cache_write_tokens=9000,
    )
    assert breakdown.input_cost == pytest.approx(12000 * 5.0 / 1e6)
    assert breakdown.output_cost == pytest.approx(800 * 25.0 / 1e6)
    assert breakdown.cached_input_cost == pytest.approx(52000 * 0.5 / 1e6)
    assert breakdown.cache_write_cost == pytest.approx(9000 * 6.25 / 1e6)
    assert breakdown.total_cost == pytest.approx(
        breakdown.input_cost
        + breakdown.output_cost
        + breakdown.cached_input_cost
        + breakdown.cache_write_cost
    )


def test_estimate_scales_with_calls():
    one = estimate_cost(PRICE, input_tokens=1000, output_tokens=100, calls=1)
    fifty = estimate_cost(PRICE, input_tokens=1000, output_tokens=100, calls=50)
    assert fifty.total_cost == pytest.approx(one.total_cost * 50)
    assert fifty.input_tokens == one.input_tokens * 50
    assert fifty.cost_per_call == pytest.approx(one.total_cost)


def test_estimate_defaults_to_zero_tokens_and_one_call():
    breakdown = estimate_cost(PRICE)
    assert breakdown.total_cost == 0.0
    assert breakdown.calls == 1


def test_estimate_zero_calls_gives_zero_cost_per_call():
    breakdown = estimate_cost(PRICE, input_tokens=1000, calls=0)
    assert breakdown.cost_per_call == 0.0


def test_estimate_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, input_tokens=-1)


def test_estimate_rejects_non_integer_tokens():
    with pytest.raises(ValueError):
        estimate_cost(PRICE, input_tokens=12.5)


def test_breakdown_to_dict_rounds_costs():
    breakdown = estimate_cost(PRICE, input_tokens=1, output_tokens=1)
    payload = breakdown.to_dict()
    assert payload["model"] == "m"
    assert payload["total_cost"] == round(breakdown.total_cost, 6)
