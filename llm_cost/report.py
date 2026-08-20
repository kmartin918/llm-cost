"""Turning priced usage records into grouped reports, and ranking models.

Every record is priced individually against its own model before it is
folded into a group, so a group that mixes models (grouping by team or by
date, say) still costs each call at the right rate rather than at some
blended average.
"""

from .estimate import estimate_cost
from .pricing import UnknownModelError

__all__ = ["GroupSummary", "Report", "build_report", "compare_models"]


class GroupSummary(object):
    """Running totals for one group in a :class:`Report`."""

    __slots__ = (
        "key",
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "cost",
    )

    def __init__(self, key):
        self.key = key
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0
        self.cost = 0.0

    def add(self, breakdown):
        self.calls += breakdown.calls
        self.input_tokens += breakdown.input_tokens
        self.output_tokens += breakdown.output_tokens
        self.cached_input_tokens += breakdown.cached_input_tokens
        self.cache_write_tokens += breakdown.cache_write_tokens
        self.cost += breakdown.total_cost

    @property
    def cost_per_call(self):
        if not self.calls:
            return 0.0
        return self.cost / self.calls

    def to_dict(self):
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost": round(self.cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }

    def __repr__(self):
        return "GroupSummary(%s, calls=%d, cost=%.6f)" % (
            self.key,
            self.calls,
            self.cost,
        )


class Report(object):
    """A priced, grouped usage log."""

    __slots__ = ("group_by", "groups", "unknown_models", "skipped")

    def __init__(self, group_by, groups, unknown_models, skipped):
        self.group_by = group_by
        self.groups = groups
        self.unknown_models = unknown_models
        self.skipped = skipped

    @property
    def total_calls(self):
        return sum(group.calls for group in self.groups)

    @property
    def total_cost(self):
        return sum(group.cost for group in self.groups)

    def to_dict(self):
        return {
            "group_by": self.group_by,
            "groups": [group.to_dict() for group in self.groups],
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 6),
            "unknown_models": sorted(self.unknown_models),
            "skipped": self.skipped,
        }


def build_report(records, table, group_by="model"):
    """Price every record against ``table`` and group the results.

    A record naming a model with no price is counted in
    ``Report.unknown_models`` and ``Report.skipped`` rather than folded into
    the total as zero.
    """
    order = []
    groups = {}
    unknown_models = set()
    skipped = 0
    for record in records:
        try:
            price = table.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            skipped += 1
            continue
        breakdown = estimate_cost(
            price,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )
        group_key = record.group_key(group_by)
        group = groups.get(group_key)
        if group is None:
            group = GroupSummary(group_key)
            groups[group_key] = group
            order.append(group_key)
        group.add(breakdown)
    return Report(group_by, [groups[key] for key in order], unknown_models, skipped)


def compare_models(table, input_tokens, output_tokens, calls=1, provider=None, models=None):
    """Price one call against every model in ``table``, cheapest first.

    Returns a list of dicts with ``model``, ``provider``, ``price``
    (the resolved :class:`~llm_cost.pricing.ModelPrice`), ``breakdown`` (the
    :class:`~llm_cost.estimate.CostBreakdown`) and ``vs_cheapest`` (the cost
    as a multiple of the cheapest model's cost).
    """
    names = models if models is not None else table.models()
    priced = []
    for name in names:
        price = table.resolve(name)
        if provider and price.provider != provider:
            continue
        breakdown = estimate_cost(
            price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            calls=calls,
        )
        priced.append((price, breakdown))
    priced.sort(key=lambda pair: pair[1].total_cost)
    if not priced:
        return []
    cheapest_cost = priced[0][1].total_cost
    results = []
    for price, breakdown in priced:
        vs_cheapest = breakdown.total_cost / cheapest_cost if cheapest_cost else 0.0
        results.append(
            {
                "model": price.model,
                "provider": price.provider,
                "price": price,
                "breakdown": breakdown,
                "vs_cheapest": vs_cheapest,
            }
        )
    return results
