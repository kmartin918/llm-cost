"""Parsing and normalising usage log records.

SDKs disagree about whether a cached prefix is counted in the input total.
OpenAI's ``prompt_tokens`` includes it; Anthropic's ``input_tokens`` does
not. Everything here normalises to the exclusive form that
:func:`llm_cost.estimate.estimate_cost` expects, so a record's
``input_tokens`` is always the uncached remainder.
"""

import json

__all__ = ["UsageRecord", "parse_record", "load_usage", "aggregate"]


class UsageRecord(object):
    """One priceable API call, plus whatever other fields the log carried."""

    __slots__ = (
        "model",
        "date",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "fields",
    )

    def __init__(
        self,
        model,
        date,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        fields,
    ):
        self.model = model
        self.date = date
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.fields = fields

    def group_key(self, field):
        """Value to group this record by for ``--group-by field``."""
        if field == "model":
            return self.model
        if field == "date":
            return self.date
        if field not in self.fields:
            raise KeyError("record has no field %r" % (field,))
        return self.fields[field]

    def __repr__(self):
        return "UsageRecord(%s, in=%d, out=%d)" % (
            self.model,
            self.input_tokens,
            self.output_tokens,
        )


def _normalise_usage(usage):
    is_anthropic_shaped = (
        "input_tokens" in usage
        or "cache_read_input_tokens" in usage
        or "cache_creation_input_tokens" in usage
    )
    if is_anthropic_shaped:
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cached_input_tokens = int(usage.get("cache_read_input_tokens", 0))
        cache_write_tokens = int(usage.get("cache_creation_input_tokens", 0))
    elif "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        cached_input_tokens = int(details.get("cached_tokens", 0))
        if cached_input_tokens > prompt_tokens:
            raise ValueError("cached_tokens exceeds prompt_tokens")
        input_tokens = prompt_tokens - cached_input_tokens
        output_tokens = int(usage.get("completion_tokens", 0))
        cache_write_tokens = 0
    else:
        raise ValueError(
            "usage object has neither Anthropic- nor OpenAI-shaped token fields"
        )
    # A negative count here (a raw negative field in the log, not just the
    # prompt/cache subtraction above) would otherwise sail through as a valid
    # record and only blow up later, as an unhandled ValueError, when
    # estimate_cost multiplies it by a price.
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cached_input_tokens", cached_input_tokens),
        ("cache_write_tokens", cache_write_tokens),
    ):
        if value < 0:
            raise ValueError("%s must not be negative, got %d" % (name, value))
    return input_tokens, output_tokens, cached_input_tokens, cache_write_tokens


def parse_record(obj):
    """Normalise one decoded JSON object into a :class:`UsageRecord`."""
    if not isinstance(obj, dict):
        raise ValueError("usage record must be a JSON object")
    model = obj.get("model")
    if not model:
        raise ValueError("usage record is missing 'model'")
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage record is missing a 'usage' object")
    input_tokens, output_tokens, cached_input_tokens, cache_write_tokens = (
        _normalise_usage(usage)
    )
    fields = dict((key, value) for key, value in obj.items() if key != "usage")
    return UsageRecord(
        model=model,
        date=obj.get("date"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        fields=fields,
    )


def load_usage(text, strict=False):
    """Parse a JSONL usage log into records plus a list of problems.

    One malformed line is noted as ``"line N: reason"`` in the returned
    problems and skipped, so it cannot lose the rest of the log. Pass
    ``strict=True`` to raise ``ValueError`` on the first bad line instead.
    """
    records = []
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = parse_record(json.loads(line))
        except ValueError as error:
            message = "line %d: %s" % (number, error)
            if strict:
                raise ValueError(message)
            problems.append(message)
            continue
        records.append(record)
    return records, problems


def aggregate(records, key=None):
    """Sum token and call counts by ``key(record)``, default grouping by model.

    Returns an ordered list of ``(group_key, totals)`` pairs in first-seen
    order, where ``totals`` is a dict of ``calls``, ``input_tokens``,
    ``output_tokens``, ``cached_input_tokens`` and ``cache_write_tokens``.
    """
    if key is None:
        key = lambda record: record.model
    order = []
    totals = {}
    for record in records:
        group_key = key(record)
        bucket = totals.get(group_key)
        if bucket is None:
            bucket = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
            }
            totals[group_key] = bucket
            order.append(group_key)
        bucket["calls"] += 1
        bucket["input_tokens"] += record.input_tokens
        bucket["output_tokens"] += record.output_tokens
        bucket["cached_input_tokens"] += record.cached_input_tokens
        bucket["cache_write_tokens"] += record.cache_write_tokens
    return [(group_key, totals[group_key]) for group_key in order]
