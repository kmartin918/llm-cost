"""The ``llm-cost`` command line tool: estimate, report, compare, models."""

import argparse
import json
import sys

from .estimate import estimate_cost
from .pricing import UnknownModelError, default_pricing, load_pricing
from .report import build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import load_usage

__all__ = ["main"]

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_UNKNOWN_MODEL = 3


class CliError(Exception):
    """A user-facing error, carrying the exit code it should produce."""

    def __init__(self, message, exit_code=EXIT_BAD_INPUT):
        super(CliError, self).__init__(message)
        self.exit_code = exit_code


def _format_price(value):
    return "%.2f" % value


def _plural(count, word):
    return "%d %s%s" % (count, word, "" if count == 1 else "s")


def _nonnegative_int(flag):
    def parse(value):
        try:
            parsed = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("%s must be an integer" % flag)
        if parsed < 0:
            raise argparse.ArgumentTypeError("%s must not be negative" % flag)
        return parsed

    return parse


def _global_parent(suppress):
    # ``--pricing``/``--json`` are accepted before or after the subcommand.
    # When this copy is attached to a subparser it must not clobber a value
    # already set by the top-level parser, so its defaults are suppressed
    # and only show up in the namespace when the user actually passes them.
    default = argparse.SUPPRESS if suppress else None
    json_default = argparse.SUPPRESS if suppress else False
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--pricing", default=default, help="JSON file overriding the built-in price table"
    )
    parent.add_argument(
        "--json", action="store_true", default=json_default, help="emit machine-readable JSON"
    )
    return parent


def build_parser():
    top_parent = _global_parent(suppress=False)
    sub_parent = _global_parent(suppress=True)

    parser = argparse.ArgumentParser(prog="llm-cost", parents=[top_parent])
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser(
        "estimate", parents=[sub_parent], help="cost of one call, or a batch of identical calls"
    )
    estimate_parser.add_argument("--model", required=True)
    estimate_parser.add_argument("--input", type=_nonnegative_int("--input"), default=0)
    estimate_parser.add_argument("--output", type=_nonnegative_int("--output"), default=0)
    estimate_parser.add_argument(
        "--cached", type=_nonnegative_int("--cached"), default=0, dest="cached_input"
    )
    estimate_parser.add_argument(
        "--cache-write", type=_nonnegative_int("--cache-write"), default=0, dest="cache_write"
    )
    estimate_parser.add_argument("--calls", type=_nonnegative_int("--calls"), default=1)
    estimate_parser.set_defaults(handler=cmd_estimate)

    report_parser = subparsers.add_parser(
        "report", parents=[sub_parent], help="cost of a JSONL usage log"
    )
    report_parser.add_argument("path", help="JSONL usage log, one API call per line")
    report_parser.add_argument("--group-by", default="model", dest="group_by")
    report_parser.add_argument(
        "--strict", action="store_true", help="fail on the first malformed line instead of skipping it"
    )
    report_parser.set_defaults(handler=cmd_report)

    compare_parser = subparsers.add_parser(
        "compare", parents=[sub_parent], help="rank models by cost for the same call"
    )
    compare_parser.add_argument("--input", type=_nonnegative_int("--input"), default=0)
    compare_parser.add_argument("--output", type=_nonnegative_int("--output"), default=0)
    compare_parser.add_argument("--calls", type=_nonnegative_int("--calls"), default=1)
    compare_parser.add_argument("--provider")
    compare_parser.add_argument("--models", help="comma-separated shortlist of model names")
    compare_parser.set_defaults(handler=cmd_compare)

    models_parser = subparsers.add_parser(
        "models", parents=[sub_parent], help="list the resolved price table"
    )
    models_parser.set_defaults(handler=cmd_models)

    return parser


def _load_table(args):
    pricing = getattr(args, "pricing", None)
    if pricing:
        try:
            return load_pricing(pricing)
        except ValueError as error:
            raise CliError(str(error), EXIT_BAD_INPUT)
    return default_pricing()


def _resolve(table, model):
    try:
        return table.resolve(model)
    except UnknownModelError as error:
        raise CliError(str(error), EXIT_UNKNOWN_MODEL)


def cmd_estimate(args, table):
    price = _resolve(table, args.model)
    breakdown = estimate_cost(
        price,
        input_tokens=args.input,
        output_tokens=args.output,
        cached_input_tokens=args.cached_input,
        cache_write_tokens=args.cache_write,
        calls=args.calls,
    )

    if args.json:
        payload = breakdown.to_dict()
        payload["as_of"] = table.as_of
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    print(
        "%s  (%s, prices as of %s)"
        % (price.model, _plural(args.calls, "call"), table.as_of)
    )
    print()
    rows = [
        ["input", format_int(breakdown.input_tokens), _format_price(price.input), format_money(breakdown.input_cost)],
        ["cached input", format_int(breakdown.cached_input_tokens), _format_price(price.cached_input), format_money(breakdown.cached_input_cost)],
        ["cache write", format_int(breakdown.cache_write_tokens), _format_price(price.cache_write), format_money(breakdown.cache_write_cost)],
        ["output", format_int(breakdown.output_tokens), _format_price(price.output), format_money(breakdown.output_cost)],
        ["total", format_int(breakdown.total_tokens), "", format_money(breakdown.total_cost)],
    ]
    print(render_table(["item", "tokens", "$/1M", "cost"], rows))
    print()
    print("cost per call: %s" % format_money(breakdown.cost_per_call))
    return EXIT_OK


def cmd_report(args, table):
    try:
        with open(args.path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise CliError("could not read %s: %s" % (args.path, error), EXIT_BAD_INPUT)

    try:
        records, problems = load_usage(text, strict=args.strict)
    except ValueError as error:
        raise CliError(str(error), EXIT_BAD_INPUT)

    try:
        report = build_report(records, table, group_by=args.group_by)
    except KeyError as error:
        raise CliError(str(error), EXIT_BAD_INPUT)

    if args.json:
        payload = report.to_dict()
        payload["problems"] = problems
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    rows = []
    for group in report.groups:
        rows.append(
            [
                str(group.key),
                format_int(group.calls),
                format_int(group.input_tokens),
                format_int(group.cached_input_tokens),
                format_int(group.output_tokens),
                format_money(group.cost),
                format_money(group.cost_per_call),
            ]
        )
    total_input = sum(group.input_tokens for group in report.groups)
    total_cached = sum(group.cached_input_tokens for group in report.groups)
    total_output = sum(group.output_tokens for group in report.groups)
    rows.append(
        [
            "TOTAL",
            format_int(report.total_calls),
            format_int(total_input),
            format_int(total_cached),
            format_int(total_output),
            format_money(report.total_cost),
            "",
        ]
    )
    headers = [args.group_by, "calls", "input", "cached", "output", "cost", "$/call"]
    print(render_table(headers, rows))

    if report.skipped:
        print()
        print(
            "skipped %d record(s) with no price: %s"
            % (report.skipped, ", ".join(sorted(report.unknown_models)))
        )
    if problems:
        print()
        print("skipped %s with a malformed line:" % _plural(len(problems), "record"))
        for problem in problems:
            print("  %s" % problem)
    return EXIT_OK


def cmd_compare(args, table):
    models = None
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]

    try:
        results = compare_models(
            table,
            args.input,
            args.output,
            calls=args.calls,
            provider=args.provider,
            models=models,
        )
    except UnknownModelError as error:
        raise CliError(str(error), EXIT_UNKNOWN_MODEL)

    if not results:
        raise CliError("no models matched the given --provider/--models filters", EXIT_BAD_INPUT)

    if args.json:
        payload = {
            "as_of": table.as_of,
            "input_tokens": args.input,
            "output_tokens": args.output,
            "calls": args.calls,
            "results": [
                {
                    "model": result["model"],
                    "provider": result["provider"],
                    "breakdown": result["breakdown"].to_dict(),
                    "vs_cheapest": round(result["vs_cheapest"], 4),
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    print(
        "%s in + %s out, %s, prices as of %s"
        % (format_int(args.input), format_int(args.output), _plural(args.calls, "call"), table.as_of)
    )
    print()
    rows = [
        [
            result["model"],
            result["provider"],
            _format_price(result["price"].input),
            _format_price(result["price"].output),
            format_money(result["breakdown"].total_cost),
            "%.1fx" % result["vs_cheapest"],
        ]
        for result in results
    ]
    print(render_table(["model", "provider", "$/1M in", "$/1M out", "cost", "vs cheapest"], rows))
    return EXIT_OK


def cmd_models(args, table):
    if args.json:
        payload = {
            "as_of": table.as_of,
            "source": table.source,
            "models": dict(
                (name, table.resolve(name).to_dict()) for name in table.models()
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    print(
        "%d models, USD per 1M tokens, as of %s (source: %s)"
        % (len(table), table.as_of, table.source)
    )
    print()
    rows = []
    for name in table.models():
        price = table.resolve(name)
        rows.append(
            [
                name,
                price.provider,
                _format_price(price.input),
                _format_price(price.output),
                _format_price(price.cached_input),
                _format_price(price.cache_write),
            ]
        )
    headers = ["model", "provider", "input", "output", "cached input", "cache write"]
    print(render_table(headers, rows))
    return EXIT_OK


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        table = _load_table(args)
        return args.handler(args, table)
    except CliError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
