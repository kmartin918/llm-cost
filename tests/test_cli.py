import json

import pytest

from llm_cost.cli import main


USAGE_LOG = "\n".join(
    [
        '{"model":"claude-opus-5","team":"agents",'
        '"usage":{"input_tokens":18400,"output_tokens":2100,'
        '"cache_read_input_tokens":52000,"cache_creation_input_tokens":9000}}',
        '{"model":"internal-router-v3","team":"ops",'
        '"usage":{"prompt_tokens":1000,"completion_tokens":100}}',
    ]
)


@pytest.fixture
def usage_log(tmp_path):
    path = tmp_path / "usage.jsonl"
    path.write_text(USAGE_LOG, encoding="utf-8")
    return str(path)


def test_estimate_prints_table_and_exits_ok(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "12000", "--output", "800"])
    out = capsys.readouterr().out
    assert code == 0
    assert "claude-opus-5" in out
    assert "cost per call:" in out


def test_estimate_json_matches_manual_math(capsys):
    code = main(["estimate", "--model", "claude-opus-5", "--input", "1000", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == "claude-opus-5"
    assert payload["total_cost"] == pytest.approx(1000 * 5.00 / 1e6)


def test_estimate_unknown_model_exits_with_code_3(capsys):
    code = main(["estimate", "--model", "not-a-real-model"])
    err = capsys.readouterr().err
    assert code == 3
    assert "not-a-real-model" in err


def test_report_prints_grouped_table_and_notes_skipped(capsys, usage_log):
    code = main(["report", usage_log])
    out = capsys.readouterr().out
    assert code == 0
    assert "claude-opus-5" in out
    assert "TOTAL" in out
    assert "skipped 1 record(s) with no price: internal-router-v3" in out


def test_report_json_includes_unknown_models(capsys, usage_log):
    code = main(["report", usage_log, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["unknown_models"] == ["internal-router-v3"]
    assert payload["skipped"] == 1


def test_report_group_by_field(capsys, usage_log):
    code = main(["report", usage_log, "--group-by", "team"])
    out = capsys.readouterr().out
    assert code == 0
    assert "agents" in out


def test_report_missing_file_exits_with_bad_input(capsys):
    code = main(["report", "/no/such/file.jsonl"])
    err = capsys.readouterr().err
    assert code == 2
    assert "/no/such/file.jsonl" in err


def test_report_strict_exits_on_malformed_line(capsys, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    code = main(["report", str(path), "--strict"])
    err = capsys.readouterr().err
    assert code == 2
    assert "line 1" in err


def test_compare_ranks_models_cheapest_first(capsys):
    code = main(["compare", "--input", "10000", "--output", "1000", "--provider", "anthropic"])
    out = capsys.readouterr().out
    assert code == 0
    lines = [line for line in out.splitlines() if line.startswith("claude-")]
    assert len(lines) > 1
    assert lines[0].split()[-1] == "1.0x"


def test_compare_no_match_exits_with_bad_input(capsys):
    code = main(["compare", "--provider", "does-not-exist"])
    err = capsys.readouterr().err
    assert code == 2
    assert "no models matched" in err


def test_models_lists_builtin_table(capsys):
    code = main(["models"])
    out = capsys.readouterr().out
    assert code == 0
    assert "claude-opus-5" in out
    assert "built-in" in out


def test_pricing_missing_file_exits_with_bad_input(capsys):
    code = main(["--pricing", "/no/such/prices.json", "estimate", "--model", "claude-opus-5"])
    err = capsys.readouterr().err
    assert code == 2
    assert "/no/such/prices.json" in err


def test_pricing_override_before_subcommand(capsys, tmp_path):
    override = tmp_path / "prices.json"
    override.write_text(
        json.dumps({"models": {"claude-opus-5": {"input": 1.0, "output": 2.0}}}),
        encoding="utf-8",
    )
    code = main(["--pricing", str(override), "estimate", "--model", "claude-opus-5", "--input", "1000", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["input_cost"] == pytest.approx(1000 * 1.0 / 1e6)
