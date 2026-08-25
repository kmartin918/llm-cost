from llm_cost.table import format_int, format_money, render_table


def test_format_int_groups_thousands():
    assert format_int(12000) == "12,000"
    assert format_int(0) == "0"
    assert format_int(1234567) == "1,234,567"


def test_format_money_defaults_to_four_decimals():
    assert format_money(0.06) == "$0.0600"


def test_format_money_respects_decimals_argument():
    assert format_money(1.5, decimals=2) == "$1.50"


def test_render_table_first_column_left_aligned_rest_right_aligned():
    headers = ["model", "cost"]
    rows = [["claude-opus-5", "$0.0800"], ["gpt-4o-mini", "$0.0031"]]
    lines = render_table(headers, rows).splitlines()
    assert lines[0] == "model" + " " * 13 + "cost"
    assert lines[2] == "claude-opus-5  $0.0800"
    assert lines[3] == "gpt-4o-mini  " + " " * 2 + "$0.0031"


def test_render_table_rule_width_matches_widest_cell_per_column():
    headers = ["a", "b"]
    rows = [["short", "1"], ["much-longer-value", "22"]]
    lines = render_table(headers, rows).splitlines()
    rule = lines[1].split("  ")
    assert rule == ["-" * len("much-longer-value"), "-" * len("22")]


def test_render_table_column_widths_grow_to_fit_the_widest_cell():
    headers = ["name", "n"]
    rows = [["x", "1"], ["a-very-long-name", "2"]]
    lines = render_table(headers, rows).splitlines()
    assert lines[0] == "name" + " " * 15 + "n"
    assert lines[2] == "x" + " " * 18 + "1"
    assert lines[3] == "a-very-long-name" + "  " + "2"


def test_render_table_no_rows_still_renders_header_and_rule():
    table = render_table(["model", "cost"], [])
    lines = table.splitlines()
    assert len(lines) == 2
    assert lines[1] == "-----  ----"
