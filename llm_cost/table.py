"""Rendering helpers shared by every report: comma-grouped integers,
fixed-point money, and the aligned tables printed by the CLI.
"""

__all__ = ["format_int", "format_money", "render_table"]


def format_int(value):
    """Comma-group an integer: ``12000`` -> ``"12,000"``."""
    return "{:,}".format(int(value))


def format_money(value, decimals=4):
    """Fixed-point dollar amount: ``0.06`` -> ``"$0.0600"``."""
    return "$%.*f" % (decimals, value)


def render_table(headers, rows):
    """Render pre-formatted cells as a monospace table.

    The first column is left-aligned (it holds names), every other column is
    right-aligned (they hold numbers), columns are separated by two spaces,
    and a dashed rule of matching width sits under the header. Callers are
    responsible for formatting each cell (``format_int``, ``format_money``,
    or plain ``str``) before it reaches this function.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def pad(cell, index):
        return cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])

    lines = ["  ".join(pad(cell, i) for i, cell in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(pad(cell, i) for i, cell in enumerate(row)))
    return "\n".join(lines)
