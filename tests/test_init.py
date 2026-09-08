import re
from pathlib import Path

import llm_cost


def test_all_exports_are_accessible_attributes():
    # This is the exact bug the package started from: __init__ named things
    # in __all__ that lived in modules which did not exist yet.
    for name in llm_cost.__all__:
        assert hasattr(llm_cost, name), "exported name %r is not an attribute of llm_cost" % name


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None
    assert llm_cost.__version__ == match.group(1)


def test_submodules_import_directly():
    import llm_cost.cli  # noqa: F401
    import llm_cost.estimate  # noqa: F401
    import llm_cost.pricing  # noqa: F401
    import llm_cost.report  # noqa: F401
    import llm_cost.table  # noqa: F401
    import llm_cost.usage  # noqa: F401
