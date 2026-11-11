"""The README's configuration reference must cover every setting, with the right default."""

from __future__ import annotations

import re
from pathlib import Path

from jarvis.config import Settings

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def table_rows() -> dict[str, str]:
    block = README.split("<!-- config-table:start", 1)[1].split("<!-- config-table:end", 1)[0]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"\| `([A-Z_]+)` \| (.*?) \| .* \|$", line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def shown_default(field) -> str:
    value = field.default if field.default_factory is None else field.default_factory()
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, list):
        return f"`{','.join(value)}`"
    return "empty" if value == "" else f"`{value}`"


def test_every_setting_is_documented_with_its_default():
    rows = table_rows()
    for name, field in Settings.model_fields.items():
        assert field.alias in rows, f"{field.alias} is missing from the README table"
        assert rows[field.alias] == shown_default(field), field.alias
    assert set(rows) == {f.alias for f in Settings.model_fields.values()}


def test_readme_no_longer_says_the_raw_urls_wait_for_a_public_repo():
    assert "once the repository is public" not in README
    assert "raw.githubusercontent.com/Golden-alt933/Project-Jarvis-personal-project/main/install.sh" in README
    assert "raw.githubusercontent.com/Golden-alt933/Project-Jarvis-personal-project/main/install.ps1" in README
