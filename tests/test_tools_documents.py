"""Tests for the office document tools, run through a real ToolRegistry."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader, PdfWriter

from jarvis.config import Settings
from jarvis.llm.types import ToolCall
from jarvis.tools.base import Risk, ToolContext
from jarvis.tools.builtin import documents
from jarvis.tools.registry import ToolRegistry

TOOLS = {
    "read_document", "create_document", "pdf_info", "pdf_merge",
    "pdf_split", "pdf_extract_pages", "pdf_rotate", "pdf_fill_form",
}
MARKDOWN = """# Quarterly Report

Intro with **bold**, *italic* and `code`: naïve café ₹500.

## Items

- first bullet
- second bullet

1. step one
2. step two

| Name | Amount |
|------|--------|
| Tea | ₹500 |
| Cake | ₹900 |

```
print("hi")
```
"""


@pytest.fixture
def reg() -> ToolRegistry:
    r = ToolRegistry()
    r.register_module(documents)
    return r


@pytest.fixture
def docs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "Documents"
    folder.mkdir()
    monkeypatch.setattr(documents, "_documents_folder", lambda: folder)
    return folder


def run(reg: ToolRegistry, name: str, **args: Any):
    ctx = ToolContext(settings=Settings())
    return asyncio.run(reg.execute(ToolCall(id="t", name=name, arguments=args), ctx))


def make_pdf(path: Path, pages: int = 3) -> Path:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for n in range(1, pages + 1):
        c.drawString(100, 700, f"Page number {n} content")
        c.showPage()
    c.save()
    path.write_bytes(buf.getvalue())
    return path


def page_texts(path: Path) -> list[str]:
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


# ------------------------------------------------------------------ registry
def test_registers_office_tools_all_safe(reg: ToolRegistry) -> None:
    assert {s.name for s in reg.specs(["office"])} == TOOLS
    assert all(s.risk is Risk.SAFE for s in reg.specs(["office"]))


def test_load_builtin_registers_tools() -> None:
    full = ToolRegistry().load_builtin()
    assert TOOLS <= {s.name for s in full.specs()}


# ---------------------------------------------------------------- page ranges
@pytest.mark.parametrize(
    "spec,total,expected",
    [
        ("1", 5, [0]),
        ("1-3", 5, [0, 1, 2]),
        ("1-3,7", 9, [0, 1, 2, 6]),
        (" 2 , 4-5 ", 5, [1, 3, 4]),
        ("4-", 6, [3, 4, 5]),
        ("3,1", 5, [2, 0]),
    ],
)
def test_parse_pages_valid(spec: str, total: int, expected: list[int]) -> None:
    assert documents.parse_pages(spec, total) == expected


@pytest.mark.parametrize("spec", ["", "  ", "0", "a", "3-1", "1,,2", "1-2-3", "-3", "9", "2-9", "1.5"])
def test_parse_pages_invalid(spec: str) -> None:
    with pytest.raises(ValueError):
        documents.parse_pages(spec, 5)


def test_parse_pages_no_pages() -> None:
    with pytest.raises(ValueError):
        documents.parse_pages("1", 0)


# -------------------------------------------------------------- create + read
@pytest.mark.parametrize("ext", ["md", "html", "pdf", "docx"])
def test_create_then_read_roundtrip(reg: ToolRegistry, tmp_path: Path, ext: str) -> None:
    out = tmp_path / f"report.{ext}"
    res = run(reg, "create_document", path=str(out), content_markdown=MARKDOWN)
    assert res.ok, res.content
    assert out.exists() and str(out) in res.content
    if ext == "pdf":
        assert "1 page" in res.content
    back = run(reg, "read_document", path=str(out))
    assert back.ok, back.content
    text = back.content
    for needle in ("Quarterly Report", "Items", "first bullet", "second bullet", "step one", "Tea", "Cake", "naïve café"):
        assert needle in text, f"{needle!r} missing from {ext} read-back"
    assert "₹500" in text


def test_docx_structure(reg: ToolRegistry, tmp_path: Path) -> None:
    from docx import Document

    out = tmp_path / "s.docx"
    assert run(reg, "create_document", path=str(out), content_markdown=MARKDOWN).ok
    doc = Document(str(out))
    styles = [p.style.name for p in doc.paragraphs]
    assert "Heading 1" in styles and "Heading 2" in styles
    assert styles.count("List Bullet") == 2 and styles.count("List Number") == 2
    assert len(doc.tables) == 1 and doc.tables[0].cell(1, 1).text == "₹500"
    runs = [r for p in doc.paragraphs for r in p.runs]
    assert any(r.bold and r.text == "bold" for r in runs)
    assert any(r.italic and r.text == "italic" for r in runs)


def test_html_has_stylesheet_and_table(reg: ToolRegistry, tmp_path: Path) -> None:
    out = tmp_path / "p.html"
    assert run(reg, "create_document", path=str(out), content_markdown=MARKDOWN).ok
    raw = out.read_text(encoding="utf-8")
    assert "<style>" in raw and "<table>" in raw and "<h1>" in raw and "<pre>" in raw


def test_explicit_format_overrides_extension(reg: ToolRegistry, tmp_path: Path) -> None:
    res = run(reg, "create_document", path=str(tmp_path / "x.pdf"), content_markdown="# Hi", format="html")
    assert res.ok and (tmp_path / "x.html").exists()
    res = run(reg, "create_document", path=str(tmp_path / "noext"), content_markdown="# Hi")
    assert res.ok and (tmp_path / "noext.md").exists()


def test_unknown_extension_and_empty_content(reg: ToolRegistry, tmp_path: Path) -> None:
    bad = run(reg, "create_document", path=str(tmp_path / "a.xyz"), content_markdown="# Hi")
    assert not bad.ok and "format" in bad.content
    empty = run(reg, "create_document", path=str(tmp_path / "a.md"), content_markdown="  \n")
    assert not empty.ok and not (tmp_path / "a.md").exists()


def test_create_never_overwrites(reg: ToolRegistry, tmp_path: Path) -> None:
    out = tmp_path / "keep.md"
    out.write_text("original", encoding="utf-8")
    res = run(reg, "create_document", path=str(out), content_markdown="# new")
    assert not res.ok and "already exists" in res.content and "Hint:" in res.content
    assert out.read_text(encoding="utf-8") == "original"


def test_bare_filename_goes_to_documents(reg: ToolRegistry, docs_dir: Path) -> None:
    res = run(reg, "create_document", path="plan.md", content_markdown="# Plan")
    assert res.ok and (docs_dir / "plan.md").exists()
    res = run(reg, "create_document", path="plan.pdf", content_markdown="# Plan")
    assert res.ok and (docs_dir / "plan.pdf").exists()
    assert str(docs_dir) in res.content


def test_bare_output_for_pdf_tools_goes_to_documents(reg: ToolRegistry, docs_dir: Path, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "src.pdf")
    res = run(reg, "pdf_extract_pages", path=str(src), pages="1", output="one.pdf")
    assert res.ok and (docs_dir / "one.pdf").exists()


# ----------------------------------------------------------------- read cases
def test_read_missing_file_hint(reg: ToolRegistry, tmp_path: Path) -> None:
    (tmp_path / "report.pdf").write_bytes(b"x")
    res = run(reg, "read_document", path=str(tmp_path / "reprot.pdf"))
    assert not res.ok and "does not exist" in res.content and "report.pdf" in res.content


def test_read_pdf_pages_and_count(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "src.pdf", pages=4)
    res = run(reg, "read_document", path=str(src), pages="2-3")
    assert res.ok and "4 pages" in res.content
    assert "Page number 2" in res.content and "Page number 3" in res.content
    assert "Page number 1 " not in res.content and "Page number 4" not in res.content
    bad = run(reg, "read_document", path=str(src), pages="9")
    assert not bad.ok and "4 pages" in bad.content


def test_read_scanned_pdf_has_no_text_layer(reg: ToolRegistry, tmp_path: Path) -> None:
    w = PdfWriter()
    w.add_blank_page(200, 200)
    out = tmp_path / "scan.pdf"
    with out.open("wb") as fh:
        w.write(fh)
    res = run(reg, "read_document", path=str(out))
    assert not res.ok and "no text layer" in res.content and "OCR" in res.content


def test_encrypted_pdf_message(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "plain.pdf", pages=2)
    w = PdfWriter(clone_from=str(src))
    w.encrypt("secret")
    locked = tmp_path / "locked.pdf"
    with locked.open("wb") as fh:
        w.write(fh)
    for name, args in [
        ("read_document", {"path": str(locked)}),
        ("pdf_merge", {"sources": [str(locked), str(src)], "output": str(tmp_path / "m.pdf")}),
        ("pdf_rotate", {"path": str(locked), "degrees": 90}),
        ("pdf_extract_pages", {"path": str(locked), "pages": "1", "output": str(tmp_path / "e.pdf")}),
    ]:
        res = run(reg, name, **args)
        assert not res.ok and "password-protected" in res.content, name
    info = run(reg, "pdf_info", path=str(locked))
    assert info.ok and "encrypted" in info.content and info.data == {"encrypted": True}
    assert not (tmp_path / "m.pdf").exists() and not (tmp_path / "e.pdf").exists()


def test_truncation_note_and_offset(reg: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_text("abcdefghij" * 100, encoding="utf-8")
    first = run(reg, "read_document", path=str(f), max_chars=300)
    assert first.ok and "Truncated" in first.content and "offset=300" in first.content
    nxt = run(reg, "read_document", path=str(f), max_chars=300, offset=900)
    assert nxt.ok and "Truncated" not in nxt.content
    past = run(reg, "read_document", path=str(f), offset=5000)
    assert not past.ok


def test_read_html_to_text(reg: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "p.html"
    f.write_text(
        "<html><head><title>T</title><style>p{}</style></head><body><h1>Head</h1>"
        "<script>evil()</script><p>Hello <b>world</b></p><ul><li>one</li></ul>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>",
        encoding="utf-8",
    )
    res = run(reg, "read_document", path=str(f))
    assert res.ok and "# Head" in res.content and "Hello world" in res.content
    assert "evil" not in res.content and "- one" in res.content and "| 1 | 2 |" in res.content


def test_read_unsupported_type(reg: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "a.xyz"
    f.write_text("hi")
    assert not run(reg, "read_document", path=str(f)).ok


# ---------------------------------------------------------------- pdf editing
def test_pdf_info_pages_metadata(reg: ToolRegistry, tmp_path: Path) -> None:
    w = PdfWriter(clone_from=str(make_pdf(tmp_path / "a.pdf", pages=2)))
    w.add_metadata({"/Title": "My Title", "/Author": "Asha"})
    out = tmp_path / "meta.pdf"
    with out.open("wb") as fh:
        w.write(fh)
    res = run(reg, "pdf_info", path=str(out))
    assert res.ok and "2 pages" in res.content and "My Title" in res.content
    assert "Asha" in res.content and "encrypted: no" in res.content and "No fillable" in res.content


def test_merge(reg: ToolRegistry, tmp_path: Path) -> None:
    a, b = make_pdf(tmp_path / "a.pdf", 2), make_pdf(tmp_path / "b.pdf", 3)
    out = tmp_path / "m.pdf"
    res = run(reg, "pdf_merge", sources=[str(a), str(b)], output=str(out))
    assert res.ok and "5 pages" in res.content
    assert len(PdfReader(str(out)).pages) == 5
    again = run(reg, "pdf_merge", sources=[str(a), str(b)], output=str(out))
    assert not again.ok and "already exists" in again.content
    assert not run(reg, "pdf_merge", sources=[str(a)], output=str(tmp_path / "z.pdf")).ok
    assert not run(reg, "pdf_merge", sources=[str(a), str(b)], output=str(tmp_path / "z.txt")).ok


def test_split(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "book.pdf", 5)
    res = run(reg, "pdf_split", path=str(src), ranges=["1-2", "3", "4-5"])
    assert res.ok, res.content
    files = res.data
    assert [len(PdfReader(f).pages) for f in files] == [2, 1, 2]
    assert "Page number 3" in page_texts(Path(files[1]))[0]
    assert Path(files[0]).name == "book - pages 1-2.pdf"
    again = run(reg, "pdf_split", path=str(src), ranges=["1-2"])
    assert not again.ok and "already exists" in again.content
    out_dir = tmp_path / "out"
    to_folder = run(reg, "pdf_split", path=str(src), ranges=["2"], output_folder=str(out_dir))
    assert to_folder.ok and (out_dir / "book - pages 2.pdf").exists()
    assert not run(reg, "pdf_split", path=str(src), ranges=["1-9"]).ok
    assert not run(reg, "pdf_split", path=str(src), ranges=[]).ok


def test_split_collision_writes_nothing(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "book.pdf", 3)
    (tmp_path / "book - pages 2.pdf").write_bytes(b"existing")
    res = run(reg, "pdf_split", path=str(src), ranges=["1", "2"])
    assert not res.ok
    assert not (tmp_path / "book - pages 1.pdf").exists()
    assert (tmp_path / "book - pages 2.pdf").read_bytes() == b"existing"


def test_extract_pages(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "src.pdf", 5)
    out = tmp_path / "pick.pdf"
    res = run(reg, "pdf_extract_pages", path=str(src), pages="5,2-3", output=str(out))
    assert res.ok and "3 pages" in res.content
    assert [t.split()[2] for t in page_texts(out)] == ["5", "2", "3"]
    assert not run(reg, "pdf_extract_pages", path=str(src), pages="0", output=str(tmp_path / "q.pdf")).ok
    assert not run(reg, "pdf_extract_pages", path=str(src), pages="1", output=str(out)).ok


def test_rotate(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_pdf(tmp_path / "src.pdf", 3)
    res = run(reg, "pdf_rotate", path=str(src), degrees=90, pages="2")
    assert res.ok, res.content
    out = tmp_path / "src - rotated.pdf"
    assert [p.rotation for p in PdfReader(str(out)).pages] == [0, 90, 0]
    assert [p.rotation for p in PdfReader(str(src)).pages] == [0, 0, 0]
    custom = tmp_path / "all.pdf"
    assert run(reg, "pdf_rotate", path=str(src), degrees=180, output=str(custom)).ok
    assert [p.rotation for p in PdfReader(str(custom)).pages] == [180, 180, 180]
    assert not run(reg, "pdf_rotate", path=str(src), degrees=90).ok  # default output exists now
    bad = run(reg, "pdf_rotate", path=str(src), degrees=45)
    assert not bad.ok and "invalid arguments" in bad.content


def make_form_pdf(path: Path) -> Path:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.drawString(50, 780, "Application form")
    c.acroForm.textfield(name="FullName", x=50, y=700, width=200, height=20, borderStyle="inset")
    c.acroForm.textfield(name="City", x=50, y=650, width=200, height=20, value="Pune")
    c.save()
    return path


def test_fill_form(reg: ToolRegistry, tmp_path: Path) -> None:
    src = make_form_pdf(tmp_path / "form.pdf")
    info = run(reg, "pdf_info", path=str(src))
    assert info.ok and "FullName" in info.content and "City" in info.content and "Pune" in info.content
    out = tmp_path / "filled.pdf"
    res = run(reg, "pdf_fill_form", path=str(src), values={"FullName": "Asha Rao", "City": "Delhi"}, output=str(out))
    assert res.ok, res.content
    fields = PdfReader(str(out)).get_fields()
    assert fields["FullName"]["/V"] == "Asha Rao" and fields["City"]["/V"] == "Delhi"
    assert PdfReader(str(src)).get_fields()["City"]["/V"] == "Pune"
    bad = run(reg, "pdf_fill_form", path=str(src), values={"Nope": "x"}, output=str(tmp_path / "f2.pdf"))
    assert not bad.ok and "Nope" in bad.content and "FullName" in bad.content
    assert not (tmp_path / "f2.pdf").exists()
    again = run(reg, "pdf_fill_form", path=str(src), values={"City": "x"}, output=str(out))
    assert not again.ok and "already exists" in again.content
    assert not run(reg, "pdf_fill_form", path=str(src), values={}, output=str(tmp_path / "f3.pdf")).ok


def test_pdf_tools_reject_non_pdf_and_missing(reg: ToolRegistry, tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("hi")
    assert "not a PDF" in run(reg, "pdf_rotate", path=str(txt), degrees=90).content
    missing = run(reg, "pdf_info", path=str(tmp_path / "ghost.pdf"))
    assert not missing.ok and "does not exist" in missing.content
