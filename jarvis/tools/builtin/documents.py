"""Document tools: read PDF/Word/text, create documents from Markdown, edit PDFs.

Ported from Friday's document_skill: render-from-Markdown (the model only ever writes
Markdown), no-overwrite writes, read-back verification, and the PDF merge/split/rotate/
form-fill logic including the form-field readback checks. The Skill/Task wrappers and
sentence parsing are gone, and Friday's "basename in a fixed folder" rule is replaced by
full paths via ``resolve_path``. A bare file name for a NEW document goes to Documents.

Safety model: every write creates a NEW file and refuses to replace an existing one, so
nothing the user already has is ever modified (hence every tool is Risk.SAFE). Encrypted
PDFs are refused with a clear message rather than guessed at.
"""

from __future__ import annotations

import html as html_lib
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from jarvis.tools.base import ToolResult, tool
from jarvis.tools.paths import bad_path as _bad_path
from jarvis.tools.paths import decode_text as _decode
from jarvis.tools.paths import format_size as _fmt_size
from jarvis.tools.paths import is_binary as _is_binary
from jarvis.tools.paths import known_folder, not_found as _missing, resolve_path

log = logging.getLogger(__name__)

GROUP = "office"

# The registry cuts results at 12000 chars; leave room for the header and the note.
MAX_CHARS_CAP = 11_000
MAX_LISTED_FIELDS = 100
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
HTML_SUFFIXES = {".html", ".htm"}
DOC_SUFFIXES = {".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
FORMAT_SUFFIX = {"md": ".md", "html": ".html", "pdf": ".pdf", "docx": ".docx"}

# xhtml2pdf supports a small CSS subset and drops the rest silently, so keep this plain.
_CSS = """
@page { size: a4 portrait; margin: 2cm; }
body { font-family: %(body)s; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }
h1, h2, h3, h4 { font-family: %(body)s; color: #111111; }
h1 { font-size: 20pt; }
h2 { font-size: 15pt; }
h3 { font-size: 12pt; }
code, pre { font-family: "Courier New", Courier, monospace; font-size: 9.5pt; }
pre { background-color: #f4f4f4; padding: 8pt; }
code { background-color: #f4f4f4; }
table { border-collapse: collapse; width: 100%%; }
th, td { border: 1px solid #999999; padding: 4pt; text-align: left; }
th { background-color: #eeeeee; }
blockquote { margin-left: 1em; padding-left: 1em; border-left: 2px solid #cccccc; color: #444444; }
"""

# (regular, bold, italic, bold-italic) candidates; the built-in PDF fonts lack characters
# such as the rupee sign, so a real TrueType family is registered when one is installed.
_FONT_FAMILIES = (
    ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"),
)
_FONT_DIRS = (
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
)


# --------------------------------------------------------------------- paths
def _documents_folder() -> Path:
    """The user's Documents folder (may be OneDrive-redirected). Tests patch this."""
    try:
        return known_folder("Documents")
    except Exception:  # best effort: fall back to the conventional location
        return Path.home() / "Documents"


def _is_bare(raw: str) -> bool:
    text = (raw or "").strip().strip("\"'")
    if not text or text.startswith(("~", "%", "$")):
        return False
    return not any(sep in text for sep in ("/", "\\", ":"))


def _new_path(raw: str) -> Path:
    """Resolve where a NEW file goes: bare names land in Documents, anything else is resolved."""
    if _is_bare(raw):
        return Path(os.path.abspath(_documents_folder() / raw.strip().strip("\"'")))
    return resolve_path(raw)


def _exists_fail(path: Path) -> ToolResult:
    return ToolResult.fail(
        f"{path} already exists; nothing was written",
        f"Existing files are never overwritten. Choose another name, e.g. {path.stem} - 2{path.suffix}.",
    )


def _write_new(path: Path, data: bytes) -> None:
    """Write bytes to a path that must not exist (raises FileExistsError if it does)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        fh.write(data)


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove partial file %s", path)


# --------------------------------------------------------------- page ranges
def parse_pages(spec: str, total: int) -> list[int]:
    """Zero-based page indexes from "1-3,7" (1-based, inclusive; "5-" means 5 to the end).

    Order is kept as given. Raises ValueError with a message the model can act on.
    """
    if total < 1:
        raise ValueError("the document has no pages")
    if not spec or not spec.strip():
        raise ValueError("page range is empty; use something like '1-3,7'")
    out: list[int] = []
    for token in spec.replace(" ", "").split(","):
        m = re.fullmatch(r"(\d+)(?:(-)(\d*))?", token)
        if not m:
            raise ValueError(f"'{token}' is not a page or range; use something like '1-3,7'")
        first = int(m.group(1))
        last = first if not m.group(2) else (int(m.group(3)) if m.group(3) else total)
        if first < 1 or last < 1:
            raise ValueError("pages start at 1")
        if last < first:
            raise ValueError(f"range '{token}' runs backwards")
        if last > total:
            raise ValueError(f"page {last} is past the end; the document has {total} pages")
        out.extend(range(first - 1, last))
    return out


# ---------------------------------------------------------------- PDF helpers
def _open_pdf(raw: str) -> tuple[Path, Any] | ToolResult:
    """Resolve and open a readable PDF, or return the failure to hand back to the model."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        path = resolve_path(raw)
    except ValueError as exc:
        return _bad_path(exc)
    if not path.exists():
        return _missing(path, "file")
    if path.is_dir():
        return ToolResult.fail(f"{path} is a folder", "Give the path of a PDF file.")
    if path.suffix.lower() != ".pdf":
        return ToolResult.fail(f"{path.name} is not a PDF", "Use read_document for other formats.")
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            return _encrypted(path)
        len(reader.pages)  # forces the page tree to parse so corruption surfaces here
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as exc:
        return ToolResult.fail(f"cannot read {path.name} as a PDF: {exc}", "The file may be damaged.")
    return path, reader


def _encrypted(path: Path) -> ToolResult:
    return ToolResult.fail(
        f"{path.name} is password-protected, so its contents cannot be read",
        "Ask the user for an unlocked copy; passwords are not handled by these tools.",
    )


def _pdf_target(raw: str) -> Path | ToolResult:
    """Resolve an output PDF path: must end in .pdf and must not exist."""
    try:
        target = _new_path(raw)
    except ValueError as exc:
        return _bad_path(exc)
    if target.suffix.lower() != ".pdf":
        return ToolResult.fail(f"output {target.name} must end in .pdf", "Add .pdf to the output name.")
    if target.exists():
        return _exists_fail(target)
    return target


def _save_pdf(writer: Any, target: Path, expected_pages: int) -> ToolResult | None:
    """Write a PDF to a new path, reopen it, and check the page count. None means success."""
    from pypdf import PdfReader

    buf = io.BytesIO()
    writer.write(buf)
    try:
        _write_new(target, buf.getvalue())
    except FileExistsError:
        return _exists_fail(target)
    except OSError as exc:
        return ToolResult.fail(f"cannot write {target}: {exc.strerror or exc}")
    try:
        check = PdfReader(str(target))
        found = len(check.pages)
    except Exception as exc:  # any reader failure means the output is bad
        _discard(target)
        return ToolResult.fail(f"the written PDF could not be reopened ({exc}); it was removed")
    if check.is_encrypted or found != expected_pages:
        _discard(target)
        return ToolResult.fail(f"expected {expected_pages} pages in the output but found {found}; it was removed")
    return None


def _pages_word(n: int) -> str:
    return f"{n} page{'s' if n != 1 else ''}"


def _pdf_text(reader: Any, indexes: list[int], stop_after: int) -> tuple[list[str], int]:
    """(per-page text blocks, pages that had text). Stops once ``stop_after`` chars exist."""
    blocks: list[str] = []
    with_text = 0
    size = 0
    for i in indexes:
        text = (reader.pages[i].extract_text() or "").strip()
        with_text += bool(text)
        block = f"--- Page {i + 1} ---\n{text or '(no text on this page)'}"
        blocks.append(block)
        size += len(block) + 2
        if size > stop_after:
            break
    return blocks, with_text


# --------------------------------------------------------------- other readers
def _docx_text(path: Path) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    lines: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if m := re.match(r"Heading (\d)", style):
                text = "#" * int(m.group(1)) + " " + text
            elif style.startswith("List Bullet"):
                text = "- " + text
            elif style.startswith("List Number"):
                text = "1. " + text
            lines.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                lines.append("| " + " | ".join(c.text.strip().replace("\n", " ") for c in row.cells) + " |")
    return "\n".join(lines)


_BLOCK_TAGS = ["p", "div", "li", "tr", "br", "section", "article", "ul", "ol", "table", "pre", "blockquote",
               "h1", "h2", "h3", "h4", "h5", "h6"]


def _html_text(raw_html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")
    for node in soup(["script", "style", "head", "noscript"]):
        node.decompose()
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        row.replace_with(soup.new_string("| " + " | ".join(cells) + " |\n"))
    for item in soup.find_all("li"):
        item.insert(0, "- ")
    for level in range(1, 7):
        for head in soup.find_all(f"h{level}"):
            head.insert(0, "#" * level + " ")
    for block in soup.find_all(_BLOCK_TAGS):
        block.append("\n")  # block elements become line breaks; inline ones stay in the line
    text = re.sub(r"[ \t]+\n", "\n", soup.get_text(""))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@tool(group=GROUP, timeout=60.0)
def read_document(path: str, pages: str | None = None, max_chars: int = 12000, offset: int = 0) -> ToolResult:
    """Read the text of a document: .pdf, .docx (headings, paragraphs, tables as rows), .md,
    .txt or .html. Use this for any PDF or Word file; read_file does not handle those.
    Returns the text, the page count for PDFs, and a note with the next offset when truncated.
    A scanned PDF with no text layer fails with advice to use the screen/OCR tools.

    Args:
        path: Document to read.
        pages: PDFs only: pages to read, like "1-3,7" or "5-" (1-based). Default: all.
        max_chars: Maximum characters to return (capped at 11000).
        offset: Character position to continue from when an earlier read was truncated.
    """
    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.exists():
        return _missing(target, "file")
    if target.is_dir():
        return ToolResult.fail(f"{target} is a folder", "Use list_directory to see its contents.")
    limit = max(200, min(max_chars, MAX_CHARS_CAP))
    start = max(0, offset)
    suffix = target.suffix.lower()
    header = f"{target}"
    blocks: list[str] = []
    indexes: list[int] = []
    try:
        if suffix == ".pdf":
            opened = _open_pdf(path)
            if isinstance(opened, ToolResult):
                return opened
            reader = opened[1]
            total = len(reader.pages)
            try:
                indexes = parse_pages(pages, total) if pages else list(range(total))
            except ValueError as exc:
                return ToolResult.fail(str(exc), f"The PDF has {total} pages.")
            blocks, with_text = _pdf_text(reader, indexes, stop_after=start + limit)
            if with_text == 0:
                return ToolResult.fail(
                    f"{target.name} ({_pages_word(total)}) has no text layer; it is probably scanned images",
                    "Open it and use the screen/OCR tools to read it.",
                )
            text = "\n\n".join(blocks)
            header += f" (PDF, {_pages_word(total)})"
        elif suffix == ".docx":
            text = _docx_text(target)
        elif suffix in HTML_SUFFIXES | TEXT_SUFFIXES:
            raw = target.read_bytes()
            if _is_binary(raw):
                return ToolResult.fail(f"{target.name} is binary, not text")
            text = _decode(raw)[0]
            if suffix in HTML_SUFFIXES:
                text = _html_text(text)
        elif suffix == ".doc":
            return ToolResult.fail(f"{target.name} is an old .doc file", "Ask the user to save it as .docx or PDF.")
        else:
            return ToolResult.fail(
                f"unsupported document type '{suffix or target.name}'",
                "read_document handles .pdf .docx .md .txt .html; use read_file for other text files.",
            )
    except ImportError as exc:
        return ToolResult.fail(f"a library needed for {suffix} is missing: {exc.name}")
    except Exception as exc:  # corrupt files raise many different library errors
        log.warning("read_document failed for %s: %s", target, exc)
        return ToolResult.fail(f"cannot read {target.name}: {type(exc).__name__}: {exc}", "The file may be damaged.")

    if not text.strip():
        return ToolResult.fail(f"{target.name} contains no text")
    chunk = text[start : start + limit]
    if not chunk:
        return ToolResult.fail(f"offset {start} is past the end ({len(text)} characters read from {target.name})")
    end = start + len(chunk)
    note = ""
    if end < len(text):
        note = f"\n[Truncated: showing characters {start}-{end} of {len(text)}. Call again with offset={end} for the rest.]"
    elif suffix == ".pdf" and len(blocks) < len(indexes):
        nxt = indexes[len(blocks)] + 1
        note = f'\n[Stopped at the size limit after page {nxt - 1}. Read on with pages="{nxt}-".]'
    return ToolResult.success(f"{header}\n{chunk}{note}", data=str(target))


# ------------------------------------------------------------ document writing
def _pdf_font() -> tuple[str, str, Path | None]:
    """(@font-face CSS, CSS family, font folder) for a TrueType family with wide Unicode coverage."""
    styles = (("normal", "normal"), ("bold", "normal"), ("normal", "italic"), ("bold", "italic"))
    for files in _FONT_FAMILIES:
        for folder in _FONT_DIRS:
            paths = [folder / f for f in files]
            if all(p.is_file() for p in paths):
                faces = "".join(
                    f'@font-face {{ font-family: JarvisBody; font-weight: {w}; font-style: {s}; '
                    f'src: url("{p.as_posix()}"); }}\n'
                    for p, (w, s) in zip(paths, styles)
                )
                return faces, "JarvisBody", folder
    log.warning("no Unicode TrueType font found; PDF text outside Latin-1 may not render")
    return "", "Helvetica, Arial, sans-serif", None


def _render_html(md_text: str, title: str, font: tuple[str, str, Path | None] | None = None) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    faces, family, _ = font or ("", 'Georgia, "Times New Roman", serif', None)
    return (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8" />\n'
        f"<title>{html_lib.escape(title)}</title>\n<style>{faces}{_CSS % {'body': family}}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _render_pdf(md_text: str, title: str) -> bytes:
    from xhtml2pdf import pisa

    font = _pdf_font()
    kwargs: dict[str, Any] = {}
    if font[2] is not None:
        try:  # xhtml2pdf confines local reads; let it read the system font folder only
            from xhtml2pdf.config.resources import ResourceAccessPolicy

            kwargs["resource_policy"] = ResourceAccessPolicy(
                base_dir=Path.cwd(), extra_roots=(font[2],)
            )
        except ImportError:
            log.warning("this xhtml2pdf has no resource policy; relying on its defaults")
    out = io.BytesIO()
    status = pisa.CreatePDF(_render_html(md_text, title, font), dest=out, encoding="utf-8", **kwargs)
    if status.err:
        raise OSError(f"xhtml2pdf reported {status.err} error(s)")
    return out.getvalue()


_INLINE = re.compile(r"\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`")


def _add_inline(paragraph: Any, text: str) -> None:
    """Add runs for **bold**, *italic*, ***both*** and `code` spans."""
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos : m.start()])
        both, bold, italic, code = m.groups()
        if both is not None:
            run = paragraph.add_run(both)
            run.bold = run.italic = True
        elif bold is not None:
            paragraph.add_run(bold).bold = True
        elif italic is not None:
            paragraph.add_run(italic).italic = True
        else:
            paragraph.add_run(code).font.name = "Courier New"
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _list_style(base: str, indent: str) -> str:
    depth = min(len(indent.replace("\t", "  ")) // 2, 2)
    return base if depth == 0 else f"{base} {depth + 1}"


def _build_docx(md_text: str) -> Any:
    """Map Markdown blocks onto Word styles: headings, lists, tables, quotes, code."""
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    lines = md_text.splitlines()
    para_buf: list[str] = []

    def flush() -> None:
        if para_buf:
            _add_inline(doc.add_paragraph(), " ".join(s.strip() for s in para_buf))
            para_buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            for n, code_line in enumerate(code):
                if n:
                    p.add_run().add_break()
                run = p.add_run(code_line)
                run.font.name, run.font.size = "Courier New", Pt(9)
        elif not stripped:
            flush()
        elif m := re.match(r"(#{1,6})\s+(.*?)\s*#*$", stripped):
            flush()
            _add_inline(doc.add_paragraph(style=f"Heading {len(m.group(1))}"), m.group(2))
        elif stripped.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush()
            rows = [_split_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            width = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=width)
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c in range(width):
                    para = table.cell(r, c).paragraphs[0]
                    _add_inline(para, row[c] if c < len(row) else "")
                    if r == 0:
                        for run in para.runs:
                            run.bold = True
            continue  # i already points past the table
        elif m := re.match(r"(\s*)[-*+]\s+(.*)", line):
            flush()
            _add_inline(doc.add_paragraph(style=_list_style("List Bullet", m.group(1))), m.group(2))
        elif m := re.match(r"(\s*)\d+[.)]\s+(.*)", line):
            flush()
            _add_inline(doc.add_paragraph(style=_list_style("List Number", m.group(1))), m.group(2))
        elif stripped.startswith(">"):
            flush()
            _add_inline(doc.add_paragraph(style="Quote"), stripped.lstrip("> ").strip())
        elif re.fullmatch(r"[-*_]{3,}", stripped):
            flush()
        else:
            para_buf.append(line)
        i += 1
    flush()
    return doc


def _final_target(raw: str, fmt_arg: str) -> tuple[Path, str] | ToolResult:
    """Pick the output path and format. The extension decides for "auto"."""
    try:
        target = _new_path(raw)
    except ValueError as exc:
        return _bad_path(exc)
    suffix = target.suffix.lower()
    if fmt_arg == "auto":
        if suffix in TEXT_SUFFIXES or not suffix:
            fmt = "md"
        elif suffix in HTML_SUFFIXES:
            fmt = "html"
        elif suffix in (".pdf", ".docx"):
            fmt = suffix[1:]
        else:
            return ToolResult.fail(
                f"cannot tell the format from '{suffix}'",
                "Use a .md, .html, .pdf or .docx name, or set format explicitly.",
            )
    else:
        fmt = fmt_arg
    wanted = FORMAT_SUFFIX[fmt]
    # .txt/.markdown/.htm are accepted spellings of md/html, so they are left as the user named them.
    spelled_ok = (fmt == "md" and suffix in TEXT_SUFFIXES) or (fmt == "html" and suffix in HTML_SUFFIXES)
    if not spelled_ok and suffix != wanted and (suffix in DOC_SUFFIXES or not suffix):
        target = target.with_name((target.stem if suffix else target.name) + wanted)
    return target, fmt


def _verify_written(path: Path, fmt: str) -> tuple[str | None, str]:
    """(problem or None, size detail for the report) after reading the file back."""
    try:
        size = path.stat().st_size
        if size <= 0:
            return "the file is empty", ""
        detail = _fmt_size(size)
        if fmt == "pdf":
            from pypdf import PdfReader

            pages = len(PdfReader(str(path)).pages)
            if pages < 1:
                return "the PDF has no pages", ""
            detail += f", {_pages_word(pages)}"
        elif fmt == "docx":
            if not _docx_text(path).strip():
                return "the Word file has no text", ""
        elif not path.read_text(encoding="utf-8").strip():
            return "the file has no text", ""
        return None, detail
    except Exception as exc:  # reading back can fail in many library-specific ways
        return f"it could not be reopened ({exc})", ""


@tool(group=GROUP, timeout=90.0)
def create_document(
    path: str,
    content_markdown: str,
    format: Literal["auto", "md", "html", "pdf", "docx"] = "auto",
) -> ToolResult:
    """Create a NEW document from Markdown: a report, letter, notes, study plan, or handout.
    Write the content as Markdown (headings, lists, tables, **bold**, code blocks) and this
    renders it as .md, .html, .pdf or .docx with a clean layout. Never overwrites: fails if the
    file exists. Reads the result back to verify it and returns the path, size and PDF page count.

    Args:
        path: Where to save it. A bare name like "plan.pdf" goes to the Documents folder; a full path is used as given. The extension picks the format when format is "auto".
        content_markdown: The full document content, in Markdown.
        format: Output format; "auto" uses the file extension.
    """
    if not content_markdown or not content_markdown.strip():
        return ToolResult.fail("content_markdown is empty", "Write the document content first.")
    chosen = _final_target(path, format)
    if isinstance(chosen, ToolResult):
        return chosen
    target, fmt = chosen
    if target.is_dir():
        return ToolResult.fail(f"{target} is a folder", "Give a file name as well.")
    if target.exists():
        return _exists_fail(target)
    try:
        if fmt == "md":
            data = content_markdown.encode("utf-8")
        elif fmt == "html":
            data = _render_html(content_markdown, target.stem).encode("utf-8")
        elif fmt == "pdf":
            data = _render_pdf(content_markdown, target.stem)
        else:
            buf = io.BytesIO()
            _build_docx(content_markdown).save(buf)
            data = buf.getvalue()
        _write_new(target, data)
    except FileExistsError:
        return _exists_fail(target)
    except ImportError as exc:
        return ToolResult.fail(f"a library needed for {fmt} is missing: {exc.name}")
    except OSError as exc:
        _discard(target)
        return ToolResult.fail(f"cannot write {target}: {exc.strerror or exc}")
    except Exception as exc:  # a renderer bug must not look like success
        log.exception("rendering %s failed", target)
        _discard(target)
        return ToolResult.fail(f"could not render {target.name}: {type(exc).__name__}: {exc}; nothing was saved")
    problem, detail = _verify_written(target, fmt)
    if problem:
        _discard(target)
        return ToolResult.fail(f"{target.name} was written but failed verification ({problem}); it was removed")
    return ToolResult.success(f"Created {target} ({fmt}, {detail}).", data=str(target))


# ------------------------------------------------------------------ PDF tools
def _pdf_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return text[1:] if text.startswith("/") else text


@tool(group=GROUP)
def pdf_info(path: str) -> ToolResult:
    """Inspect a PDF without reading its text: page count, title/author metadata, whether it
    is encrypted, and any fillable form fields with their current values. Use this before
    pdf_fill_form to learn the exact field names.

    Args:
        path: PDF file to inspect.
    """
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        target = resolve_path(path)
    except ValueError as exc:
        return _bad_path(exc)
    if not target.exists():
        return _missing(target, "file")
    if target.suffix.lower() != ".pdf" or target.is_dir():
        return ToolResult.fail(f"{target.name} is not a PDF file")
    try:
        reader = PdfReader(str(target))
        if reader.is_encrypted and not reader.decrypt(""):
            return ToolResult.success(
                f"{target.name}: encrypted (password-protected); pages, metadata and fields "
                "cannot be read without the password.",
                data={"encrypted": True},
            )
        enc = "yes (opens without a password)" if reader.is_encrypted else "no"
        lines = [f"{target.name}: {_pages_word(len(reader.pages))}, {_fmt_size(target.stat().st_size)}, encrypted: {enc}"]
        meta = reader.metadata
        for key, label in (("/Title", "title"), ("/Author", "author"), ("/Subject", "subject"), ("/CreationDate", "created")):
            if meta and meta.get(key):
                lines.append(f"{label}: {meta[key]}")
        fields = reader.get_fields() or {}
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as exc:
        return ToolResult.fail(f"cannot read {target.name} as a PDF: {exc}", "The file may be damaged.")
    if fields:
        lines.append(f"Form fields ({len(fields)}):")
        for name, field in list(fields.items())[:MAX_LISTED_FIELDS]:
            kind = str(field.get("/FT") or "unknown").lstrip("/")
            value = field.get("/V")
            lines.append(f"  {name} [{kind}] = {_pdf_value(value) if value is not None else '(empty)'}")
        if len(fields) > MAX_LISTED_FIELDS:
            lines.append(f"  ... {len(fields) - MAX_LISTED_FIELDS} more not shown")
    else:
        lines.append("No fillable form fields.")
    return ToolResult.success(
        "\n".join(lines), data={"pages": len(reader.pages), "fields": list(fields), "encrypted": False}
    )


@tool(group=GROUP, timeout=60.0)
def pdf_merge(sources: list[str], output: str) -> ToolResult:
    """Combine several PDFs into one NEW PDF, in the order given. The originals are not
    changed and an existing output file is never overwritten.

    Args:
        sources: PDF paths in the order they should appear (at least two).
        output: Path for the merged PDF (must end in .pdf). A bare name goes to Documents.
    """
    from pypdf import PdfWriter

    if len(sources) < 2:
        return ToolResult.fail("need at least two PDFs to merge", "Give two or more paths in sources.")
    target = _pdf_target(output)
    if isinstance(target, ToolResult):
        return target
    writer, total = PdfWriter(), 0
    for raw in sources:
        opened = _open_pdf(raw)
        if isinstance(opened, ToolResult):
            return opened
        for page in opened[1].pages:
            writer.add_page(page)
            total += 1
    failure = _save_pdf(writer, target, total)
    if failure:
        return failure
    return ToolResult.success(f"Merged {len(sources)} PDFs into {target} ({_pages_word(total)}).", data=str(target))


@tool(group=GROUP, timeout=60.0)
def pdf_split(path: str, ranges: list[str], output_folder: str | None = None) -> ToolResult:
    """Split a PDF into several NEW PDFs, one per page range. Each range such as "1-3" or
    "7" becomes its own file named "<name> - pages 1-3.pdf". The original is unchanged and
    nothing is overwritten (if any output name is taken, nothing is written). To keep just
    some pages in a single file use pdf_extract_pages instead.

    Args:
        path: PDF to split.
        ranges: Page ranges, one file each, e.g. ["1-3", "4-10", "11"].
        output_folder: Folder for the new files. Default: the folder of the original.
    """
    from pypdf import PdfWriter

    opened = _open_pdf(path)
    if isinstance(opened, ToolResult):
        return opened
    source, reader = opened
    if not ranges:
        return ToolResult.fail("ranges is empty", 'Give ranges like ["1-3", "4-6"].')
    try:
        folder = resolve_path(output_folder) if output_folder else source.parent
        plan = [(parse_pages(r, len(reader.pages)), r) for r in ranges]
    except ValueError as exc:
        return ToolResult.fail(str(exc), f"The PDF has {len(reader.pages)} pages. Use ranges like '1-3'.")
    if folder.exists() and not folder.is_dir():
        return ToolResult.fail(f"{folder} is a file, not a folder")
    jobs: list[tuple[Path, list[int]]] = []
    for indexes, label in plan:
        clean = label.replace(" ", "").replace(",", "_")
        target = folder / f"{source.stem} - pages {clean}.pdf"
        if target.exists() or any(target == t for t, _ in jobs):
            return _exists_fail(target)
        jobs.append((target, indexes))
    made: list[str] = []
    for target, indexes in jobs:
        writer = PdfWriter()
        for i in indexes:
            writer.add_page(reader.pages[i])
        failure = _save_pdf(writer, target, len(indexes))
        if failure:
            done = f" Already created: {', '.join(made)}." if made else ""
            return ToolResult.fail(failure.content.removeprefix("Error: ") + done)
        made.append(str(target))
    return ToolResult.success(f"Split {source.name} into {len(made)} files:\n" + "\n".join(made), data=made)


@tool(group=GROUP, timeout=60.0)
def pdf_extract_pages(path: str, pages: str, output: str) -> ToolResult:
    """Copy selected pages of a PDF into one NEW PDF (e.g. pages "2,5-6", kept in that order).
    The original is unchanged and an existing output file is never overwritten.

    Args:
        path: Source PDF.
        pages: Pages to keep, like "1-3,7" (1-based).
        output: Path for the new PDF (must end in .pdf). A bare name goes to Documents.
    """
    from pypdf import PdfWriter

    opened = _open_pdf(path)
    if isinstance(opened, ToolResult):
        return opened
    reader = opened[1]
    try:
        indexes = parse_pages(pages, len(reader.pages))
    except ValueError as exc:
        return ToolResult.fail(str(exc), f"The PDF has {len(reader.pages)} pages.")
    target = _pdf_target(output)
    if isinstance(target, ToolResult):
        return target
    writer = PdfWriter()
    for i in indexes:
        writer.add_page(reader.pages[i])
    failure = _save_pdf(writer, target, len(indexes))
    if failure:
        return failure
    return ToolResult.success(f"Extracted {_pages_word(len(indexes))} into {target}.", data=str(target))


@tool(group=GROUP, timeout=60.0)
def pdf_rotate(
    path: str,
    degrees: Literal[90, 180, 270],
    pages: str | None = None,
    output: str | None = None,
) -> ToolResult:
    """Rotate pages of a PDF clockwise and save the result as a NEW PDF. The original is
    unchanged and nothing is overwritten.

    Args:
        path: Source PDF.
        degrees: Clockwise rotation: 90, 180 or 270.
        pages: Pages to rotate like "1-3,7". Default: every page.
        output: Path for the new PDF. Default: "<name> - rotated.pdf" beside the original.
    """
    from pypdf import PdfWriter

    opened = _open_pdf(path)
    if isinstance(opened, ToolResult):
        return opened
    source, reader = opened
    try:
        chosen = set(parse_pages(pages, len(reader.pages))) if pages else set(range(len(reader.pages)))
    except ValueError as exc:
        return ToolResult.fail(str(exc), f"The PDF has {len(reader.pages)} pages.")
    target = _pdf_target(output or str(source.with_name(f"{source.stem} - rotated.pdf")))
    if isinstance(target, ToolResult):
        return target
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    for i, page in enumerate(writer.pages):
        if i in chosen:
            page.rotate(degrees)
    failure = _save_pdf(writer, target, len(reader.pages))
    if failure:
        return failure
    return ToolResult.success(
        f"Rotated {_pages_word(len(chosen))} by {degrees} degrees clockwise into {target}.", data=str(target)
    )


@tool(group=GROUP, timeout=60.0)
def pdf_fill_form(path: str, values: dict[str, str], output: str) -> ToolResult:
    """Fill the fillable form fields of a PDF and save a NEW filled copy. Use pdf_info first
    to get the exact field names. Fails listing any name that does not exist, and verifies each
    value was stored. The original is unchanged and nothing is overwritten.

    Args:
        path: Source PDF containing form fields.
        values: Map of field name to the text to put in it.
        output: Path for the filled PDF (must end in .pdf). A bare name goes to Documents.
    """
    from pypdf import PdfReader, PdfWriter

    if not values:
        return ToolResult.fail("values is empty", 'Give field values like {"Name": "Asha"}.')
    opened = _open_pdf(path)
    if isinstance(opened, ToolResult):
        return opened
    source, reader = opened
    target = _pdf_target(output)
    if isinstance(target, ToolResult):
        return target
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    fields = writer.get_fields() or {}
    if set(values) - set(fields) and hasattr(writer, "reattach_fields"):
        writer.reattach_fields()
        fields = writer.get_fields() or {}
    if missing := sorted(set(values) - set(fields)):
        known = ", ".join(list(fields)[:30]) or "none; the PDF has no form fields"
        return ToolResult.fail(f"form fields not found: {', '.join(missing)}", f"Fields in {source.name}: {known}")
    try:
        writer.update_page_form_field_values(None, dict(values), auto_regenerate=False)
        failure = _save_pdf(writer, target, len(reader.pages))
        if failure:
            return failure
        saved = PdfReader(str(target)).get_fields() or {}
        for name, expected in values.items():
            if _pdf_value(saved.get(name, {}).get("/V")) != _pdf_value(expected):
                raise ValueError(f"field {name} did not keep its new value")
    except (ValueError, OSError) as exc:
        _discard(target)
        return ToolResult.fail(f"could not fill {source.name}: {exc}; nothing was saved")
    return ToolResult.success(f"Filled {len(values)} field(s) into {target}.", data=str(target))
