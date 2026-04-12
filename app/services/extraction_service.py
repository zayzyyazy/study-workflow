"""Extract plain text from common lecture file types. Never raises to callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractionResult:
    ok: bool
    text: str
    message: str


def _read_txt_or_md(path: Path) -> ExtractionResult:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return ExtractionResult(True, text, "Extracted as UTF-8 text.")
    except OSError as e:
        return ExtractionResult(False, "", f"Could not read file: {e}")


def _read_pdf(path: Path) -> ExtractionResult:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ExtractionResult(
            False,
            "",
            "PDF support requires the 'pypdf' package (install dependencies).",
        )
    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    parts.append(t)
            except Exception as page_err:  # noqa: BLE001
                parts.append(f"\n[Page extract error: {page_err}]\n")
        text = "\n".join(parts).strip()
        if not text:
            return ExtractionResult(
                False,
                "",
                "PDF contained no extractable text (scanned PDFs need OCR, not enabled).",
            )
        return ExtractionResult(True, text, "Extracted text from PDF.")
    except Exception as e:  # noqa: BLE001
        return ExtractionResult(False, "", f"PDF extraction failed: {e}")


def _read_docx(path: Path) -> ExtractionResult:
    try:
        import docx  # python-docx
    except ImportError:
        return ExtractionResult(
            False,
            "",
            "DOCX support requires the 'python-docx' package (install dependencies).",
        )
    try:
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text]
        text = "\n".join(parts).strip()
        if not text:
            return ExtractionResult(False, "", "DOCX had no paragraph text.")
        return ExtractionResult(True, text, "Extracted text from DOCX.")
    except Exception as e:  # noqa: BLE001
        return ExtractionResult(False, "", f"DOCX extraction failed: {e}")


def extract_text_from_file(path: Path) -> ExtractionResult:
    """
    Route by extension. Unknown types return a clear message without crashing.
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return _read_txt_or_md(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".docx"}:
        return _read_docx(path)
    if suffix == ".doc":
        return ExtractionResult(
            False,
            "",
            "Legacy .doc format is not supported; save as .docx or PDF and re-upload.",
        )
    return ExtractionResult(
        False,
        "",
        f"No extractor for '{suffix}' yet. Use .txt, .md, .pdf, or .docx.",
    )
