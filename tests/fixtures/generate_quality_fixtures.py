"""Generate low-quality PDF fixtures for the quality gate tests (C2.5).

Creates three PDFs next to this script (pypdfium2-based, no PyMuPDF needed):

- low_quality.pdf : pages filled with garbled characters (invalid symbols,
  punctuation) -> fails valid_char_ratio >= 0.80.
- scan_like.pdf   : pages with (almost) no text layer, like an OCR-less scan
  -> fails text_density >= 0.50.
- healthy.pdf     : normal ASCII text that must pass all thresholds (sanity).

Note: pdfium standard fonts are ASCII-only, so garbled content uses Latin-1
supplement symbols (still "invalid" per the CJK+alnum rule).
"""

from pathlib import Path

import ctypes

import pypdfium2 as pdfium
import pypdfium2._helpers.pageobjects as po
import pypdfium2.raw as pdfium_c

HERE = Path(__file__).parent

# Latin-1 supplement symbols: non-CJK, non-alphanumeric -> "invalid" chars.
_GARBLED = "«»§¶†‡°±µ·¸¹º»¼½¾¿×÷" * 40


def _add_text_page(doc: "pdfium.PdfDocument", text: str, y: float = 700.0) -> None:
    """Append a page containing a single text object at the top-left."""
    page = doc.new_page(600, 800)
    font = pdfium_c.FPDFText_LoadStandardFont(doc.raw, b"Helvetica")
    raw_obj = pdfium_c.FPDFPageObj_CreateTextObj(doc.raw, font, 10.0)
    buf = (ctypes.c_ushort * (len(text) + 1))(*[ord(c) for c in text] + [0])
    pdfium_c.FPDFText_SetText(raw_obj, buf)
    pdfium_c.FPDFPageObj_Transform(raw_obj, 1.0, 0.0, 0.0, 1.0, 40.0, y)
    page.insert_obj(po.PdfTextObj(raw_obj, pdf=doc))
    page.gen_content()


def build_low_quality(path: Path, pages: int = 4) -> None:
    """Garbled text on every page: passes nothing else, fails ratio."""
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        _add_text_page(doc, _GARBLED)
    doc.save(str(path))
    doc.close()


def build_scan_like(path: Path, pages: int = 4) -> None:
    """Empty pages (no text objects): like an OCR-less scan, fails density."""
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(600, 800)
    doc.save(str(path))
    doc.close()


def build_healthy(path: Path, pages: int = 3) -> None:
    """Normal ASCII text: must pass all thresholds (sanity fixture)."""
    body = (
        "Retrieval-Augmented Generation combines vector search with LLM "
        "generation. Chunking quality and metadata enrichment both matter. "
    )
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        _add_text_page(doc, body * 8)
    doc.save(str(path))
    doc.close()


if __name__ == "__main__":
    build_low_quality(HERE / "low_quality.pdf")
    build_scan_like(HERE / "scan_like.pdf")
    build_healthy(HERE / "healthy.pdf")
    print("fixtures written to", HERE)
