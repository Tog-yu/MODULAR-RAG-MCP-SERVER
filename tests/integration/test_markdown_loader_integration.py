"""Integration tests for the Markdown loader using a real fixture file.

Verifies the Markdown loader end-to-end against a committed fixture
(``tests/fixtures/sample_documents/sample.md``) that exercises frontmatter,
headings, code blocks, local image extraction, reference-style images, and
remote image preservation.
"""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.markdown_loader import MarkdownLoader


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SAMPLE_MD = FIXTURES_DIR / "sample.md"


class TestMarkdownLoaderWithRealFile:
    def test_fixture_exists(self):
        assert SAMPLE_MD.exists(), f"Test fixture not found: {SAMPLE_MD}"

    def test_load_sample_document(self):
        loader = MarkdownLoader()
        doc = loader.load(SAMPLE_MD)

        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        assert len(doc.text) > 0

        # Required metadata.
        assert doc.metadata["source_path"] == str(SAMPLE_MD)
        assert doc.metadata["doc_type"] == "markdown"
        assert "doc_hash" in doc.metadata

        # Title comes from frontmatter.
        assert doc.metadata["title"] == "Sample Markdown Document"
        assert doc.metadata.get("frontmatter", {}).get("author") == "WorkBuddy"

        # Heading outline captured.
        outline = doc.metadata["heading_outline"]
        levels = [h["level"] for h in outline]
        assert 1 in levels and 2 in levels and 3 in levels

        # Frontmatter delimiters stripped from body.
        assert doc.text.strip().startswith("# Sample Markdown Document")
        assert "---" not in doc.text

    def test_sample_document_code_block_preserved(self):
        loader = MarkdownLoader()
        doc = loader.load(SAMPLE_MD)
        assert "def greet(name: str)" in doc.text
        assert "```python" in doc.text

    def test_sample_document_local_image_extracted(self):
        loader = MarkdownLoader(extract_images=True)
        doc = loader.load(SAMPLE_MD)
        images = doc.metadata.get("images", [])
        # Two local images: inline + reference-style (deduplicated to same file -> 1).
        assert len(images) == 1
        img = images[0]
        assert Path(img["path"]).exists()
        assert f"[IMAGE: {img['id']}]" in doc.text
        # Remote image reference preserved verbatim.
        assert "https://example.com/banner.png" in doc.text

    def test_sample_document_image_extraction_disabled(self):
        loader = MarkdownLoader(extract_images=False)
        doc = loader.load(SAMPLE_MD)
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
        # Original references retained when extraction disabled.
        assert "![Sample diagram](sample_image.png)" in doc.text

    def test_factory_dispatch_picks_markdown(self):
        # The pipeline relies on LoaderFactory; verify it resolves .md -> MarkdownLoader.
        loader = LoaderFactory.get_loader(str(SAMPLE_MD))
        assert isinstance(loader, MarkdownLoader)
