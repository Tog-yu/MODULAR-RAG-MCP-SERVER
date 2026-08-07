"""Unit tests for Markdown Loader contract and behavior.

Tests verify:
- BaseLoader interface compliance
- MarkdownLoader initialization and configuration
- Input validation (extension / existence)
- Helper methods (hash, title, heading outline, frontmatter)
- Core Markdown loading: text extraction, metadata, frontmatter, structure
- Optional local image extraction reusing the multimodal pipeline
- Graceful degradation for remote / missing images and disabled extraction

Note: Additional integration coverage lives in
tests/integration/test_markdown_loader_integration.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.markdown_loader import MarkdownLoader


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_png(path: Path) -> None:
    """Write a tiny valid PNG so image copy can be exercised."""
    # 1x1 transparent PNG.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000001e221bc330000000049454e44"
        "ae426082"
    )
    path.write_bytes(png)


def _make_markdown(tmp_path: Path, body: str, image_name: str | None = None) -> Path:
    """Create a markdown file (optionally referencing a local image)."""
    md = tmp_path / "doc.md"
    content = body
    if image_name is not None:
        _make_png(tmp_path / image_name)
    md.write_text(content, encoding="utf-8")
    return md


# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------
class TestMarkdownLoaderInitialization:
    def test_default_initialization(self):
        loader = MarkdownLoader()
        assert loader.extract_images is True
        assert loader.image_storage_dir == Path("data/images")

    def test_custom_initialization(self):
        loader = MarkdownLoader(
            extract_images=False,
            image_storage_dir="custom/path",
            extract_markdown_images=False,
        )
        assert loader.extract_images is False
        assert loader.image_storage_dir == Path("custom/path")

    def test_extract_markdown_images_takes_precedence(self):
        loader = MarkdownLoader(extract_images=True, extract_markdown_images=False)
        assert loader.extract_images is False
        loader2 = MarkdownLoader(extract_images=False, extract_markdown_images=True)
        assert loader2.extract_images is True


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
class TestMarkdownLoaderValidation:
    def test_load_rejects_non_markdown(self, tmp_path):
        txt = tmp_path / "note.txt"
        txt.write_text("not markdown")
        loader = MarkdownLoader()
        with pytest.raises(ValueError, match="not a Markdown"):
            loader.load(txt)

    def test_load_nonexistent_file(self):
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.md")


# --------------------------------------------------------------------------
# Helper methods
# --------------------------------------------------------------------------
class TestMarkdownLoaderHelpers:
    def test_compute_file_hash_consistency(self, tmp_path):
        f = tmp_path / "a.md"
        f.write_bytes(b"consistent content")
        loader = MarkdownLoader()
        assert loader._compute_file_hash(f) == loader._compute_file_hash(f)
        assert len(loader._compute_file_hash(f)) == 64

    def test_compute_file_hash_differs(self, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_bytes(b"one")
        f2.write_bytes(b"two")
        loader = MarkdownLoader()
        assert loader._compute_file_hash(f1) != loader._compute_file_hash(f2)

    def test_extract_title_from_heading(self):
        loader = MarkdownLoader()
        assert loader._extract_title("# Hello World\n\nbody") == "Hello World"

    def test_extract_title_fallback_first_line(self):
        loader = MarkdownLoader()
        assert loader._extract_title("Just a line\n\nbody") == "Just a line"

    def test_extract_title_empty(self):
        loader = MarkdownLoader()
        assert loader._extract_title("") is None

    def test_extract_heading_outline(self):
        loader = MarkdownLoader()
        text = "# Top\n\n## Sub\n\n### Deep\n\npara"
        outline = loader._extract_heading_outline(text)
        assert outline == [
            {"level": 1, "text": "Top"},
            {"level": 2, "text": "Sub"},
            {"level": 3, "text": "Deep"},
        ]

    def test_parse_frontmatter(self):
        loader = MarkdownLoader()
        body, fm = loader._parse_frontmatter(
            "---\ntitle: My Doc\ntags:\n  - a\n  - b\n---\n# Heading\n\nbody"
        )
        assert fm == {"title": "My Doc", "tags": ["a", "b"]}
        assert body.startswith("# Heading")

    def test_parse_frontmatter_absent(self):
        loader = MarkdownLoader()
        body, fm = loader._parse_frontmatter("# No frontmatter\n\nbody")
        assert fm == {}
        assert body.startswith("# No frontmatter")

    def test_generate_image_id_format(self):
        image_id = MarkdownLoader._generate_image_id("abc123def456", 3)
        assert image_id == "abc123de_3"


# --------------------------------------------------------------------------
# Core loading
# --------------------------------------------------------------------------
class TestMarkdownLoadCore:
    def test_load_basic_structure(self, tmp_path):
        md = _make_markdown(tmp_path, "# Title\n\nSome **bold** text and `code`.\n")
        doc = MarkdownLoader().load(md)
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        assert len(doc.text) > 0
        assert doc.metadata["source_path"] == str(md)
        assert doc.metadata["doc_type"] == "markdown"
        assert "doc_hash" in doc.metadata
        # Frontmatter stripped from body text.
        assert "---" not in doc.text

    def test_load_title_from_frontmatter(self, tmp_path):
        md = _make_markdown(
            tmp_path, "---\ntitle: Front Title\n---\n# Real Heading\n\nbody"
        )
        doc = MarkdownLoader().load(md)
        assert doc.metadata["title"] == "Front Title"

    def test_load_title_from_heading(self, tmp_path):
        md = _make_markdown(tmp_path, "# Heading Title\n\nbody")
        doc = MarkdownLoader().load(md)
        assert doc.metadata["title"] == "Heading Title"

    def test_load_title_fallback_to_stem(self, tmp_path):
        # Empty body -> no heading/no first line -> fall back to filename stem.
        md = tmp_path / "my-doc.md"
        md.write_text("", encoding="utf-8")
        doc = MarkdownLoader().load(md)
        assert doc.metadata["title"] == "my-doc"

    def test_load_title_from_first_line(self, tmp_path):
        # No heading: first non-empty line is used as title (matches PdfLoader).
        md = _make_markdown(tmp_path, "Opening sentence without heading\n\nbody")
        doc = MarkdownLoader().load(md)
        assert doc.metadata["title"] == "Opening sentence without heading"

    def test_load_heading_outline_present(self, tmp_path):
        md = _make_markdown(tmp_path, "# A\n\n## B\n\ntext")
        doc = MarkdownLoader().load(md)
        assert doc.metadata["heading_outline"] == [
            {"level": 1, "text": "A"},
            {"level": 2, "text": "B"},
        ]

    def test_load_word_and_line_counts(self, tmp_path):
        md = _make_markdown(tmp_path, "# A\n\nword1 word2 word3")
        doc = MarkdownLoader().load(md)
        assert doc.metadata["word_count"] >= 3
        assert doc.metadata["line_count"] >= 1

    def test_load_preserves_code_block(self, tmp_path):
        body = "# T\n\n```python\nprint('hi')\n```\n"
        md = _make_markdown(tmp_path, body)
        doc = MarkdownLoader().load(md)
        assert "print('hi')" in doc.text
        assert "```python" in doc.text

    def test_document_serialization_roundtrip(self, tmp_path):
        md = _make_markdown(tmp_path, "# T\n\nbody text")
        doc = MarkdownLoader().load(md)
        doc_dict = doc.to_dict()
        assert "source_path" in doc_dict["metadata"]
        assert doc_dict["metadata"]["doc_type"] == "markdown"
        recreated = Document.from_dict(doc_dict)
        assert recreated.id == doc.id
        assert recreated.text == doc.text
        assert recreated.metadata == doc.metadata

    def test_document_hash_idempotent(self, tmp_path):
        md = _make_markdown(tmp_path, "# T\n\nbody")
        loader = MarkdownLoader()
        assert loader.load(md).id == loader.load(md).id


# --------------------------------------------------------------------------
# Image extraction (multimodal pipeline)
# --------------------------------------------------------------------------
class TestMarkdownImageExtraction:
    def test_local_image_extracted_and_replaced(self, tmp_path):
        md = _make_markdown(
            tmp_path, "# T\n\n![alt text](pic.png)\n\nmore", image_name="pic.png"
        )
        doc = MarkdownLoader(extract_images=True).load(md)
        images = doc.metadata.get("images", [])
        assert len(images) == 1
        img = images[0]
        assert img["id"].startswith(doc.metadata["doc_hash"][:8] + "_")
        assert Path(img["path"]).exists()
        assert f"[IMAGE: {img['id']}]" in doc.text
        # Original ![]() reference should be gone.
        assert "![alt text](pic.png)" not in doc.text

    def test_reference_style_image_extracted(self, tmp_path):
        body = "# T\n\n![alt][pic]\n\n[pic]: pic.png\n"
        md = _make_markdown(tmp_path, body, image_name="pic.png")
        doc = MarkdownLoader().load(md)
        images = doc.metadata.get("images", [])
        assert len(images) == 1
        assert f"[IMAGE: {images[0]['id']}]" in doc.text
        # Orphaned reference definition stripped.
        assert "[pic]:" not in doc.text

    def test_remote_image_skipped(self, tmp_path):
        md = _make_markdown(
            tmp_path, "# T\n\n![alt](https://example.com/pic.png)\n\nbody"
        )
        doc = MarkdownLoader().load(md)
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
        # Remote reference preserved verbatim.
        assert "https://example.com/pic.png" in doc.text

    def test_missing_local_image_skipped(self, tmp_path):
        md = _make_markdown(
            tmp_path, "# T\n\n![alt](does_not_exist.png)\n\nbody"
        )
        doc = MarkdownLoader().load(md)
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
        assert "does_not_exist.png" in doc.text

    def test_extraction_disabled(self, tmp_path):
        md = _make_markdown(
            tmp_path, "# T\n\n![alt](pic.png)\n\nbody", image_name="pic.png"
        )
        doc = MarkdownLoader(extract_images=False).load(md)
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
        assert "![alt](pic.png)" in doc.text

    def test_html_img_local_extracted(self, tmp_path):
        body = '# T\n\n<img src="pic.png" alt="x">\n\nbody'
        md = _make_markdown(tmp_path, body, image_name="pic.png")
        doc = MarkdownLoader().load(md)
        assert len(doc.metadata.get("images", [])) == 1


# --------------------------------------------------------------------------
# Encoding robustness
# --------------------------------------------------------------------------
class TestMarkdownEncoding:
    def test_gbk_encoded_file(self, tmp_path):
        md = tmp_path / "gbk.md"
        md.write_bytes("# 标题\n\n中文内容".encode("gbk"))
        doc = MarkdownLoader().load(md)
        assert "标题" in doc.text
        assert "中文内容" in doc.text
