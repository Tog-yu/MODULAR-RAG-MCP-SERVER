"""Unit tests for the LoaderFactory extension-based dispatch.

Verifies:
- Built-in registration of PDF and Markdown loaders
- Extension resolution (incl. case-insensitivity, .markdown alias)
- Graceful "unsupported" handling
- Config forwarding filtered to each loader's signature
- Custom registration via ``register`` / ``unregister``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader


class TestLoaderFactoryBuiltins:
    def test_markdown_registered(self):
        assert LoaderFactory.is_supported("guide.md")
        loader = LoaderFactory.get_loader("guide.md")
        assert isinstance(loader, MarkdownLoader)

    def test_markdown_alias_registered(self):
        assert LoaderFactory.is_supported("guide.markdown")
        loader = LoaderFactory.get_loader("guide.markdown")
        assert isinstance(loader, MarkdownLoader)

    def test_pdf_registered(self):
        assert LoaderFactory.is_supported("report.PDF")  # case-insensitive
        loader = LoaderFactory.get_loader("report.PDF")
        assert isinstance(loader, PdfLoader)

    def test_supported_extensions_sorted(self):
        exts = LoaderFactory.supported_extensions()
        assert ".md" in exts
        assert ".markdown" in exts
        assert ".pdf" in exts
        # Sorted lexicographically.
        assert exts == sorted(exts)


class TestLoaderFactoryUnsupported:
    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            LoaderFactory.get_loader("notes.txt")

    def test_is_supported_false_for_unknown(self):
        assert not LoaderFactory.is_supported("notes.txt")


class TestLoaderFactoryConfigForwarding:
    def test_config_forwarded_to_markdown_loader(self, tmp_path):
        loader = LoaderFactory.get_loader(
            "guide.md",
            image_storage_dir=str(tmp_path / "images"),
            extract_markdown_images=False,
        )
        # extract_markdown_images takes precedence over extract_images default.
        assert isinstance(loader, MarkdownLoader)
        assert loader.extract_images is False
        assert loader.image_storage_dir == tmp_path / "images"

    def test_unknown_config_key_is_filtered(self):
        # PdfLoader does not accept extract_markdown_images; it must be ignored
        # rather than raising a TypeError.
        loader = LoaderFactory.get_loader("report.pdf", extract_markdown_images=False)
        assert isinstance(loader, PdfLoader)


class TestLoaderFactoryCustomRegistration:
    def test_custom_registration(self):
        class _Dummy(BaseLoader):
            def load(self, file_path):  # pragma: no cover - test only
                raise NotImplementedError

        try:
            LoaderFactory.register([".dummy"], _Dummy)
            assert LoaderFactory.is_supported("x.dummy")
            assert isinstance(LoaderFactory.get_loader("x.dummy"), _Dummy)
        finally:
            # Clean up so other tests are unaffected.
            LoaderFactory.unregister([".dummy"])

    def test_unregister(self):
        class _Dummy(BaseLoader):
            def load(self, file_path):  # pragma: no cover - test only
                raise NotImplementedError

        LoaderFactory.register([".zzz"], _Dummy)
        assert LoaderFactory.is_supported("x.zzz")
        LoaderFactory.unregister([".zzz"])
        assert not LoaderFactory.is_supported("x.zzz")
