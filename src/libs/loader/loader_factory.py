"""LoaderFactory: extension-based dispatch for document loaders.

Maps file extensions to ``BaseLoader`` implementations so the ingestion
pipeline can stay format-agnostic. Built-in loaders (PDF, Markdown) register
themselves when this module is imported; new formats only need to implement
``BaseLoader`` and call ``LoaderFactory.register``.

Example:
    >>> from src.libs.loader.loader_factory import LoaderFactory
    >>> loader = LoaderFactory.get_loader("docs/guide.md", image_storage_dir="data/images")
    >>> isinstance(loader, MarkdownLoader)
    True
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type

from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class LoaderFactory:
    """Factory that resolves a file to its loader by extension."""

    _registry: Dict[str, Type[BaseLoader]] = {}

    # ----------------------------------------------------------------- register
    @classmethod
    def register(
        cls,
        extensions: List[str],
        loader_cls: Type[BaseLoader],
    ) -> None:
        """Register a loader class for one or more file extensions.

        Args:
            extensions: List of extensions, e.g. ``[".md", ".markdown"]``.
            loader_cls: The ``BaseLoader`` subclass to instantiate.
        """
        for ext in extensions:
            normalized = ext.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            cls._registry[normalized] = loader_cls
            logger.debug(f"Registered loader {loader_cls.__name__} for {normalized}")

    @classmethod
    def unregister(cls, extensions: List[str]) -> None:
        """Remove a loader registration for one or more extensions.

        Useful for tests or dynamic reconfiguration. Silently ignores
        extensions that are not currently registered.

        Args:
            extensions: List of extensions, e.g. ``[".dummy"]``.
        """
        for ext in extensions:
            normalized = ext.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            cls._registry.pop(normalized, None)

    # ------------------------------------------------------------------- query
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return all registered extensions (sorted)."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Return True if a loader is registered for the file's extension."""
        return Path(file_path).suffix.lower() in cls._registry

    # ------------------------------------------------------------------- build
    @classmethod
    def get_loader(cls, file_path: str | Path, **config: object) -> BaseLoader:
        """Return a loader instance appropriate for *file_path*.

        Args:
            file_path: The file to be loaded (extension drives dispatch).
            **config: Keyword args forwarded to the loader constructor
                (e.g. ``image_storage_dir``, ``extract_images``).

        Returns:
            A ``BaseLoader`` instance.

        Raises:
            ValueError: If no loader is registered for the extension.
        """
        ext = Path(file_path).suffix.lower()
        loader_cls = cls._registry.get(ext)
        if loader_cls is None:
            supported = ", ".join(cls.supported_extensions()) or "(none)"
            raise ValueError(
                f"Unsupported file type '{ext}' for {file_path}. "
                f"Supported extensions: {supported}"
            )
        # Only forward constructor kwargs the loader actually accepts, so a
        # format-specific option (e.g. extract_markdown_images) is safely
        # ignored by loaders that don't define it.
        sig = inspect.signature(loader_cls.__init__)
        accepted = {
            name
            for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and name != "self"
        }
        filtered = {k: v for k, v in config.items() if k in accepted}
        return loader_cls(**filtered)


# ---------------------------------------------------------------------------
# Built-in registration (import side-effects).
# ---------------------------------------------------------------------------
from src.libs.loader.markdown_loader import MarkdownLoader  # noqa: E402
from src.libs.loader.pdf_loader import PdfLoader  # noqa: E402

LoaderFactory.register([".pdf"], PdfLoader)
LoaderFactory.register([".md", ".markdown"], MarkdownLoader)
