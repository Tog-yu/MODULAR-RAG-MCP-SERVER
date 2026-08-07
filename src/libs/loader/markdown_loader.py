"""Markdown Loader implementation.

This module implements Markdown parsing for the ingestion pipeline.

Design notes:
- Markdown is already the canonical format the downstream pipeline expects
  (``Document.text`` is standardized Markdown consumed by the Markdown-aware
  ``RecursiveCharacterTextSplitter``). Therefore this loader does **not** run
  the source through any converter (unlike the PDF loader, which must convert
  PDF -> Markdown). It reads the raw Markdown text directly, preserving the
  heading / list / code-block structure that the splitter relies on.
- It extracts structured metadata: YAML frontmatter, title, a heading outline,
  and (optionally) local images referenced via ``![alt](path)``.
- Images referenced by *local* paths are copied into
  ``data/images/{doc_hash}/`` and registered in ``metadata.images`` with a
  ``[IMAGE: {id}]`` placeholder -- exactly mirroring the PDF loader's
  multimodal flow so "search text -> return image" works uniformly.
  Remote (http/https/data:) images and missing local files are skipped with
  graceful degradation.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# --- Regexes ---------------------------------------------------------------
# Inline image: ![alt](url)  or  ![alt](url "title")
_INLINE_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# Reference-style image: ![alt][label]
_REF_IMG_RE = re.compile(r"!\[([^\]]*)\]\[([^\]]+)\]")
# HTML <img> tag: <img src="..." ...>
_HTML_IMG_RE = re.compile(r"<img\b[^>]*?src=[\"']([^\"']+)[\"'][^>]*?>", re.IGNORECASE)
# Reference definition: [label]: url
_REF_DEF_RE = re.compile(r"^\s*\[([^\]]+)\]:\s*(\S+)\s*$", re.MULTILINE)
# Heading: # Heading  ..  ###### Heading
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
# YAML frontmatter at the very start of the document.
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

# Encodings attempted when decoding a Markdown file.
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin-1")


class MarkdownLoader(BaseLoader):
    """Markdown Loader.

    Parses a ``.md`` / ``.markdown`` file into a standardized ``Document``:

    - ``text``: raw Markdown body (frontmatter stripped, image placeholders
      inserted) -- ready for the Markdown-aware splitter.
    - ``metadata``: ``source_path`` (required), ``doc_type='markdown'``,
      ``doc_hash``, ``title``, ``heading_outline``, ``word_count``,
      ``line_count``, optional ``frontmatter`` and ``images``.

    Configuration:
        extract_images:    Copy local images into the image store and insert
                           ``[IMAGE: {id}]`` placeholders (default: True).
        image_storage_dir: Base directory for extracted images
                           (default: data/images). Final path is
                           ``{image_storage_dir}/{doc_hash}/``.
    """

    # Extensions this loader is responsible for (used by LoaderFactory).
    SUPPORTED_EXTENSIONS = (".md", ".markdown")

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        extract_markdown_images: Optional[bool] = None,
    ) -> None:
        """Initialize the Markdown loader.

        Args:
            extract_images: Whether to extract/copy local images. Used when
                ``extract_markdown_images`` is not provided.
            image_storage_dir: Base directory for storing extracted images.
            extract_markdown_images: Explicit toggle for Markdown image
                extraction; takes precedence over ``extract_images`` when set.
        """
        if extract_markdown_images is not None:
            self.extract_images = extract_markdown_images
        else:
            self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)

    # ------------------------------------------------------------------ load
    def load(self, file_path: str | Path) -> Document:
        """Load and parse a Markdown file.

        Args:
            file_path: Path to the ``.md`` / ``.markdown`` file.

        Returns:
            Document with Markdown text and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a Markdown file.
            RuntimeError: If reading/parsing fails critically.
        """
        path = self._validate_file(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"File is not a Markdown file: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        # Read raw text (encoding fallback).
        try:
            raw_text = self._read_text(path)
        except Exception as e:
            logger.error(f"Failed to read Markdown {path}: {e}")
            raise RuntimeError(f"Markdown reading failed: {e}") from e

        # Split frontmatter from body.
        body, frontmatter = self._parse_frontmatter(raw_text)

        # Build base metadata.
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "markdown",
            "doc_hash": doc_hash,
            "word_count": len(body.split()),
            "line_count": body.count("\n") + 1,
        }

        # Title: frontmatter.title > first H1 > first non-empty line > filename.
        title = frontmatter.get("title") if isinstance(frontmatter.get("title"), str) else None
        if not title:
            title = self._extract_title(body)
        if not title:
            title = path.stem
        metadata["title"] = title

        if frontmatter:
            metadata["frontmatter"] = frontmatter

        # Heading outline (structure-aware navigation aid).
        outline = self._extract_heading_outline(body)
        if outline:
            metadata["heading_outline"] = outline

        # Image extraction (with graceful degradation).
        if self.extract_images:
            try:
                body, images_metadata = self._extract_and_process_images(
                    path, body, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, "
                    f"continuing with text-only: {e}"
                )

        return Document(id=doc_id, text=body, metadata=metadata)

    # ------------------------------------------------------------- helpers
    def _read_text(self, path: Path) -> str:
        """Read file content with encoding fallback.

        Tries UTF-8 (with BOM), UTF-8, GBK, then Latin-1. As a last resort
        decodes with replacement characters rather than crashing.

        Args:
            path: Resolved Path to the file.

        Returns:
            Decoded text content.
        """
        raw = path.read_bytes()
        for enc in _TEXT_ENCODINGS:
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        # Final fallback -- never raise on decode.
        return raw.decode("utf-8", errors="replace")

    def _parse_frontmatter(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Split YAML frontmatter from the Markdown body.

        Args:
            text: Full file text.

        Returns:
            (body_without_frontmatter, frontmatter_dict). If no valid
            frontmatter is found, returns (text, {}).
        """
        if not text.lstrip("\ufeff").startswith("---"):
            return text, {}
        match = _FRONTMATTER_RE.match(text)
        if not match:
            return text, {}
        try:
            fm = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML frontmatter: {e}")
            return text, {}
        if not isinstance(fm, dict):
            return text, {}
        return text[match.end():], fm

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line."""
        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in lines[:10]:
            line = line.strip()
            if line:
                return line
        return None

    def _extract_heading_outline(self, text: str) -> List[Dict[str, Any]]:
        """Extract a structured heading outline (level + text)."""
        outline: List[Dict[str, Any]] = []
        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))
            heading = m.group(2).strip()
            outline.append({"level": level, "text": heading})
        return outline

    def _extract_and_process_images(
        self,
        md_path: Path,
        text_content: str,
        doc_hash: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract local images and insert ``[IMAGE: {id}]`` placeholders.

        Mirrors the PDF loader's multimodal flow: local image references are
        copied into ``{image_storage_dir}/{doc_hash}/`` and registered in
        ``metadata.images``. Remote (http/https/data:) and missing files are
        left untouched.

        Args:
            md_path: Path to the source Markdown file (for relative resolution).
            text_content: Markdown body text.
            doc_hash: Document hash (used for the image directory & ids).

        Returns:
            (modified_text, images_metadata_list).
        """
        if not self.extract_images:
            return text_content, []

        # Build reference map: label -> url (lowercased keys).
        ref_map: Dict[str, str] = {}
        for m in _REF_DEF_RE.finditer(text_content):
            ref_map[m.group(1).strip().lower()] = m.group(2).strip()

        # Collect every image occurrence with its resolved url and span.
        occurrences: List[Tuple[int, int, str]] = []  # (start, end, url)
        for m in _INLINE_IMG_RE.finditer(text_content):
            occurrences.append((m.start(), m.end(), m.group(2).strip()))
        for m in _REF_IMG_RE.finditer(text_content):
            label = m.group(2).strip().lower()
            url = ref_map.get(label, "")
            occurrences.append((m.start(), m.end(), url))
        for m in _HTML_IMG_RE.finditer(text_content):
            occurrences.append((m.start(), m.end(), m.group(1).strip()))

        if not occurrences:
            return text_content, []

        # Prepare image storage directory.
        image_dir = self.image_storage_dir / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        images_metadata: List[Dict[str, Any]] = []
        # Deduplicate by resolved absolute path so the same image is stored once.
        seen_paths: Dict[str, str] = {}
        seq = 0

        new_parts: List[str] = []
        last = 0
        for start, end, url in occurrences:
            new_parts.append(text_content[last:start])
            if not url or self._is_remote(url):
                # Leave remote / unresolvable references untouched.
                new_parts.append(text_content[start:end])
                last = end
                continue

            src = (md_path.parent / url).resolve()
            if not src.exists() or not src.is_file():
                logger.debug(f"Image not found, skipping: {url}")
                new_parts.append(text_content[start:end])
                last = end
                continue

            # Reuse id if the same file was already extracted.
            if str(src) in seen_paths:
                image_id = seen_paths[str(src)]
            else:
                seq += 1
                image_id = self._generate_image_id(doc_hash, seq)
                ext = src.suffix or ".png"
                target = image_dir / f"{image_id}{ext}"
                try:
                    shutil.copyfile(src, target)
                except Exception as e:
                    logger.warning(f"Failed to copy image {src}: {e}")
                    new_parts.append(text_content[start:end])
                    last = end
                    continue
                seen_paths[str(src)] = image_id

                # Store path relative to cwd when possible (matches PdfLoader).
                try:
                    stored_path = str(target.relative_to(Path.cwd()))
                except ValueError:
                    stored_path = str(target.absolute())

                images_metadata.append({
                    "id": image_id,
                    "path": stored_path,
                    "page": 0,  # Markdown has no pages
                    "text_offset": len("".join(new_parts)),
                    "text_length": len(f"[IMAGE: {image_id}]"),
                    "position": {"line": 0, "index": seq},
                })

            placeholder = f"[IMAGE: {image_id}]"
            new_parts.append(placeholder)
            last = end

        new_parts.append(text_content[last:])
        modified_text = "".join(new_parts)

        # Strip orphaned reference-definition lines (urls already resolved).
        modified_text = _REF_DEF_RE.sub("", modified_text)

        return modified_text, images_metadata

    @staticmethod
    def _is_remote(url: str) -> bool:
        """Return True for http(s)/data/ protocol-relative image urls."""
        return url.startswith(("http://", "https://", "data:", "//"))

    @staticmethod
    def _generate_image_id(doc_hash: str, sequence: int) -> str:
        """Generate a unique image id (page-less variant for Markdown)."""
        return f"{doc_hash[:8]}_{sequence}"
