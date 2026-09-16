"""Document Quality Gate: pre-load quality inspection for ingested files.

This module implements the C2.5 quality pre-check stage of the ingestion
pipeline. Before the (potentially expensive) full parse performed by the
loaders, a lightweight sampling of the first ``preview_pages`` pages is
analyzed to estimate the quality of the PDF's text layer:

- ``valid_char_ratio``: fraction of valid characters (CJK + letters + digits)
  among all non-whitespace characters.  A low value indicates corrupted,
  mis-encoded, or garbled text that would poison downstream chunks.
- ``text_density``: fraction of sampled pages that carry a meaningful amount
  of text (``>= min_chars_per_page`` valid characters).  A low value
  indicates a scan-like PDF without an OCR text layer.

A document is admitted only when BOTH metrics pass their thresholds.  Files
registered for inspection (default: ``.pdf``) are checked; other formats are
allowed through untouched to avoid false positives on Markdown/plain text.

Design Principles:
- Fail fast at the entrance: rejection happens BEFORE any Document parsing
  or image extraction side effects.
- Config-driven: thresholds / sampling size come from settings.yaml.
- Format-agnostic dispatch: per-extension registration, mirroring
  ``LoaderFactory`` style.
- Rejection is a business outcome, not an exception: the pipeline decides
  how to surface it (PipelineResult.rejected=True, Dashboard message).

Example:
    >>> from src.libs.loader.quality_gate import DocumentQualityGate, QualityGateConfig
    >>> gate = DocumentQualityGate(QualityGateConfig())
    >>> result = gate.check("data/documents/report.pdf")
    >>> if not result.passed:
    ...     print(result.reason)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# User-facing message required by the spec (Dashboard displays it verbatim).
QUALITY_REJECT_MESSAGE = "该文档质量不达标，请检查后重新上传"

# Valid characters: CJK unified ideographs + ASCII letters/digits.
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# Default extensions subject to quality inspection.  Text-native formats
# (Markdown, plain text) have no "text layer" concept and must not be
# rejected, so they are not registered.
DEFAULT_QUALITY_EXTENSIONS = [".pdf"]


@dataclass(frozen=True)
class QualityGateConfig:
    """Configuration for the document quality gate.

    Attributes:
        enabled: Master switch.  When False the gate is skipped entirely.
        min_valid_char_ratio: Minimum fraction of valid characters among
            non-whitespace characters (default 0.80 per spec C2.5).
        min_text_density: Minimum fraction of sampled pages that are
            "text-bearing" (default 0.50 per spec C2.5).
        preview_pages: Number of leading pages to sample (default 3).
        min_chars_per_page: Minimum valid characters for a page to count
            as text-bearing (default 50).
        max_sample_chars: Safety cap on characters extracted per page to
            avoid pathological memory usage on huge single-page PDFs.
        extensions: File extensions subject to inspection.
    """

    enabled: bool = True
    min_valid_char_ratio: float = 0.80
    min_text_density: float = 0.50
    preview_pages: int = 3
    min_chars_per_page: int = 50
    max_sample_chars: int = 20000
    extensions: List[str] = field(default_factory=lambda: list(DEFAULT_QUALITY_EXTENSIONS))

    def applies_to(self, file_path: str | Path) -> bool:
        """Return True if the file's extension is subject to inspection."""
        ext = Path(file_path).suffix.lower()
        normalized = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in self.extensions]
        return ext in normalized


@dataclass
class QualityCheckResult:
    """Outcome of a single quality check.

    Attributes:
        passed: Whether the document may proceed to loading.
        valid_char_ratio: Computed valid-character ratio (0 if unavailable).
        text_density: Computed text density (0 if unavailable).
        sampled_pages: Number of pages actually sampled.
        reason: Human-readable rejection reason (None when passed).  The
            first line is always ``QUALITY_REJECT_MESSAGE`` so the Dashboard
            can display it directly; metric details follow for debugging.
    """

    passed: bool
    valid_char_ratio: float = 0.0
    text_density: float = 0.0
    sampled_pages: int = 0
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for PipelineResult.stages / trace consumption."""
        return {
            "passed": self.passed,
            "valid_char_ratio": round(self.valid_char_ratio, 4),
            "text_density": round(self.text_density, 4),
            "sampled_pages": self.sampled_pages,
            "reason": self.reason,
        }


class DocumentQualityGate:
    """Lightweight pre-load quality inspector for ingested documents.

    Samples the first ``preview_pages`` pages with PyMuPDF (fitz), computes
    two quality metrics and compares them against configured thresholds.

    Example:
        >>> gate = DocumentQualityGate(QualityGateConfig())
        >>> result = gate.check("tests/fixtures/low_quality.pdf")
        >>> result.passed
        False
    """

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()

    # ------------------------------------------------------------------ api
    def check(self, file_path: str | Path) -> QualityCheckResult:
        """Run the quality pre-check on a file.

        Args:
            file_path: Path to the file to inspect.

        Returns:
            QualityCheckResult describing the outcome.  For formats not
            registered in ``config.extensions`` the check passes trivially
            (``passed=True``) without sampling.
        """
        path = Path(file_path)

        if not self.config.enabled:
            logger.debug("Quality gate disabled, passing through: %s", path)
            return QualityCheckResult(passed=True, sampled_pages=0)

        if not self.config.applies_to(path):
            logger.debug("Quality gate not registered for this format, passing: %s", path)
            return QualityCheckResult(passed=True, sampled_pages=0)

        if not path.exists():
            return QualityCheckResult(
                passed=False,
                reason=f"{QUALITY_REJECT_MESSAGE}（文件不存在: {path}）",
            )

        # Lazy import: fitz is an optional hard dependency for this gate.
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not available; quality gate cannot sample, rejecting.")
            return QualityCheckResult(
                passed=False,
                reason=f"{QUALITY_REJECT_MESSAGE}（预检依赖 PyMuPDF 不可用）",
            )

        page_texts = self._extract_sample_text(path, fitz)
        if page_texts is None:
            return QualityCheckResult(
                passed=False,
                reason=f"{QUALITY_REJECT_MESSAGE}（文件损坏或非合法 PDF，无法解析）",
            )

        sampled_pages = len(page_texts)
        full_text = "\n".join(page_texts)

        ratio = self._compute_valid_char_ratio(full_text)
        density = self._compute_text_density(page_texts)

        failures: List[str] = []
        if ratio < self.config.min_valid_char_ratio:
            failures.append(
                f"有效字符占比 {ratio:.2%} 低于阈值 {self.config.min_valid_char_ratio:.0%}"
            )
        if density < self.config.min_text_density:
            failures.append(
                f"可识别文本密度 {density:.2%} 低于阈值 {self.config.min_text_density:.0%}"
            )

        if failures:
            reason = f"{QUALITY_REJECT_MESSAGE}（" + "；".join(failures) + "）"
            logger.info("Quality gate rejected %s: %s", path.name, reason)
            return QualityCheckResult(
                passed=False,
                valid_char_ratio=ratio,
                text_density=density,
                sampled_pages=sampled_pages,
                reason=reason,
            )

        logger.info(
            "Quality gate passed %s (ratio=%.2f%%, density=%.2f%%, pages=%d)",
            path.name,
            ratio * 100,
            density * 100,
            sampled_pages,
        )
        return QualityCheckResult(
            passed=True,
            valid_char_ratio=ratio,
            text_density=density,
            sampled_pages=sampled_pages,
        )

    # ------------------------------------------------------------ internals
    def _extract_sample_text(self, path: Path, fitz) -> Optional[List[str]]:
        """Extract text of the first ``preview_pages`` pages.

        Args:
            path: PDF file path.
            fitz: The imported ``fitz`` module (injected for testability).

        Returns:
            List of per-page text strings (possibly empty strings for pages
            without a text layer), or None when the file cannot be opened.
        """
        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.warning("Quality gate failed to open %s: %s", path, e)
            return None

        try:
            if doc.page_count <= 0:
                return None
            pages: List[str] = []
            for page_num in range(min(self.config.preview_pages, doc.page_count)):
                page = doc[page_num]
                text = page.get_text("text") or ""
                pages.append(text[: self.config.max_sample_chars])
            return pages
        except Exception as e:
            logger.warning("Quality gate failed to sample %s: %s", path, e)
            return None
        finally:
            doc.close()

    def _compute_valid_char_ratio(self, text: str) -> float:
        """Compute valid characters / non-whitespace characters.

        Valid characters are CJK ideographs plus letters and digits.
        Returns 1.0 for empty input so a blank-but-parseable document is
        judged by text_density instead (which will reject it).
        """
        non_ws = [ch for ch in text if not ch.isspace()]
        if not non_ws:
            return 1.0
        valid = sum(1 for ch in non_ws if _CJK_RE.fullmatch(ch) or _ALNUM_RE.fullmatch(ch))
        return valid / len(non_ws)

    def _compute_text_density(self, page_texts: List[str]) -> float:
        """Compute the fraction of sampled pages bearing meaningful text."""
        if not page_texts:
            return 0.0
        text_pages = sum(
            1
            for t in page_texts
            if self._compute_valid_char_ratio(t) > 0  # pages with any content
            and self._count_valid_chars(t) >= self.config.min_chars_per_page
        )
        return text_pages / len(page_texts)

    @staticmethod
    def _count_valid_chars(text: str) -> int:
        """Count valid characters (CJK + alnum) in text."""
        return sum(1 for ch in text if _CJK_RE.fullmatch(ch) or _ALNUM_RE.fullmatch(ch))
