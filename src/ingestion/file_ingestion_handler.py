from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterator

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from src.config import AGENT_CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SUPPORTED_ASSET_EXTENSIONS = frozenset({"pdf", "docx", "xlsx", "csv", "txt"})


class FileIngestionHandler:
    """Convert asset files (PDF, etc.) to markdown and optionally write ``.md`` siblings."""

    def __init__(self) -> None:
        """Configure paths from settings and build the Docling :class:`DocumentConverter`."""
        self.assets_dir = AGENT_CONFIG.ASSETS_PATH

        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = False
        self.doc_converter: DocumentConverter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.XLSX,
                InputFormat.CSV,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
            }
        )

    def iter_supported_asset_files(self) -> Iterator[Path]:
        """Yield asset files that participate in ingestion (same extensions as ``get_markdown_docs``)."""
        for extension in sorted(SUPPORTED_ASSET_EXTENSIONS):
            yield from self.assets_dir.rglob(f"*.{extension}")

    def get_asset_file_fingerprints(self) -> dict[str, dict[str, int]]:
        """Relative path (from ``assets_dir``) -> ``mtime_ns`` and ``size`` for change detection."""
        fingerprints: dict[str, dict[str, int]] = {}
        for file_path in self.iter_supported_asset_files():
            try:
                st = file_path.stat()
            except OSError:
                continue
            rel = file_path.relative_to(self.assets_dir).as_posix()
            fingerprints[rel] = {"mtime_ns": st.st_mtime_ns, "size": st.st_size}
        return dict(sorted(fingerprints.items()))

    def index_names_for_current_assets(self) -> set[str]:
        """FAISS index names implied by the current tree (parent folder of each supported file)."""
        return {p.parent.name for p in self.iter_supported_asset_files()}

    def ingest_file(self, file_path: Path) -> str:
        """Convert ``file_path`` with Docling and return exported markdown text."""
        result = self.doc_converter.convert(file_path)
        doc = result.document
        markdown_content = doc.export_to_markdown()
        return markdown_content

    def _write_markdown_to_file(self, markdown_content: str, file_path: Path) -> None:
        """Write markdown content to a file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def get_markdown_docs(self) -> dict[str, list[dict]]:
        """Load all PDFs under ``assets/``, export markdown, and group by parent folder name."""
        markdown_docs: DefaultDict[str, list[str]] = defaultdict(list)

        for extension in sorted(SUPPORTED_ASSET_EXTENSIONS):
            for file_path in self.assets_dir.rglob(f"*.{extension}"):
                markdown_content = self.ingest_file(file_path)
                self._write_markdown_to_file(markdown_content, file_path.with_suffix(".md"))
                markdown_docs[file_path.parent.name].append({"markdown_content": markdown_content, "extension": extension})
        
        return dict(markdown_docs)

if __name__ == "__main__":
    file_ingestion = FileIngestionHandler()
    markdown_docs = file_ingestion.get_markdown_docs()
    print(f"Markdown docs: {markdown_docs}")

