from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from src.config import AGENT_CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FileIngestionHandler:
    """Convert asset files (PDF, etc.) to markdown and optionally write ``.md`` siblings."""

    def __init__(self) -> None:
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

    def ingest_file(self, file_path: Path) -> str:
        """Ingest a file into the vectorstore."""
        result = self.doc_converter.convert(file_path)
        doc = result.document
        markdown_content = doc.export_to_markdown()
        return markdown_content

    def _write_markdown_to_file(self, markdown_content: str, file_path: Path) -> None:
        """Write markdown content to a file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def get_markdown_docs(self) -> dict[str, list[str]]:
        """Load all PDFs under ``assets/``, export markdown, and group by parent folder name."""
        SUPPORTED_EXTENSIONS = {"*.pdf", "*.docx", "*.xlsx", "*.csv", "*.txt"}
    
        markdown_docs: DefaultDict[str, list[str]] = defaultdict(list)
        
        for extension in SUPPORTED_EXTENSIONS:
            for file_path in self.assets_dir.rglob(extension):
                markdown_content = self.ingest_file(file_path)
                self._write_markdown_to_file(markdown_content, file_path.with_suffix(".md"))
                markdown_docs[file_path.parent.name].append(markdown_content)
        
        return dict(markdown_docs)

if __name__ == "__main__":
    file_ingestion = FileIngestionHandler()
    markdown_docs = file_ingestion.get_markdown_docs()
    print(f"Markdown docs: {markdown_docs}")

