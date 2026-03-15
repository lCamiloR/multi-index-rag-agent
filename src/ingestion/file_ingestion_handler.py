
from collections import defaultdict
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter, 
    PdfFormatOption, 
    ExcelFormatOption, 
    WordFormatOption,
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FileIngestionHandler:

    def __init__(self):
        self.assets_dir = Path(__file__).resolve().parents[2] / 'assets'

        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = False
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                InputFormat.DOCX: WordFormatOption(),
                InputFormat.XLSX: ExcelFormatOption()
            }
        )
    
    def ingest_file(self, file_path: Path) -> str:
        """Ingest a file into the vectorstore."""
        result = self.doc_converter.convert(file_path)
        doc = result.document
        markdown_content = doc.export_to_markdown()
        return markdown_content

    def _write_markdown_to_file(self, markdown_content: str, file_path: Path):
        """Write markdown content to a file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

    def get_markdown_docs(self) -> list[str]:
        """Get all markdown documents from the assets directory."""
        markdown_docs = defaultdict(list)
        for file_path in self.assets_dir.rglob('*.pdf'):
            if file_path.parent.name not in markdown_docs:
                markdown_docs[file_path.parent.name] = []
            markdown_content = self.ingest_file(file_path)
            self._write_markdown_to_file(markdown_content, file_path.with_suffix('.md'))
            markdown_docs[file_path.parent.name].append(markdown_content)
        return markdown_docs

if __name__ == "__main__":
    file_ingestion = FileIngestionHandler()
    markdown_docs = file_ingestion.get_markdown_docs()
    print(f"Markdown docs: {markdown_docs}")

