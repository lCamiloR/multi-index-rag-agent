from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class Chunker:
    """Chunker class to split markdown content into chunks."""

    def __init__(self) -> None:
        self.headers_to_split_on: list[tuple[str, str]] = [
            ("##", "Sections"),
        ]

    def get_doc_chunks(self, markdown_content: str) -> list[Document]:
        """Split one markdown string into LangChain ``Document`` chunks."""
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        chunks = splitter.split_text(markdown_content)
        return chunks

    def get_all_documents_chunks(
        self,
        markdown_docs: dict[str, list[str]],
    ) -> dict[str, list[list[Document]]]:
        """Chunk every markdown string per asset key into lists of ``Document`` lists."""
        payload: DefaultDict[str, list[list[Document]]] = defaultdict(list)
        for document_index, content_list in markdown_docs.items():
            payload[document_index] = [
                self.get_doc_chunks(mrk_dw) for mrk_dw in content_list
            ]
        return dict(payload)

if __name__ == "__main__":
    test_payload = {
        "TEST": [
            "## Hello World!\nLorem Ipsum\n## John doe\nIpsum Lorem",
            "## Cars\nFord Mustang\n## Bikes\nSuzuki GSX-R1000 K5/K6"
        ]
    }
    chunker = Chunker()
    result = chunker.get_all_documents_chunks(test_payload)
    print(f"Result payload: {result}")