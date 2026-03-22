from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class Chunker:
    """Chunker class to split markdown content into chunks."""

    def get_doc_chunks(self, markdown_content: str) -> list[Document]:
        """Split one markdown string into LangChain ``Document`` chunks."""
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Sections")])
        chunks = splitter.split_text(markdown_content)
        return chunks

    def get_table_chunks(self, markdown_content: str) -> list[Document]:
        """Split tabular markdown into line-oriented chunks with zero-based row metadata."""
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"])
        chunks = splitter.split_text(markdown_content)
        return [Document(page_content=chunk, metadata={"row_number": index}) for index, chunk in enumerate(chunks)]

    def get_all_documents_chunks(
        self,
        markdown_docs: dict[str, list[dict]],
    ) -> dict[str, list[list[Document]]]:
        """Chunk every markdown string per asset key into lists of ``Document`` lists."""
        payload: DefaultDict[str, list[list[Document]]] = defaultdict(list)
        for document_index, content_list in markdown_docs.items():
            doc_list = []
            for mrk_dw in content_list:
                if mrk_dw["extension"] == "csv":
                    doc_list.append(self.get_table_chunks(mrk_dw["markdown_content"]))
                else:
                    doc_list.append(self.get_doc_chunks(mrk_dw["markdown_content"]))
            payload[document_index] = doc_list
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