from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.base_embedding_handler import BaseEmbeedngHandler


class VectorstoreHandler(BaseEmbeedngHandler):
    """Persist and list FAISS indexes under ``faiss_local_path``."""

    def save_chunks_to_vectorstore(
        self,
        doc_chunks: dict[str, list[list[Document]]],
    ) -> None:
        """Embbed the document chunks and save them to the FAISS vector store"""
        for index, chunks_matrix in doc_chunks.items():
            for chunks in chunks_matrix:
                vectorstore = FAISS.from_documents(chunks, self.sentence_transformer)
                vectorstore.save_local(self.faiss_local_path, index)

    def list_indexes(self) -> list[str]:
        """List all available FAISS indexes in the vector store directory"""
        vectorstore_path = Path(self.faiss_local_path)
        
        if not vectorstore_path.exists():
            return []
        
        indexes = [d.name.replace(d.suffix, "") for d in vectorstore_path.iterdir() if d.is_file() and d.suffix == ".faiss"]
        return sorted(indexes)
 