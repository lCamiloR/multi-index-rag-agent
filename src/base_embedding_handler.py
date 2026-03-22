from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings


class BaseEmbeedngHandler:
    """Base class for components that use HuggingFace sentence embeddings and a FAISS path."""

    def __init__(
        self,
        embedding_model: str,
        faiss_local_path: Path,
    ) -> None:
        self.embedding_model: str = embedding_model
        self.faiss_local_path: Path = faiss_local_path
        self.sentence_transformer: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )
    
