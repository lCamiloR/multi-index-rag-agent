from pathlib import Path

from langchain_community.vectorstores import FAISS

from base_embedding_handler import BaseEmbeedngHandler


class VectorstoreHandler(BaseEmbeedngHandler):
    
    def save_chunks_to_vectorstore(self, doc_chunks: dict[str, list[list[str]]]) -> None:
        """Embbed the document chunks and save them to the FAISS vector store"""
        for index, chunks in doc_chunks.items():
            vectorstore = FAISS.from_documents(chunks, self.sentence_transformer)
            vectorstore.save_local(self.faiss_local_path, index)

    def list_indexes(self) -> list[str]:
        """List all available FAISS indexes in the vector store directory"""
        vectorstore_path = Path(self.faiss_local_path)
        
        if not vectorstore_path.exists():
            return []
        
        # Get all subdirectories that represent indexes
        indexes = [d.name for d in vectorstore_path.iterdir() if d.is_dir()]
        return sorted(indexes)
    
    
