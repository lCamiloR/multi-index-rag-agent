from langchain_community.vectorstores import FAISS

from base_embedding_handler import BaseEmbeedngHandler


class VectorstorHandler(BaseEmbeedngHandler):
    
    def save_chunks_to_vectorstore(self, doc_chunks: dict[str, list[list[str]]]) -> None:
        """Embbed the document chunks and save them to the FAISS vector store"""
        for index, chunks in doc_chunks.items():
            vectorstore = FAISS.from_documents(chunks, self.sentence_transformer)
            vectorstore.save_local(self.faiss_local_path, index)
    
