
from langchain_community.embeddings import HuggingFaceEmbeddings


class BaseEmbeedngHandler:

    def __init__(self, embedding_model: str, faiss_local_path: str):
        self.embedding_model: str = embedding_model
        self.faiss_local_path: str = faiss_local_path
        self.sentence_transformer = HuggingFaceEmbeddings(model_name=self.embedding_model)
    
