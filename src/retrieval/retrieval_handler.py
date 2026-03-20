from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from src.base_embedding_handler import BaseEmbeedngHandler

class RetrievalHandler(BaseEmbeedngHandler):

    def __init__(self, embedding_model: str, faiss_local_path: str, index_name: str = "index"):
        super().__init__(embedding_model, faiss_local_path)
        self.loaded_vectorstore = FAISS.load_local(
            self.faiss_local_path,
            self.sentence_transformer,
            index_name,
            allow_dangerous_deserialization=True
        )
        self.retriever: VectorStoreRetriever | None = None
        self.top_k: int | None = None

    def _load_retrieval(self, top_k: int):
        """Create the retriever object"""
        return self.loaded_vectorstore.as_retriever(kwargs={ "search_type": "similarity", "search_kwargs": { "k": top_k }})

    def query_vectorstore(self, query: str, top_k: int = 5) -> str:
        """
        Perform a similarity search on the vectorstore about owasp top 10 llm security vulns.
        Args:
            query (str): The search query.
        Returns:
            List of relevant documents.
        """
        if not self.retriever or top_k != self.top_k:
            self.top_k = top_k
            self.retriever = self._load_retrieval(top_k=self.top_k)
        docs = self.retriever.invoke(query)

        return "\n\n".join(doc.page_content for doc in docs)
