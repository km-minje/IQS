from typing import List, Callable, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class BM25OkapiRetriever:
    def __init__(self, bm25, documents, k=10):
        self.bm25 = bm25
        self.documents = documents
        self.k = k

    @classmethod
    def from_documents(cls, documents: List[Document], preprocess_func: Callable[[str], List[str]], k: int):
        # 문서의 텍스트를 토큰화
        tokenized_corpus = [preprocess_func(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25=bm25, documents=documents, k=k)

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_tokens = self.bm25.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:self.k]
        return [self.documents[i] for i in top_k_indices]
    def _run(self, input_data: str) -> List[Document]:
        """
        `_run` method is required for Runnable, it acts as the execution method.
        """
        return self.get_relevant_documents(input_data)

    def run(self, input_data: str) -> List[Document]:
        """
        This method provides a simple interface to call the run method of Runnable.
        """
        return self._run(input_data)
