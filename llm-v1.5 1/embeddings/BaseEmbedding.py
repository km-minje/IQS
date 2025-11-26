from typing import List, Callable, Tuple

from langchain.embeddings.base import Embeddings


class BaseEmbedding():
    def __init__(self):
        self.model = None 
        # self.model.eval() 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return None
        # return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return None
        # return self.model.encode(text).tolist()


# class BaseEmbedding(Embeddings):
#     def __init__(self, path: str):
#         super().__init__()
#         self.model = path
#         # self.model.eval() 

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
        
#         """Embed search docs.

#         Args:
#             texts: List of text to embed.

#         Returns:
#             List of embeddings.
#         """
#         return None
#         # return self.model.encode(texts).tolist()

#     def embed_query(self, text: str) -> List[float]:
        
#         """Embed query text.

#         Args:
#             text: Text to embed.

#         Returns:
#             Embedding.
#         """
#         return None
#         # return self.model.encode(text).tolist()
