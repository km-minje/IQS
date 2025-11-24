import torch
from typing import List, Callable, Tuple

# from langchain.embeddings.base import Embeddings
from embeddings.BaseEmbedding import BaseEmbedding
from FlagEmbedding import BGEM3FlagModel

class FlagColBERT(BaseEmbedding):
    """
    FlagEmbedding 클래스는 Flag 모델을 사용하여 문서와 쿼리를 임베딩하는 기능을 제공합니다.
    """
    
    def __init__(self, config):
        """
        클래스의 생성자. 주어진 설정을 사용하여 FlagModel을 초기화합니다.
        
        :param config: 모델에 대한 설정 정보(모델 경로 포함)
        """
        super().__init__()  # 부모 클래스(BaseEmbedding)의 초기화 호출
        path = config.get("path")  # 설정에서 모델 경로 가져오기
        self.model = BGEM3FlagModel(
            path,
            # query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True,  # 반정밀도(FP16)를 사용하여 메모리 최적화
        )
        # self.model.eval()  # 모델을 평가 모드로 설정 (주석으로 처리됨)
        # self.embed_query("test")  # 테스트용 쿼리 임베딩 (주석으로 처리됨)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        주어진 문서 텍스트 리스트를 임베딩하여 결과 리스트를 반환합니다.
        
        :param texts: 임베딩할 문서 텍스트의 리스트
        :return: 각 문서에 대한 임베딩 리스트
        """
        return self.model.encode(texts, return_colbert_vecs=True)  # Flag 모델을 사용하여 문서 임베딩 후 리스트 반환

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """
        주어진 쿼리 텍스트 리스트를 임베딩하여 결과 리스트를 반환합니다.
        
        :param texts: 임베딩할 쿼리 텍스트의 리스트
        :return: 각 쿼리에 대한 임베딩 리스트
        """
        return self.model.encode(texts)['colbert_vecs'] # Flag 모델을 사용하여 쿼리 임베딩 후 리스트 반환

    def embed_query(self, text: str) -> List[float]:
        """
        주어진 단일 쿼리 텍스트를 임베딩하여 결과 리스트를 반환합니다.
        
        :param text: 임베딩할 쿼리 텍스트
        :return: 쿼리에 대한 임베딩 리스트
        """
        return self.model.encode([text], return_colbert_vecs=True) # Flag 모델을 사용하여 쿼리 임베딩 후 리스트 반환
    
    def colbert_score(self, q_reps, p_reps):
        """Compute colbert scores of input queries and passages.

        Args:
            q_reps (np.ndarray): Multi-vector embeddings for queries.
            p_reps (np.ndarray): Multi-vector embeddings for passages/corpus.

        Returns:
            torch.Tensor: Computed colbert scores.
        """
        q_reps, p_reps = torch.from_numpy(q_reps), torch.from_numpy(p_reps)
        token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / q_reps.size(0)
        return scores