from typing import List, Callable, Tuple

# from langchain.embeddings.base import Embeddings
from embeddings.BaseEmbedding import BaseEmbedding
from FlagEmbedding import FlagReranker

class FlagEmbeddingReranker(BaseEmbedding):
    """
    FlagEmbeddingReranker 클래스는 FlagReranker 모델을 사용하여
    쿼리와 텍스트의 점수를 계산하는 기능을 제공합니다.
    """
    
    def __init__(self, config):
        """
        클래스의 생성자. 주어진 설정을 사용하여 FlagReranker 모델을 초기화합니다.
        
        :param config: 모델에 대한 설정 정보(모델 경로 포함)
        """
        super().__init__()  # 부모 클래스(BaseEmbedding)의 초기화 호출
        path = config.get("path")  # 설정에서 모델 경로 가져오기
        self.model = FlagReranker(
            path,
            use_fp16=True,  # 반정밀도(FP16)를 사용하여 메모리 최적화
            normalize=True   # 점수 정규화 여부 설정
        )
        # self.compute_score("test", "test")  # 테스트용 점수 계산 (주석으로 처리됨)

    def compute_score(self, query_text: str, text: str) -> List[float]:
        """
        주어진 쿼리와 텍스트에 대해 점수를 계산합니다.
        
        :param query_text: 쿼리 텍스트
        :param text: 비교할 텍스트
        :return: 계산된 점수 리스트
        """
        return self.model.compute_score((query_text, text))  # FlagReranker 모델을 사용하여 점수 반환
    
    def compute_scores(self, query_texts: List[str], texts: List[str]) -> List[float]:
        """
        주어진 쿼리 텍스트 리스트와 비교할 텍스트 리스트에 대한 점수를 계산합니다.
        
        :param query_texts: 쿼리 텍스트의 리스트
        :param texts: 비교할 텍스트의 리스트
        :return: 각 쿼리와 대응되는 텍스트에 대한 계산된 점수 리스트
        """
        pairs = zip(query_texts, texts)  # 쿼리와 텍스트를 쌍으로 묶기
        return self.model.compute_score(pairs)  # FlagReranker 모델을 사용하여 점수 반환