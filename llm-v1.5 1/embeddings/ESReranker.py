from typing import List, Callable, Tuple
from embeddings.BaseEmbedding import BaseEmbedding
from elasticsearch import Elasticsearch

class ESReranker(BaseEmbedding):
    """
    ESReranker 클래스는 Elasticsearch를 통해 문서의 재랭킹을 수행하는 기능을 제공합니다.
    """
    
    def __init__(self, config, all_config):
        """
        클래스의 생성자. Elasticsearch 클라이언트를 초기화하고 모델 설정을 구성합니다.
        
        :param config: 모델의 설정 정보
        :param all_config: Elasticsearch 연결에 필요한 전체 설정
        """
        super().__init__()  # 부모 클래스(BaseEmbedding)의 초기화 호출

        host = all_config["es"]["host"]  # Elasticsearch 호스트 주소
        id = all_config["es"]["id"]      # 인증 ID
        pw = all_config["es"]["pw"]      # 인증 비밀번호
        self.es = Elasticsearch(
            [host],                       # 외부 Elasticsearch 호스트
            basic_auth=(id, pw)          # 기본 인증
        )
        self.serving_model = config.get("serving_model")  # 사용될 모델 이름

    def compute_score(self, query, text):
        """
        주어진 쿼리와 텍스트에 대해 단일 재랭킹 점수를 계산합니다.
        
        :param query: 쿼리 텍스트
        :param text: 평가할 텍스트(문자열 또는 문자열 리스트)
        :return: 계산된 재랭킹 점수 리스트
        """
        if not self.serving_model:
            raise Exception("Serving model not specified.")  # 모델이 지정되지 않은 경우 예외 발생
        
        if isinstance(text, str):
            text = [text]  # 문자열이면 리스트로 변환
        elif isinstance(text, list):
            text = text  # 리스트인 경우 그대로 사용
        else:
            raise TypeError("text must be either a string or a list.")  # 타입 에러
        
        body = {
            "query": query,
            "input": text,
            "task_settings": {"return_documents": False},
        }
        
        # Elasticsearch에 쿼리 전송 및 응답 받기
        response = self.es.inference.inference(inference_id=self.serving_model, body=body)
        score = response.body['rerank'][0]['relevance_score']  # 응답에서 점수 추출
        return [score]  # 점수 리스트 반환
    
    def compute_scores(self, query, text):
        """
        주어진 쿼리와 텍스트 리스트에 대해 재랭킹 점수들을 계산합니다.
        
        :param query: 쿼리 텍스트
        :param text: 평가할 텍스트 리스트
        :return: 각 텍스트에 대한 계산된 점수 리스트
        """
        if not self.serving_model:
            raise Exception("Serving model not specified.")  # 모델이 지정되지 않은 경우 예외 발생
        
        if isinstance(text, list):
            text = text  # 리스트인 경우 그대로 사용
        else:
            raise TypeError("text must be a list.")  # 타입 에러
        
        body = {
            "query": query,
            "input": text,
            "task_settings": {"return_documents": False},
        }
        
        # Elasticsearch에 쿼리 전송 및 응답 받기
        response = self.es.inference.inference(inference_id=self.serving_model, body=body)
        
        # 점수 리스트 초기화
        scores = [0] * len(text)  # 텍스트 길이에 맞춰 초기화
        for rerank in response.body['rerank']:
            scores[rerank['index']] = rerank['relevance_score']  # 점수 할당
        return scores  # 점수 리스트 반환
    
    def compute_score_with_docs_and_scores(self, query, docs_and_scores):
        """
        주어진 쿼리와 문서 및 점수 리스트를 기반으로 재랭킹 점수를 계산합니다.
        
        :param query: 쿼리 텍스트
        :param docs_and_scores: 문서와 점수의 튜플 리스트
        :return: 문서와 관련된 점수 리스트
        """
        if not self.serving_model:
            raise Exception("Serving model not specified.")  # 모델이 지정되지 않은 경우 예외 발생
        
        if isinstance(docs_and_scores, str):
            docs_and_scores = [docs_and_scores]  # 문자열이면 리스트로 변환
        elif isinstance(docs_and_scores, list):
            docs_and_scores = docs_and_scores  # 리스트인 경우 그대로 사용
        else:
            raise TypeError("docs_and_scores must be either a string or a list.")  # 타입 에러
        
        body = {
            "query": query,
            "input": list(map(lambda x: x[0].page_content, docs_and_scores)),  # 문서 내용만 추출
            "task_settings": {"return_documents": False},
        }
        
        # Elasticsearch에 쿼리 전송 및 응답 받기
        response = self.es.inference.inference(inference_id=self.serving_model, body=body)
        
        semantic_search_result = []
        for document in response.body["rerank"]:
            # 결과 문서와 점수를 리스트에 추가
            semantic_search_result.append(
                (docs_and_scores[document['_index']], document['relevance_score'])
            ) 
        
        return semantic_search_result  # 문서와 점수의 결과 리스트 반환