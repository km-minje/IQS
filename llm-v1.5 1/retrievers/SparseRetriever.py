from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from typing import List, Optional
import pickle
import os
from langchain_core.documents import Document

class BM25RetrieverWithScores(BM25Retriever):
    """
    BM25 기준으로 문서 검색 점수를 계산하는 검색기 클래스.
    """

    def similarity_search_with_score(self, query, k):
        """
        주어진 쿼리에 대해 BM25 모델을 사용하여 유사 문서를 검색하고 점수를 반환합니다.
        
        :param query: 검색할 쿼리
        :param k: 반환할 상위 문서 수
        :return: 문서와 점수의 리스트
        """
        processed_query = self.preprocess_func(query)  # 쿼리 전처리
        scores = self.vectorizer.get_scores(processed_query)  # 점수 계산
        docs_and_scores = [(doc, score) for doc, score in zip(self.docs, scores)]  # 문서와 점수 매칭
        sorted_docs_and_scores = sorted(
            docs_and_scores, key=lambda x: x[1], reverse=True
        )  # 점수를 기준으로 내림차순 정렬
        return sorted_docs_and_scores[:k]  # 상위 k개 반환


class BM25OkapiRetrieverWithScores:
    """
    BM25Okapi 알고리즘을 사용하여 문서 검색과 점수를 계산하는 검색기 클래스.
    """

    def __init__(self, docs: List[str], preprocess_func):
        """
        BM25OkapiRetrieverWithScores 초기화.
        
        :param docs: 문서 목록
        :param preprocess_func: 문서 전처리 함수
        """
        self.docs = docs  # 문서 저장
        self.preprocess_func = preprocess_func  # 전처리 함수 저장
        self.tokenized_docs = [self.preprocess_func(doc.page_content) for doc in docs]  # 문서 토큰화
        self.bm25 = BM25Okapi(self.tokenized_docs)  # BM25Okapi 모델 초기화

    @classmethod
    def from_documents(cls, documents: List[str], preprocess_func) -> "BM25OkapiRetrieverWithScores":
        """
        주어진 문서 목록에서 인스턴스를 생성합니다.

        :param documents: 문서 목록
        :param preprocess_func: 문서 전처리 함수
        :return: BM25OkapiRetrieverWithScores 인스턴스
        """
        return cls(documents, preprocess_func)

    def similarity_search_with_score(self, query: str, k: Optional[int] = 5):
        """
        주어진 쿼리에 대해 BM25Okapi를 사용하여 유사 문서를 검색하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 상위 문서 수
        :return: 문서와 점수의 리스트
        """
        tokenized_query = self.preprocess_func(query)  # 쿼리 전처리 및 토큰화
        scores = self.bm25.get_scores(tokenized_query)  # BM25 모델을 통해 점수 계산
        docs_and_scores = list(zip(self.docs, scores))  # 문서와 점수 매칭
        sorted_docs_and_scores = sorted(
            docs_and_scores, key=lambda x: x[1], reverse=True
        )  # 점수를 기준으로 내림차순 정렬
        return sorted_docs_and_scores[:k]  # 상위 k개 반환

    def save(self, path: str):
        """
        BM25 모델의 문서와 토큰화된 문서를 저장합니다.

        :param path: 저장할 경로
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)  # 경로가 없으면 생성

        with open(os.path.join(path, "bm25.pickle"), "wb") as f:
            pickle.dump(
                {
                    "docs": self.docs,  # 문서 저장
                    "tokenized_docs": self.tokenized_docs,  # 토큰화된 문서 저장
                },
                f,
            )

    @classmethod
    def load(cls, path: str, preprocess_func):
        """
        저장된 BM25 데이터 및 메타데이터를 로드합니다.

        :param path: 로드할 경로
        :param preprocess_func: 문서 전처리 함수
        :return: BM25OkapiRetrieverWithScores 인스턴스
        """
        with open(os.path.join(path, "bm25.pickle"), "rb") as f:
            data = pickle.load(f)  # 저장된 데이터 로드

        # 로드된 문서와 전처리 함수로 BM25 인스턴스 재생성
        obj = cls.__new__(cls)
        obj.docs = data["docs"]  # 문서 할당
        obj.preprocess_func = preprocess_func  # 전처리 함수 할당
        obj.tokenized_docs = data["tokenized_docs"]  # 토큰화된 문서 할당
        obj.bm25 = BM25Okapi(obj.tokenized_docs)  # BM25Okapi 모델 재생성

        return obj  # 인스턴스 반환


class ESBM25RetrieverWithScores:
    """
    Elasticsearch 기반 BM25 검색기의 구현. 
    """

    def __init__(self, es, index, target_field, tokenizer_func):
        """
        ESBM25RetrieverWithScores 초기화.
        
        :param es: Elasticsearch 인스턴스
        :param index: 검색할 인덱스 이름
        :param target_field: 대상 필드 이름
        :param tokenizer_func: 쿼리 토크나이저 함수
        """
        self.es = es  # Elasticsearch 인스턴스 저장
        self.index = index  # 인덱스 이름 저장
        self.target_field = target_field  # 대상 필드 이름 저장
        self.tokenizer_func = tokenizer_func  # 토크나이저 함수 저장

    def similarity_search_with_score(self, query: str, k: Optional[int] = 5):
        """
        주어진 쿼리에 대해 Elasticsearch BM25를 사용하여 유사 문서를 검색하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 상위 문서 수 (기본값: 5)
        :return: 문서와 점수의 리스트
        """
        # BM25 쿼리 구성
        if not self.tokenizer_func:
            response = self.es.search(index=self.index, size=k, query={"match": {self.target_field: query}})
        else:
            response = self.es.search(index=self.index, size=k, query={"match": {self.target_field: " ".join(self.tokenizer_func(query))}})

        docs_and_scores = []  # 문서와 점수를 저장할 리스트
        for rep in response['hits']['hits']:
            docs_and_scores.append([
                Document(page_content=f"{rep.get('_source').get('text')}", metadata={"_id": int(rep.get('_id'))}),
                rep.get('_score')
            ])

        sorted_docs_and_scores = sorted(
            docs_and_scores, key=lambda x: x[1], reverse=True
        )  # 점수를 기준으로 내림차순 정렬
        return sorted_docs_and_scores[:k]  # 상위 k개 반환