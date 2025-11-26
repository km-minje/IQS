from typing import List, Callable, Tuple
from langchain_core.documents import Document
from retrievers.BaseRetriever import BaseRetrieverWithScores
from typing import List, Optional

class EnsembleRetrieverWithScores:
    """
    여러 검색기를 조합하여 검색 결과와 점수를 통합하는 클래스.
    """

    def __init__(self, retrievers: List, weights: List[float], search_type: str, c=60.0):
        """
        EnsembleRetrieverWithScores 초기화.
        
        :param retrievers: 사용될 검색기 목록
        :param weights: 각 검색기에 대한 가중치
        :param search_type: 검색 유형 (BM25 또는 KNN 등)
        :param c: 랭크 조정의 상수
        """
        self._retrievers = retrievers  # 검색기 목록 저장
        self._weights = weights  # 가중치 저장
        self._c = c  # 랭크 조정 상수 저장
        self._search_type = search_type  # 검색 유형 저장

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        주어진 쿼리로부터 여러 검색기를 통해 점수를 계산하여 결과를 반환합니다.
        
        :param query: 검색할 쿼리
        :param k: 반환할 상위 문서 수
        :return: 문서와 점수의 리스트
        """
        all_results = []  # 모든 결과 저장용 리스트

        for retriever, weight in zip(self._retrievers, self._weights):
            results = retriever.similarity_search_with_score(query, k=k)  # 개별 검색기에서 결과 가져오기
            for rank, (doc, score) in enumerate(results, start=1):
                normal_score = score * weight  # 점수에 가중치 적용
                rrf_score = weight / (rank + self._c)  # RRF 점수 계산
                all_results.append((doc, rrf_score))  # 결과 저장

        # 중복 문서 통합 (page_content 기준으로)
        doc_score_dict = {}
        for doc, score in all_results:
            _id = doc.metadata.get("_id")  # 문서 ID 가져오기
            if _id is None:
                raise ValueError("Document must contain '_id' in metadata.")

            if _id in doc_score_dict:
                doc_score_dict[_id]["score"] += score  # 기존 점수 추가
            else:
                doc_score_dict[_id] = {"doc": doc, "score": score}  # 새로운 문서 추가

        # 점수 기준 정렬 후 상위 k개 반환
        sorted_docs_and_scores = sorted(
            [(entry["doc"], entry["score"]) for entry in doc_score_dict.values()],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_docs_and_scores[:k]  # 상위 k개의 문서 반환


class ESEnsembleRetrieverWithScores:
    """
    Elasticsearch 기반 KNN 검색과 BM25 검색을 결합하여 결과를 반환하는 클래스.
    """

    def __init__(self, es, index, target_field, tokenizer, embedding_model, serving_model, weights, c):
        """
        ESEnsembleRetrieverWithScores 초기화.
        
        :param es: Elasticsearch 인스턴스
        :param index: 검색할 인덱스 이름
        :param target_field: 대상 필드 이름
        :param tokenizer: 쿼리 토크나이저
        :param embedding_model: 임베딩 모델
        :param serving_model: 서빙 모델 ID
        :param weights: 각 검색기 가중치
        :param c: 랭크 조정의 상수
        """
        self.es = es  # Elasticsearch 인스턴스 저장
        self.index = index  # 인덱스 이름 저장
        self.target_field = target_field  # 대상 필드 이름 저장
        self.tokenizer = tokenizer  # 토크나이저 저장
        self.embedding_model = embedding_model  # 임베딩 모델 저장
        self.serving_model = serving_model  # 서빙 모델 저장
        self.weights = weights  # 가중치 저장
        self.c = c  # 랭크 조정 상수 저장
        self.num_candidates = 1000  # KNN 후보 수설정

    def similarity_search_with_score(self, query: str, k: Optional[int] = 5):
        """
        주어진 쿼리에 대해 BM25와 KNN을 사용하여 유사도 검색을 수행하고 점수를 반환합니다.
        
        :param query: 검색할 쿼리
        :param k: 반환할 상위 문서 수 (기본값: 5)
        :return: 문서와 점수의 리스트
        """
        # BM25 쿼리 구성
        if not self.tokenizer:
            bm25_query = {"match": {self.target_field[0]: query}}
        else:
            bm25_query = {"match": {self.target_field[0]: " ".join(self.tokenizer(query))}}

        # KNN 쿼리 구성
        if self.serving_model:
            knn_query = {
                "k": k,
                "num_candidates": self.num_candidates,
                "field": self.target_field[1],
                "query_vector_builder": {
                    self.target_field[1]: {
                        "model_id": self.serving_model,
                        "model_text": query
                    }
                }
            }
        elif self.embedding_model:
            knn_query = {
                "field": self.target_field[1],
                "query_vector": self.embedding_model.embed_query(query),
                "k": k,
                "num_candidates": self.num_candidates
            }
        else:
            raise Exception("Either serving_model or embedding_model must be provided.")
        
        doc_score_dict = {}  # 문서 및 점수 저장용 딕셔너리
        # BM25 검색 수행
        response = self.es.search(index=self.index, size=k, query=bm25_query)
        for rank, rep in enumerate(response['hits']['hits'], start=1):
            normal_score = self.weights[0] * rep['_score']  # 가중치 적용 점수
            rrf_score = self.weights[0] / (rank + self.c)  # RRF 점수 계산
            score = rrf_score
            _id = rep.get("_id")  # 문서 ID 가져오기
            if _id in doc_score_dict:
                doc_score_dict[_id]["score"] += score  # 기존 점수 추가
            else:
                doc_score_dict[_id] = {
                    "doc": Document(page_content=f"{rep.get('_source').get('text')}", metadata={"_id": int(rep.get('_id'))}),
                    "score": score
                }

        # KNN 검색 수행
        response = self.es.search(index=self.index, size=k, knn=knn_query)
        for rank, rep in enumerate(response['hits']['hits'], start=1):
            normal_score = self.weights[1] * rep['_score']  # 가중치 적용 점수
            rrf_score = self.weights[1] / (rank + self.c)  # RRF 점수 계산
            score = rrf_score
            _id = rep.get("_id")  # 문서 ID 가져오기
            if _id in doc_score_dict:
                doc_score_dict[_id]["score"] += score  # 기존 점수 추가
            else:
                doc_score_dict[_id] = {
                    "doc": Document(page_content=f"{rep.get('_source').get('text')}", metadata={"_id": int(rep.get('_id'))}),
                    "score": score
                }
        
        # 점수 기준 정렬 후 상위 k개 반환
        sorted_docs_and_scores = sorted(
            [(entry["doc"], entry["score"]) for entry in doc_score_dict.values()],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_docs_and_scores[:k]  # 상위 k개의 문서 반환