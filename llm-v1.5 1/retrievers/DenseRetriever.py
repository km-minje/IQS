from retrievers.BaseRetriever import BaseRetrieverWithScores
from typing import List, Optional
import numpy as np
import faiss
import pickle
import os
from langchain_core.documents import Document
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class FaissRetrieverWithScores(BaseRetrieverWithScores):
    """
    Faiss 기반의 검색기를 구현한 클래스.
    주어진 쿼리에 대해 유사도 검색을 수행하며 점수를 계산합니다.
    """

    def __init__(self, faiss, k, instruction=False):
        """
        FaissRetrieverWithScores 초기화.

        :param faiss: Faiss 인스턴스
        :param k: 최상위 k개의 검색할 문서 수
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        """
        super().__init__(k=k)  # BaseRetrieverWithScores 초기화
        self._faiss = faiss  # Faiss 인스턴스 저장
        self._instruction = instruction  # 지침 플래그 저장
        if self._instruction:
            self._faiss.embedding_function.model.query_instruction_for_retrieval = "Represent this sentence for searching relevant passages:"  # 쿼리 지침 설정
        self._update_queries = False  # 쿼리 업데이트 여부 초기화

    def update_querie_embeddings(self, queries):
        """
        주어진 쿼리에 대한 임베딩을 업데이트합니다.

        :param queries: 업데이트할 쿼리 목록
        """
        self._update_queries = True  # 업데이트 플래그 설정
        self._query_embeddings = {}  # 쿼리 임베딩 초기화
        for i, v in enumerate(
            self._faiss.embedding_function.model.encode_queries(queries)
        ):
            self._query_embeddings[queries[i]] = v  # 쿼리 임베딩 저장

    def similarity_search_with_score(self, query, k):
        """
        주어진 쿼리에 대해 유사도 검색을 수행하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 문서 수
        :return: 문서와 점수의 리스트
        """
        if self._instruction and self._update_queries:
            q_embedding = self._query_embeddings[query].astype(
                np.float32
            )  # 쿼리 임베딩 가져오기
            return self._faiss.similarity_search_with_score_by_vector(
                embedding=q_embedding, k=k
            )  # Faiss를 통해 유사도 검색 수행
        else:
            return self._faiss.similarity_search_with_score(
                query=query, k=k
            )  # 일반 검색 수행


class CustomFaissIndexRetrieverWithScores:
    """
    사용자 정의 Faiss 인덱스를 기반으로 한 검색기 클래스
    """

    def __init__(self, faiss_index, docs, embed_model, instruction=False):       
        """
        CustomFaissIndexRetrieverWithScores 초기화.

        :param faiss_index: Faiss 인덱스
        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        """
        self._faiss_index = faiss_index  # Faiss 인덱스 저장
        self._instruction = instruction  # 지침 플래그 저장
        self._embed_model = embed_model  # 임베딩 모델 저장
        self._docs = docs  # 문서 목록 저장
        self.pre_sentence = ""
        if self._instruction:
            self.pre_sentence = str(
                "Represent this sentence for searching relevant passages:"
            )  # 쿼리 지침 설정
        self._update_queries = False  # 쿼리 업데이트 여부 초기화

    def update_querie_embeddings(self, queries):
        """
        주어진 쿼리에 대한 임베딩을 업데이트합니다.

        :param queries: 업데이트할 쿼리 목록
        """
        self._update_queries = True  # 업데이트 플래그 설정
        if self._instruction:
            updated_queries = [self.pre_sentence + q for q in queries]  # 지침 추가
        else:
            updated_queries = queries  # 그대로 사용
        self._query_embeddings = {}  # 쿼리 임베딩 초기화
        for i, v in enumerate(self._embed_model.embed_documents(updated_queries)):
            self._query_embeddings[queries[i]] = np.array(v)  # 쿼리 임베딩 저장

    @classmethod
    def from_documents(cls, docs, embed_model, instruction):
        """
        문서 목록에서 Faiss 인덱스를 생성합니다.

        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        :return: CustomFaissIndexRetrieverWithScores 인스턴스
        """
        _docs = docs  # 문서 저장
        _embed_model = embed_model  # 임베딩 모델 저장
        corpus = [doc.page_content for doc in docs]  # 문서 내용 수집
        batch_size = 128  # 배치 크기 설정
        # corpus_embeddings = []  # 문서 임베딩 초기화

        # # 배치 단위로 문서 임베딩 생성
        # for i in range(0, len(corpus), batch_size):
        #     temp_embeddings = _embed_model.embed_documents(corpus[i : i + batch_size])
        #     corpus_embeddings.extend(temp_embeddings)  # 임베딩 추가
        ## colbert 안쓰기
        corpus_embeddings = _embed_model.embed_documents(corpus)
        corpus_embeddings = np.array(corpus_embeddings)  # numpy 배열로 변환
        corpus_embeddings = corpus_embeddings.astype(np.float32)  # float32로 변환

        dim = corpus_embeddings.shape[-1]  # 임베딩 차원
        _faiss_index = faiss.index_factory(
            dim, "Flat", faiss.METRIC_INNER_PRODUCT
        )  # Faiss 인덱스 생성
        _faiss_index.train(corpus_embeddings)  # 훈련 데이터로 인덱스 훈련
        _faiss_index.add(corpus_embeddings)  # 임베딩 추가

        return cls(
            faiss_index=_faiss_index,
            docs=_docs,
            embed_model=_embed_model,
            instruction=instruction,
        )

    def similarity_search_with_score(self, query, k):
        """
        주어진 쿼리에 대해 유사도 검색을 수행하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 문서 수
        :return: 문서와 점수의 리스트
        """
        if self._update_queries:
            q_embedding = self._query_embeddings[query].astype(
                np.float32
            )  # 쿼리 임베딩 가져오기
            q_embedding = q_embedding.reshape(-1, q_embedding.shape[-1])  # 형태 변환

            D, I = self._faiss_index.search(q_embedding, k)  # Faiss로 검색 수행
            results = []
            for idx, score in zip(I[0], D[0]):  # 검색된 결과 정리
                doc = self._docs[idx]
                results.append((doc, float(score)))
            return results

        else:
            q_embedding = self._embed_model.embed_query(query)  # 쿼리 임베딩 생성
            q_embedding = np.array(q_embedding).astype(np.float32)
            q_embedding = q_embedding.reshape(-1, q_embedding.shape[-1])  # 형태 변환
            D, I = self._faiss_index.search(q_embedding, k)  # Faiss로 검색 수행
            results = []
            for idx, score in zip(I[0], D[0]):  # 검색된 결과 정리
                doc = self._docs[idx]
                results.append((doc, float(score)))
            return results

    def save(self, path: str):
        """
        Faiss 인덱스와 메타데이터를 저장합니다.

        :param path: 저장할 경로
        """
        faiss.write_index(self._faiss_index, f"{os.path.join(path, 'index.faiss')}")
        with open(f"{os.path.join(path,'meta.pkl')}", "wb") as f:
            pickle.dump(
                {
                    "instruction": self._instruction,
                    "docs": self._docs,
                    "update_queries": self._update_queries,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, embed_model):
        """
        저장된 Faiss 인덱스와 메타데이터를 로드합니다.

        :param path: 로드할 경로
        :param embed_model: 임베딩 모델
        :return: CustomFaissIndexRetrieverWithScores 인스턴스
        """
        _faiss_index = faiss.read_index(f"{os.path.join(path, 'index.faiss')}")
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        t_cls = cls(
            faiss_index=_faiss_index,
            docs=meta["docs"],
            embed_model=embed_model,
            instruction=meta["instruction"],
        )
        t_cls._update_queries = meta["update_queries"]  # 업데이트 플래그 설정

        return t_cls


class CustomColBERTRetrieverWithScores:
    """
    사용자 정의 colbert 기반으로 한 검색기 클래스
    """

    def __init__(self, doc_output, docs, embed_model, instruction=False):        
        """
        CustomFaissIndexRetrieverWithScores 초기화.

        :param doc_output: 임베딩한 문서 목록
        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        """
        self._instruction = instruction  # 지침 플래그 저장
        self._embed_model = embed_model  # 임베딩 모델 저장
        self._docs = docs  # 문서 목록 저장
        self._doc_outputs = doc_output
        self.pre_sentence = ""
        if self._instruction:
            self.pre_sentence = str(
                "Represent this sentence for searching relevant passages:"
            )  # 쿼리 지침 설정
        self._update_queries = False  # 쿼리 업데이트 여부 초기화

    def update_querie_embeddings(self, queries):
        """
        주어진 쿼리에 대한 임베딩을 업데이트합니다.

        :param queries: 업데이트할 쿼리 목록
        """
        self._update_queries = True  # 업데이트 플래그 설정
        if self._instruction:
            updated_queries = [self.pre_sentence + q for q in queries]  # 지침 추가
        else:
            updated_queries = queries  # 그대로 사용
        self._query_embeddings = {}  # 쿼리 임베딩 초기화
        for i, v in enumerate(self._embed_model.embed_documents(updated_queries)):
            self._query_embeddings[queries[i]] = np.array(v)  # 쿼리 임베딩 저장

    @classmethod
    def from_documents(cls, docs, embed_model, instruction):
        """
        문서 목록에서 Faiss 인덱스를 생성합니다.

        :param doc_output: 임베딩한 문서 목록
        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        :return: CustomFaissIndexRetrieverWithScores 인스턴스
        """
        _docs = docs  # 문서 저장
        _embed_model = embed_model  # 임베딩 모델 저장
        corpus = [doc.page_content for doc in docs]  # 문서 내용 수집
        batch_size = 128  # 배치 크기 설정
        # corpus_embeddings = []  # 문서 임베딩 초기화

        # # 배치 단위로 문서 임베딩 생성
        # for i in range(0, len(corpus), batch_size):
        #     temp_embeddings = _embed_model.embed_documents(corpus[i : i + batch_size])
        #     corpus_embeddings.extend(temp_embeddings)  # 임베딩 추가
        _doc_output = _embed_model.embed_documents(corpus)

        return cls(
            doc_output = _doc_output,
            docs=_docs,
            embed_model=_embed_model,
            instruction=instruction,
        )

    def similarity_search_with_score(self, query, k):
        """
        주어진 쿼리에 대해 유사도 검색을 수행하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 문서 수
        :return: 문서와 점수의 리스트
        """
        if self._update_queries:
            q_embedding = self._query_embeddings[query].astype(
                np.float32
            )  # 쿼리 임베딩 가져오기
            q_embedding = q_embedding.reshape(-1, q_embedding.shape[-1])  # 형태 변환

            D, I = self._faiss_index.search(q_embedding, k)  # Faiss로 검색 수행
            results = []
            for idx, score in zip(I[0], D[0]):  # 검색된 결과 정리
                doc = self._docs[idx]
                results.append((doc, float(score)))
            return results

        else:
            q_embedding = self._embed_model.embed_query(query)["colbert_vecs"][0]
            scores = []
            for i, doc_vec in enumerate(self._doc_outputs["colbert_vecs"]):
                score = self._embed_model.colbert_score(q_embedding, doc_vec).item()
                scores.append((self._docs[i], score))
            results = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
            return results
    
    def save(self, path: str):
        """
        Faiss 인덱스와 메타데이터를 저장합니다.

        :param path: 저장할 경로
        """
        faiss.write_index(self._faiss_index, f"{os.path.join(path, 'index.faiss')}")
        with open(f"{os.path.join(path,'meta.pkl')}", "wb") as f:
            pickle.dump(
                {
                    "instruction": self._instruction,
                    "docs": self._docs,
                    "update_queries": self._update_queries,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, embed_model):
        """
        저장된 Faiss 인덱스와 메타데이터를 로드합니다.

        :param path: 로드할 경로
        :param embed_model: 임베딩 모델
        :return: CustomFaissIndexRetrieverWithScores 인스턴스
        """
        _faiss_index = faiss.read_index(f"{os.path.join(path, 'index.faiss')}")
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        t_cls = cls(
            faiss_index=_faiss_index,
            docs=meta["docs"],
            embed_model=embed_model,
            instruction=meta["instruction"],
        )
        t_cls._update_queries = meta["update_queries"]  # 업데이트 플래그 설정

        return t_cls


class CustomSparseRetrieverWithScores:
    """
    사용자 정의 colbert 기반으로 한 검색기 클래스
    """

    def __init__(self, doc_output, vectorizer, docs, embed_model, instruction=False):        
        """
        CustomFaissIndexRetrieverWithScores 초기화.

        :param doc_output: 임베딩한 문서 목록
        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        """
        self._instruction = instruction  # 지침 플래그 저장
        self._embed_model = embed_model  # 임베딩 모델 저장
        self._docs = docs  # 문서 목록 저장
        self._doc_outputs = doc_output
        self._vectorizer = vectorizer
        self.pre_sentence = ""
        # if self._instruction:
        #     self.pre_sentence = str(
        #         "Represent this sentence for searching relevant passages:"
        #     )  # 쿼리 지침 설정
        self._update_queries = False  # 쿼리 업데이트 여부 초기화

    def update_querie_embeddings(self, queries):
        """
        주어진 쿼리에 대한 임베딩을 업데이트합니다.

        :param queries: 업데이트할 쿼리 목록
        """
        self._update_queries = True  # 업데이트 플래그 설정
        if self._instruction:
            updated_queries = [self.pre_sentence + q for q in queries]  # 지침 추가
        else:
            updated_queries = queries  # 그대로 사용
        self._query_embeddings = {}  # 쿼리 임베딩 초기화
        for i, v in enumerate(self._embed_model.embed_documents(updated_queries)):
            self._query_embeddings[queries[i]] = np.array(v)  # 쿼리 임베딩 저장

    @classmethod
    def from_documents(cls, docs, embed_model, instruction):
        """
        문서 목록에서 Faiss 인덱스를 생성합니다.

        :param doc_output: 임베딩한 문서 목록
        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        :return: CustomFaissIndexRetrieverWithScores 인스턴스
        """
        _docs = docs  # 문서 저장
        _embed_model = embed_model  # 임베딩 모델 저장
        corpus = [doc.page_content for doc in docs]  # 문서 내용 수집
        batch_size = 128  # 배치 크기 설정
        
        _doc_output = _embed_model.embed_documents(corpus)["lexical_weights"]
        _vectorizer = DictVectorizer(sparse=True)
        _doc_output = _vectorizer.fit_transform(_doc_output)
        return cls(
            vectorizer = _vectorizer,
            doc_output = _doc_output,
            docs=_docs,
            embed_model=_embed_model,
            instruction=instruction,
        )

    def similarity_search_with_score(self, query, k):
        """
        주어진 쿼리에 대해 유사도 검색을 수행하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 문서 수
        :return: 문서와 점수의 리스트
        """
        if self._update_queries:
            q_embedding = self._query_embeddings[query].astype(
                np.float32
            )  # 쿼리 임베딩 가져오기
            q_embedding = q_embedding.reshape(-1, q_embedding.shape[-1])  # 형태 변환

            D, I = self._faiss_index.search(q_embedding, k)  # Faiss로 검색 수행
            results = []
            for idx, score in zip(I[0], D[0]):  # 검색된 결과 정리
                doc = self._docs[idx]
                results.append((doc, float(score)))
            return results

        else:
            q_embedding = self._embed_model.embed_query(query)["lexical_weights"]
            q_embedding = self._vectorizer.transform(q_embedding)
            results = []
            total_score = cosine_similarity(q_embedding, self._doc_outputs).flatten()
            top_indices = total_score.argsort()[::-1][:k]
            for i in top_indices:
                results.append((self._docs[i], total_score[i]))
            return results
    
    def save(self, path: str):
        """
        Faiss 인덱스와 메타데이터를 저장합니다.

        :param path: 저장할 경로
        """
        faiss.write_index(self._faiss_index, f"{os.path.join(path, 'index.faiss')}")
        with open(f"{os.path.join(path,'meta.pkl')}", "wb") as f:
            pickle.dump(
                {
                    "instruction": self._instruction,
                    "docs": self._docs,
                    "update_queries": self._update_queries,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, embed_model):
        """
        저장된 Faiss 인덱스와 메타데이터를 로드합니다.

        :param path: 로드할 경로
        :param embed_model: 임베딩 모델
        :return: CustomFaissIndexRetrieverWithScores 인스턴스
        """
        _faiss_index = faiss.read_index(f"{os.path.join(path, 'index.faiss')}")
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        t_cls = cls(
            faiss_index=_faiss_index,
            docs=meta["docs"],
            embed_model=embed_model,
            instruction=meta["instruction"],
        )
        t_cls._update_queries = meta["update_queries"]  # 업데이트 플래그 설정

        return t_cls


class FaissIndexRetrieverWithScores:
    """
    Faiss 인덱스를 사용하여 검색을 수행하는 클래스.
    """

    def __init__(self, faiss_index, docs, embed_model, instruction=False):
        """
        FaissIndexRetrieverWithScores 초기화.

        :param faiss_index: Faiss 인덱스
        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        """
        self._faiss_index = faiss_index  # Faiss 인덱스 저장
        self._instruction = instruction  # 지침 플래그 저장
        self._embed_model = embed_model  # 임베딩 모델 저장
        self._docs = docs  # 문서 목록 저장
        if self._instruction:
            self._embed_model.model.query_instruction_for_retrieval = "Represent this sentence for searching relevant passages:"  # 쿼리 지침 설정
        self._update_queries = False  # 쿼리 업데이트 여부 초기화

    def update_querie_embeddings(self, queries):
        """
        주어진 쿼리에 대한 임베딩을 업데이트합니다.

        :param queries: 업데이트할 쿼리 목록
        """
        self._update_queries = True  # 업데이트 플래그 설정
        self._query_embeddings = {}  # 쿼리 임베딩 초기화
        for i, v in enumerate(self._embed_model.model.encode_queries(queries)):
            self._query_embeddings[queries[i]] = v  # 쿼리 임베딩 저장

    @classmethod
    def from_documents(cls, docs, embed_model, instruction):
        """
        문서 목록에서 Faiss 인덱스를 생성합니다.

        :param docs: 문서 목록
        :param embed_model: 임베딩 모델
        :param instruction: 쿼리에 대한 지침을 사용할지 여부
        :return: FaissIndexRetrieverWithScores 인스턴스
        """
        _docs = docs  # 문서 저장
        _embed_model = embed_model  # 임베딩 모델 저장
        corpus = [doc.page_content for doc in docs]  # 문서 내용 수집
        batch_size = 512  # 배치 크기 설정
        corpus_embeddings = []  # 문서 임베딩 초기화

        # 배치 단위로 문서 임베딩 생성
        for i in range(0, len(corpus), batch_size):
            temp_embeddings = _embed_model.model.encode(corpus[i : i + batch_size])
            corpus_embeddings.extend(temp_embeddings)  # 임베딩 추가

        corpus_embeddings = np.array(corpus_embeddings)  # numpy 배열로 변환
        corpus_embeddings = corpus_embeddings.astype(np.float32)  # float32로 변환
        dim = corpus_embeddings.shape[-1]  # 임베딩 차원

        _faiss_index = faiss.index_factory(
            dim, "Flat", faiss.METRIC_INNER_PRODUCT
        )  # Faiss 인덱스 생성
        _faiss_index.train(corpus_embeddings)  # 훈련 데이터로 인덱스 훈련
        _faiss_index.add(corpus_embeddings)  # 임베딩 추가

        return cls(
            faiss_index=_faiss_index,
            docs=_docs,
            embed_model=_embed_model,
            instruction=instruction,
        )

    def similarity_search_with_score(self, query, k):
        """
        주어진 쿼리에 대해 유사도 검색을 수행하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 문서 수
        :return: 문서와 점수의 리스트
        """
        if self._instruction and self._update_queries:
            q_embedding = self._query_embeddings[query].astype(
                np.float32
            )  # 쿼리 임베딩 가져오기
            q_embedding = q_embedding.reshape(-1, q_embedding.shape[-1])  # 형태 변환

            D, I = self._faiss_index.search(q_embedding, k)  # Faiss로 검색 수행
            results = []
            for idx, score in zip(I[0], D[0]):  # 검색된 결과 정리
                doc = self._docs[idx]
                results.append((doc, float(score)))
            return results
        else:
            q_embedding = self._embed_model.model.encode_queries(
                query
            )  # 쿼리 임베딩 생성
            q_embedding = q_embedding.reshape(-1, q_embedding.shape[-1])  # 형태 변환

            D, I = self._faiss_index.search(q_embedding, k)  # Faiss 검색 수행
            results = []
            for idx, score in zip(I[0], D[0]):  # 검색된 결과 정리
                doc = self._docs[idx]
                results.append((doc, float(score)))
            return results

    def save(self, path: str):
        """
        Faiss 인덱스와 메타데이터를 저장합니다.

        :param path: 저장할 경로
        """
        faiss.write_index(self._faiss_index, f"{path}_index.faiss")  # Faiss 인덱스 저장
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "instruction": self._instruction,
                    "docs": self._docs,
                    "update_queries": self._update_queries,
                },
                f,
            )  # 메타데이터 저장

    @classmethod
    def load(cls, path: str, embed_model):
        """
        저장된 Faiss 인덱스와 메타데이터를 로드합니다.

        :param path: 로드할 경로
        :param embed_model: 임베딩 모델
        :return: FaissIndexRetrieverWithScores 인스턴스
        """
        _faiss_index = faiss.read_index(f"{path}_index.faiss")  # 인덱스 로드
        with open(f"{path}_meta.pkl", "rb") as f:
            meta = pickle.load(f)  # 메타데이터 로드

        t_cls = cls(
            faiss_index=_faiss_index,
            docs=meta["docs"],
            embed_model=embed_model,
            instruction=meta["instruction"],
        )
        t_cls._update_queries = meta["update_queries"]  # 업데이트 플래그 설정

        return t_cls


class ESKNNRetrieverWithScores:
    """
    Elasticsearch K-Nearest Neighbor (KNN) 검색기를 구현한 클래스.
    """

    def __init__(self, es, index, target_field, embedding_model, serving_model):
        """
        ESKNNRetrieverWithScores 초기화.

        :param es: Elasticsearch 인스턴스
        :param index: 검색할 인덱스 이름
        :param target_field: 대상 필드 이름
        :param embedding_model: 임베딩 모델
        :param serving_model: 서빙 모델 ID
        """
        self.es = es  # Elasticsearch 인스턴스 저장
        self.index = index  # 인덱스 이름 저장
        self.target_field = target_field  # 대상 필드 이름 저장
        self.embedding_model = embedding_model  # 임베딩 모델 저장
        self.serving_model = serving_model  # 서빙 모델 저장
        self.num_candidates = 10000  # 후보 수 설정

    def similarity_search_with_score(self, query: str, k: Optional[int] = 5):
        """
        주어진 쿼리로 KNN 검색을 수행하고 점수를 반환합니다.

        :param query: 검색할 쿼리
        :param k: 반환할 문서 수 (기본값: 5)
        :return: 문서 및 점수의 리스트
        """
        if self.serving_model:
            response = self.es.search(
                index=self.index,
                size=k,
                knn={
                    "k": k,
                    "num_candidates": self.num_candidates,
                    "field": self.target_field,
                    "query_vector_builder": {
                        self.target_field: {
                            "model_id": self.serving_model,
                            "model_text": query,
                        }
                    },
                },
            )
        elif self.embedding_model:
            response = self.es.search(
                index=self.index,
                knn={
                    "field": self.target_field,
                    "query_vector": self.embedding_model.embed_query(query),
                    "k": k,
                    "num_candidates": self.num_candidates,
                },
            )
        else:
            raise Exception("Either serving_model or embedding_model must be provided.")

        docs_and_scores = []  # 문서 및 점수 저장 리스트
        for rep in response["hits"]["hits"]:  # Elasticsearch 응답에서 결과 추출
            docs_and_scores.append(
                [
                    Document(
                        page_content=f"{rep.get('_source').get('text')}",
                        metadata={"_id": int(rep.get("_id"))},
                    ),
                    rep.get("_score") * 2 - 1,  # 점수 계산 및 저장
                ]
            )

        # 점수를 기준으로 내림차순 정렬
        sorted_docs_and_scores = sorted(
            docs_and_scores, key=lambda x: x[1], reverse=True
        )
        return sorted_docs_and_scores[:k]  # 상위 k개의 문서 및 점수 반환
