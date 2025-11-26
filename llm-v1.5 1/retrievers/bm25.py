#!/usr/bin/env python

import math
import numpy as np
from multiprocessing import Pool, cpu_count

"""
모든 알고리즘은 다음 논문에서 인용되었습니다:
Trotmam et al, Improvements to BM25 and Language Models Examined

여기서는 BM25의 모든 변형을 구현합니다.
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        """
        BM25 기본 클래스 초기화. 코퍼스와 선택적으로 토크나이저를 설정합니다.
        
        :param corpus: 문서 목록
        :param tokenizer: (선택적) 문서를 토큰화하는 데 사용할 토크나이저
        """
        self.corpus_size = 0  # 문서 수
        self.avgdl = 0  # 평균 문서 길이
        self.doc_freqs = []  # 문서 빈도
        self.idf = {}  # 역문서 빈도 (IDF)
        self.doc_len = []  # 각 문서의 길이
        self.tokenizer = tokenizer  # 토크나이저 저장

        # 토크나이저가 지정된 경우 코퍼스 토큰화
        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)  # 코퍼스 초기화
        self._calc_idf(nd)  # IDF 계산

    def _initialize(self, corpus):
        """
        코퍼스를 초기화하고 각 단어의 문서 빈도를 계산합니다.
        
        :param corpus: 문서 목록
        :return: 단어와 문서 수의 딕셔너리
        """
        nd = {}  # word -> 문서 수
        num_doc = 0  # 총 단어 수
        for document in corpus:
            self.doc_len.append(len(document))  # 각 문서의 길이 저장
            num_doc += len(document)  # 총 단어 수 증가

            frequencies = {}  # 단어 빈도 기록 용도
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1  # 단어 등장 횟수 카운트
            self.doc_freqs.append(frequencies)  # 문서 빈도 리스트에 추가

            # 각 단어의 문서 수 계산
            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1
            
            self.corpus_size += 1  # 문서 수 증가

        self.avgdl = num_doc / self.corpus_size  # 평균 문서 길이 계산
        return nd  # 단어와 문서 수 반환

    def _tokenize_corpus(self, corpus):
        """
        주어진 코퍼스를 토큰화합니다.
        
        :param corpus: 문서 목록
        :return: 토큰화된 문서
        """
        pool = Pool(cpu_count())  # CPU 수에 맞춰 프로세스 풀 생성
        tokenized_corpus = pool.map(self.tokenizer, corpus)  # 병렬로 문서 토큰화
        return tokenized_corpus

    def _calc_idf(self, nd):
        """
        IDF를 계산하는 메서드. 구체적인 구현은 하위 클래스에 정의되어야 합니다.
        
        :param nd: 단어와 문서 수의 딕셔너리
        """
        raise NotImplementedError()

    def get_scores(self, query):
        """
        주어진 쿼리를 기반으로 문서 점수를 계산합니다.
        구체적인 구현은 하위 클래스에 정의되어야 합니다.

        :param query: 검색할 쿼리
        :return: 문서 점수 배열
        """
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        """
        주어진 쿼리와 문서 ID의 배치 점수를 계산합니다.
        구체적인 구현은 하위 클래스에 정의되어야 합니다.

        :param query: 검색할 쿼리
        :param doc_ids: 문서 ID 목록
        :return: 문서 점수 배열
        """
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        """
        주어진 쿼리에 대해 상위 N개의 관련 문서를 반환합니다.
        
        :param query: 검색할 쿼리
        :param documents: 문서 목록
        :param n: 반환할 문서 수
        :return: 상위 N개 문서
        """
        assert self.corpus_size == len(documents), "주어진 문서가 색인 코퍼스와 일치하지 않습니다!"

        scores = self.get_scores(query)  # 쿼리에 대한 점수 계산
        top_n = np.argsort(scores)[::-1][:n]  # 상위 N개의 문서 인덱스
        return [documents[i] for i in top_n]  # 상위 N개의 문서 반환


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        """
        BM25Okapi 알고리즘 초기화.
        
        :param corpus: 문서 목록
        :param tokenizer: (선택적) 문서를 토큰화하는 데 사용할 토크나이저
        :param k1: 문서 길이 조정 매개변수
        :param b: 문서 길이 비율 조정 매개변수
        :param epsilon: IDF의 최소값을 설정하는 매개변수
        """
        self.k1 = k1  # 문서 길이 팩터
        self.b = b  # 문서 길이 비율 조정 변수
        self.epsilon = epsilon  # IDF의 하한선 설정
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        문서에서 용어 빈도를 계산하고 IDF를 설정합니다.
        IDF 값에 epsilon 기반의 하한선을 설정합니다.
        
        :param nd: 단어와 문서 수의 딕셔너리
        """
        idf_sum = 0  # IDF 합계
        negative_idfs = []  # 음수 IDF 단어 저장
        for word, freq in nd.items():
            idf = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf  # IDF 계산
            idf_sum += idf  # IDF 합계 갱신
            if idf < 0:  # 음수인 경우
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)  # 평균 IDF 계산

        eps = self.epsilon * self.average_idf  # epsilon으로 IDF의 하한 계산
        for word in negative_idfs:  # 음수 IDF 단어에 대해 하한 설정
            self.idf[word] = eps

    def get_scores(self, query):
        """
        쿼리에 대한 BM25Okapi 점수 계산. 
        IDF 값을 로그를 사용하여 계산하며 음수인 경우 epsilon 값을 추가합니다.
        
        :param query: 검색할 쿼리
        :return: 문서 점수 배열
        """
        score = np.zeros(self.corpus_size)  # 점수 배열 초기화
        doc_len = np.array(self.doc_len)  # 문서 길이 배열
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])  # 각 문서에서 쿼리 용어 빈도
            score += 2.5 * (self.idf.get(q) or 0) * (q_freq) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        return score  # 점수 반환

    def get_batch_scores(self, query, doc_ids):
        """
        쿼리와 문서 ID의 배치 점수를 계산합니다.
        
        :param query: 검색할 쿼리
        :param doc_ids: 문서 ID 목록
        :return: 문서 점수 배열
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)  # 문서 ID 유효성 검사
        score = np.zeros(len(doc_ids))  # 점수 배열 초기화
        doc_len = np.array(self.doc_len)[doc_ids]  # 선택한 문서의 길이 배열
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])  # 쿼리 용어 빈도
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()  # 점수 리스트 반환


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        """
        BM25L 알고리즘 초기화.
        
        :param corpus: 문서 목록
        :param tokenizer: (선택적) 문서를 토큰화하는 데 사용할 토크나이저
        :param k1: 문서 길이 조정 매개변수
        :param b: 문서 길이 비율 조정 매개변수
        :param delta: 조정하는 매개변수
        """
        self.k1 = k1  # 문서 길이 팩터
        self.b = b  # 문서 길이 비율 조정 변수
        self.delta = delta  # 조정 매개변수
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        IDF를 계산합니다.
        
        :param nd: 단어와 문서 수의 딕셔너리
        """
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf  # IDF 계산 후 저장

    def get_scores(self, query):
        """
        쿼리에 대한 BM25L 점수 계산.
        
        :param query: 검색할 쿼리
        :return: 문서 점수 배열
        """
        score = np.zeros(self.corpus_size)  # 점수 배열 초기화
        doc_len = np.array(self.doc_len)  # 문서 길이 배열
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])  # 각 문서에서 쿼리 용어 빈도
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)  # 정규화된 빈도 계산
            score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)  # BM25L 점수 계산
        return score  # 점수 반환

    def get_batch_scores(self, query, doc_ids):
        """
        쿼리와 문서 ID의 배치 점수를 계산합니다.
        
        :param query: 검색할 쿼리
        :param doc_ids: 문서 ID 목록
        :return: 문서 점수 배열
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)  # 문서 ID 유효성 검사
        score = np.zeros(len(doc_ids))  # 점수 배열 초기화
        doc_len = np.array(self.doc_len)[doc_ids]  # 선택한 문서의 길이 배열
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])  # 쿼리 용어 빈도
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)  # 정규화된 빈도 계산
            score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)  # BM25L 점수 계산
        return score.tolist()  # 점수 리스트 반환


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        """
        BM25Plus 알고리즘 초기화.
        
        :param corpus: 문서 목록
        :param tokenizer: (선택적) 문서를 토큰화하는 데 사용할 토크나이저
        :param k1: 문서 길이 조정 매개변수
        :param b: 문서 길이 비율 조정 매개변수
        :param delta: 조정하는 매개변수
        """
        self.k1 = k1  # 문서 길이 팩터
        self.b = b  # 문서 길이 비율 조정 변수
        self.delta = delta  # 조정 매개변수
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        IDF를 계산합니다.
        
        :param nd: 단어와 문서 수의 딕셔너리
        """
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)  # IDF 계산
            self.idf[word] = idf  # IDF 저장

    def get_scores(self, query):
        """
        쿼리에 대한 BM25Plus 점수 계산.
        
        :param query: 검색할 쿼리
        :return: 문서 점수 배열
        """
        score = np.zeros(self.corpus_size)  # 점수 배열 초기화
        doc_len = np.array(self.doc_len)  # 문서 길이 배열
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])  # 각 문서에서 쿼리 용어 빈도
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))  # BM25Plus 점수 계산
        return score  # 점수 반환

    def get_batch_scores(self, query, doc_ids):
        """
        쿼리와 문서 ID의 배치 점수를 계산합니다.
        
        :param query: 검색할 쿼리
        :param doc_ids: 문서 ID 목록
        :return: 문서 점수 배열
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)  # 문서 ID 유효성 검사
        score = np.zeros(len(doc_ids))  # 점수 배열 초기화
        doc_len = np.array(self.doc_len)[doc_ids]  # 선택한 문서의 길이 배열
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])  # 쿼리 용어 빈도
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))  # BM25Plus 점수 계산
        return score.tolist()  # 점수 리스트 반환

# BM25Adpt 및 BM25T 알고리즘은 이전 알고리즘보다 더 복잡합니다.
# 여기서는 점수 계산 전에 용어별 k1 매개변수를 계산합니다.

# class BM25Adpt(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # 알고리즘 전용 매개변수
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
#
# class BM25T(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # 알고리즘 전용 매개변수
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score