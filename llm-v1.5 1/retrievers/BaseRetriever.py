from kiwipiepy import Kiwi
from konlpy.tag import Okt, Kkma

from mecab import MeCab
from kiwi_custom import tokenizer as custom_tokenizer
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

#custom_kiwi, _ = custom_tokenizer.custom_kiwi()  # 사용자 정의 토크나이저 생성
# Define tokenization functions
def kiwi_tokenize(text):
    """
    Kiwipiepy 라이브러리를 사용하여 주어진 텍스트를 토큰화합니다.
    
    :param text: 토큰화할 텍스트
    :return: 토큰 리스트
    """
    text = text.strip().lower()  # 텍스트 전처리
    return [token.form for token in Kiwi().tokenize(text)]  # 토큰화 후 리스트 반환


def custom_kiwi_tokenize(text):
    """
    사용자 정의 Kiwipiepy 토크나이저를 사용하여 텍스트를 토큰화합니다.
    
    :param text: 토큰화할 텍스트
    :return: 토큰 리스트
    """
    # custom_kiwi, _ = custom_tokenizer.custom_kiwi()  # 사용자 정의 토크나이저 생성
    text = text.strip().lower()  # 텍스트 전처리
    return [token.form for token in custom_kiwi.tokenize(text)]  # 토큰화 후 리스트 반환


def okt_tokenize(text):
    """
    Konlpy 라이브러리의 Okt를 사용하여 텍스트를 토큰화합니다.
    
    :param text: 토큰화할 텍스트
    :return: 토큰 리스트
    """
    text = text.strip().lower()  # 텍스트 전처리
    return Okt().morphs(text)  # Okt를 사용하여 형태소 분석 후 리스트 반환


def mecab_tokenize(text):
    """
    MeCab 라이브러리를 사용하여 텍스트를 토큰화합니다.
    
    :param text: 토큰화할 텍스트
    :return: 토큰 리스트
    """
    text = text.strip().lower()  # 텍스트 전처리
    return MeCab().morphs(text)  # MeCab을 사용하여 형태소 분석 후 리스트 반환


def kkma_tokenize(text):
    """
    Kkma 라이브러리를 사용하여 텍스트를 토큰화합니다.
    
    :param text: 토큰화할 텍스트
    :return: 토큰 리스트
    """
    text = text.strip().lower()  # 텍스트 전처리
    return Kkma().morphs(text)  # Kkma를 사용하여 형태소 분석 후 리스트 반환


# Mapping of tokenizer names to functions
tokenizer_map = {
    "kiwi": kiwi_tokenize,              # Kiwipiepy 토크나이저
    "custom_kiwi": custom_kiwi_tokenize,  # 사용자 정의 Kiwipiepy 토크나이저
    "okt": okt_tokenize,                  # Okt 형태소 분석기
    "mecab": mecab_tokenize,              # MeCab 형태소 분석기
    "kkma": kkma_tokenize,                # Kkma 형태소 분석기
}


def get_tokenizer(name: str):
    """
    주어진 이름에 해당하는 토크나이저 함수를 반환합니다.
    
    :param name: 요청된 토크나이저 이름
    :return: 요청된 토크나이저 함수
    :raises ValueError: 지원되지 않는 토크나이저 이름인 경우.
    """
    try:
        return tokenizer_map[name]  # 해당 토크나이저 함수 반환
    except KeyError:
        raise ValueError(
            f"Tokenizer '{name}' is not supported. Choose from: {list(tokenizer_map.keys())}"
        )  # 지원되지 않는 토크나이저 이름일 경우 예외 발생


class BaseRetrieverWithScores(BaseRetriever):
    """
    BaseRetrieverWithScores 클래스는 기본 데이터 검색 기능을 제공하며,
    관련 문서의 점수를 함께 제공합니다.
    """
    
    def __init__(self, k):
        """
        클래스의 생성자. 검색할 문서의 수(k)를 초기화합니다.
        
        :param k: 검색할 문서의 수
        """
        super().__init__()  # 부모 클래스(BaseRetriever)의 초기화 호출
        self._k = k  # 문서 수 저장

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """
        주어진 쿼리에 대해 관련 문서를 검색하고 반환합니다.
        
        :param query: 검색할 쿼리
        :return: 관련 문서 리스트
        """
        docs_and_scores = self.similarity_search_with_score(query=query, k=self._k)  # 유사도 검색 수행
        self.docs = [doc for doc, score in docs_and_scores]  # 문서 추출
        return self.docs[: self._k]  # 상위 k개의 문서 반환

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """
        비동기적으로 주어진 쿼리에 대해 관련 문서를 검색합니다.
        
        :param query: 검색할 쿼리
        :return: 관련 문서 리스트
        """
        docs_and_scores = self.similarity_search_with_score(query=query, k=self._k)  # 유사도 검색 수행
        self.docs = [doc for doc, score in docs_and_scores]  # 문서 추출
        return self.docs[: self._k]  # 상위 k개의 문서 반환

    def similarity_search_with_score(self, query, k):
        """
        쿼리와 k에 기반하여 유사도 검색을 수행해야 메서드입니다.
        기본 구현은 None을 반환합니다.
        
        :param query: 검색할 쿼리
        :param k: 반환할 문서 수
        :return: 문서와 점수를 포함하는 리스트
        """
        return None  # 기본적으로 None 반환 (구현 필요)

# Main execution
if __name__ == "__main__":
    get_tokenizer()  # 사용자 요청된 토크나이저 반환 (일반적으로 이름 필요)
    kiwi_tokenize("예시 텍스트입니다.")  # 예시 텍스트를 Kiwipiepy로 토큰화
    custom_kiwi_tokenize("예시 텍스트입니다.")  # 예시 텍스트를 사용자 정의 Kiwipiepy로 토큰화
    okt_tokenize("예시 텍스트입니다.")  # 예시 텍스트를 Okt로 토큰화
    mecab_tokenize("예시 텍스트입니다.")  # 예시 텍스트를 MeCab으로 토큰화
    kkma_tokenize("예시 텍스트입니다.")  # 예시 텍스트를 Kkma로 토큰화