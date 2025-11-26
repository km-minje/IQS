import requests
from openai import OpenAI
from typing import List, Callable, Tuple


class OpenAIEmbeddings:
    """
    OpenAIEmbeddings 클래스는 OpenAI API를 사용하여 텍스트의 임베딩을 생성하는 기능을 제공합니다.
    """

    def __init__(self, config):
        """
        클래스의 생성자. 주어진 설정을 사용하여 OpenAI API 클라이언트를 초기화합니다.

        :param config: 모델, API URL, API 키 등을 포함하는 설정 정보.
        """
        self.model = config.get("model")  # 설정에서 모델 이름 가져오기
        self.base_url = config.get("url")  # 설정에서 API URL 가져오기
        self.api_key = config.get("api")  # 설정에서 API 키 가져오기

        self.client = OpenAI(
            api_key="EMPTY", base_url=self.base_url
        )  # OpenAI 클라이언트 초기화
        self.model_id = self.client.models.list().data[0].id  # 사용할 모델 ID 가져오기

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        주어진 문서 텍스트 리스트를 임베딩하여 결과 리스트를 반환합니다.

        :param texts: 임베딩할 문서 텍스트의 리스트
        :return: 각 문서에 대한 임베딩 리스트
        """
        batch_size = 256  # 배치 크기 설정
        while True:
            print(batch_size)
            try:
                self.client.embeddings.create(
                    input=texts[0:batch_size], model=self.model_id
                )
                break
            except:
                batch_size = batch_size // 2
                continue

        ret = []  # 반환 리스트 초기화
        from tqdm import tqdm

        for i in tqdm(range(0, len(texts), batch_size)):
            if i == len(texts):
                continue  # 범위를 벗어나는 경우 건너뛰기

            # 배치별로 임베딩 생성
            temp_ret = [
                data.embedding
                for data in self.client.embeddings.create(
                    input=texts[i : i + batch_size],
                    model=self.model_id,
                ).data
            ]
            ret.extend(temp_ret)  # 결과 리스트에 추가
        return ret  # 임베딩 결과 리스트 반환

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """
        주어진 쿼리 텍스트 리스트를 임베딩하여 결과 리스트를 반환합니다.

        :param texts: 임베딩할 쿼리 텍스트의 리스트
        :return: 각 쿼리에 대한 임베딩 리스트
        """
        batch_size = 256  # 배치 크기 설정
        while True:
            try:
                self.client.embeddings.create(
                    input=texts[0:batch_size], model=self.model_id
                )
                break
            except:
                batch_size = batch_size // 2
                continue

        ret = []  # 반환 리스트 초기화
        for i in range(0, len(texts), batch_size):
            if i == len(texts):
                continue  # 범위를 벗어나는 경우 건너뛰기

            # 배치별로 임베딩 생성
            temp_ret = [
                data.embedding
                for data in self.client.embeddings.create(
                    input=texts[i : i + batch_size],
                    model=self.model_id,
                ).data
            ]
            ret.extend(temp_ret)  # 결과 리스트에 추가
        return ret  # 임베딩 결과 리스트 반환

    def embed_query(self, text: str) -> List[float]:
        """
        주어진 단일 쿼리 텍스트를 임베딩하여 결과 리스트를 반환합니다.

        :param text: 임베딩할 쿼리 텍스트
        :return: 쿼리에 대한 임베딩 리스트
        """
        return (
            self.client.embeddings.create(input=[text], model=self.model_id)
            .data[0]
            .embedding  # 첫 번째 결과의 임베딩 반환
        )
