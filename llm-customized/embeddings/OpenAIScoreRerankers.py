import requests

class OpenAIScoreRerankers:
    """
    OpenAIScoreRerankers 클래스는 OpenAI API를 통해 쿼리와 텍스트 간의 점수를 계산하는 기능을 제공합니다.
    """
    
    def __init__(self, config):
        """
        클래스의 생성자. 주어진 설정을 사용하여 API 모델과 URL, API 키를 초기화합니다.
        
        :param config: 모델, API URL, API 키 등을 포함하는 설정 정보.
        """
        self.model = config.get("model")  # 설정에서 모델 이름 가져오기
        self.base_url = config.get("url")  # 설정에서 API URL 가져오기
        self.api_key = config.get("api")    # 설정에서 API 키 가져오기

        self.headers = {
            "Content-Type": "application/json",  # 요청 내용의 형식
            "Accept": "application/json",         # 응답 형식
        }

    def compute_score(self, query, text):
        """
        주어진 쿼리와 텍스트에 대해 점수를 계산합니다.
        
        :param query: 비교할 쿼리 텍스트
        :param text: 비교할 텍스트
        :return: 계산된 점수와 추가 정보 (현재는 None)
        """
        payload = {
            "model": self.model,  # 사용할 모델
            "text_1": query,      # 첫 번째 텍스트 (쿼리)
            "text_2": text,       # 두 번째 텍스트 (비교할 텍스트)
        }

        response = requests.post(self.base_url, json=payload, headers=self.headers)  # POST 요청 전송
        score = response.json()["data"][0]["score"]  # 응답에서 점수 추출
        return score, None  # 점수와 None 반환

    def get_result(self, query, text):
        """
        주어진 쿼리와 텍스트에 대한 결과를 가져옵니다.
        
        :param query: 비교할 쿼리 텍스트
        :param text: 비교할 텍스트
        :return: API 호출의 전체 응답 결과
        """
        payload = {
            "model": self.model,  # 사용할 모델
            "text_1": query,      # 첫 번째 텍스트 (쿼리)
            "text_2": text,       # 두 번째 텍스트 (비교할 텍스트)
        }

        response = requests.post(self.base_url, json=payload, headers=self.headers)  # POST 요청 전송
        result = response.json()  # 응답을 JSON 형식으로 변환
        return result  # 전체 결과 반환