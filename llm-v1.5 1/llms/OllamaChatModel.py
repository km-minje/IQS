from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

import ollama

class OllamaChatModel(BaseChatModel):
    """
    OllamaChatModel 클래스는 Ollama API를 사용하여 대화 응답을 생성하는 기능을 제공합니다.
    """
    
    def __init__(self, config):
        """
        클래스의 생성자. 주어진 설정을 사용하여 Ollama 모델 및 최대 토큰 수를 초기화합니다.
        
        :param config: 모델 경로 및 최대 토큰 수를 포함하는 설정 정보
        """
        super().__init__()  # 부모 클래스(BaseChatModel)의 초기화 호출

        max_tokens = config.get("max_tokens")  # 설정에서 최대 토큰 수 가져오기
        self._model = config.get("path")  # 설정에서 모델 경로 가져오기
        self._max_tokens = max_tokens  # 최대 토큰 수 저장
        
        self.invoke("test")  # 테스트 메시지를 호출 (주석: 활용처 불명)

    def set_max_tokens(self, max_tokens):
        """
        최대 토큰 수를 설정합니다.
        
        :param max_tokens: 최대 토큰 수
        """
        self._max_tokens = max_tokens

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        주어진 메시지를 기반으로 Ollama 모델을 사용하여 텍스트 생성을 수행합니다.
        
        :param messages: 사용자 및 AI 메시지의 리스트
        :param stop: 생성을 중지할 토큰의 리스트 (선택적)
        :param run_manager: 콜백 관리자의 인스턴스 (선택적)
        :return: 생성된 응답 결과
        """
        prompt = ""  # 프롬프트 초기화
        for msg in messages:
            # 메시지 유형에 따라 역할 설정
            role = "user" if msg.type == "human" else "assistant"
            prompt += f"{role}: {msg.content}\n"  # 프롬프트에 메시지 추가
        prompt += "assistant:"  # AI 응답 시작

        # Ollama API를 호출하여 응답 생성
        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt}  # 사용자 프롬프트 설정
            ],
            options={
                "num_predict": self._max_tokens  # 최대 토큰 수 설정
            }
        )

        # 응답에서 텍스트 추출
        response_text = response['message']['content'].strip()
        message = AIMessage(content=response_text)  # AI 메시지 객체 생성
        generation = ChatGeneration(message=message)  # ChatGeneration 객체 생성
        return ChatResult(generations=[generation])  # 결과 반환

    def _llm_type(self) -> str:
        """
        모델 유형을 반환합니다. (여기서는 'ollama'로 설정)
        
        :return: 하드웨어 관련 문자열
        """
        return "ollama"  # 모델 유형 반환