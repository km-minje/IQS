from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from vllm import LLM, SamplingParams

class VLLMChatModel(BaseChatModel):
    """
    VLLMChatModel 클래스는 VLLM 라이브러리를 사용하여 대화 응답을 생성하는 기능을 제공합니다.
    """
    
    def __init__(self, model: str = "meta-llama/Llama-2-7b-chat-hf", max_tokens: int = 128):
        """
        클래스의 생성자. 주어진 모델 및 최대 토큰 수로 초기화합니다.
        
        :param model: 사용할 모델의 이름 (기본값: "meta-llama/Llama-2-7b-chat-hf")
        :param max_tokens: 최대 생성 토큰 수 (기본값: 128)
        """
        super().__init__()  # 부모 클래스(BaseChatModel)의 초기화 호출
        self._model = model  # 모델 이름 저장
        self._max_tokens = max_tokens  # 최대 토큰 수 저장
        self.llm = LLM(model=model)  # LLM 인스턴스 생성
        self.sampling_params = SamplingParams(max_tokens=max_tokens)  # 샘플링 파라미터 설정
        
    def set_max_tokens(self, max_tokens):
        """
        최대 토큰 수를 설정합니다.

        :param max_tokens: 최대 토큰 수
        """
        self.sampling_params = SamplingParams(max_tokens=max_tokens)  # 새 최대 토큰 수로 샘플링 파라미터 업데이트

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        주어진 메시지를 기반으로 VLLM 모델을 사용하여 텍스트 생성을 수행합니다.
        
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

        # 모델을 통해 텍스트 생성
        outputs = self.llm.generate(prompt, self.sampling_params)
        response_text = outputs[0].outputs[0].text.strip()  # 응답에서 텍스트 추출

        message = AIMessage(content=response_text)  # AI 메시지 객체 생성
        generation = ChatGeneration(message=message)  # ChatGeneration 객체 생성

        return ChatResult(generations=[generation])  # 결과 반환

    def _llm_type(self) -> str:
        """
        모델 유형을 반환합니다. (여기서는 'vllm'로 설정)
        
        :return: 하드웨어 관련 문자열
        """
        return "vllm"  # 모델 유형 반환