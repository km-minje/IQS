from typing import Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)

from langchain_core.outputs import ChatGeneration, ChatResult

class CustomChatModel(BaseChatModel):
    """
    CustomChatModel 클래스는 특정 대화 모델을 사용하여 자연어 응답을 생성하는 기능을 제공합니다.
    """
    
    def __init__(self, config, all_config):
        """
        클래스의 생성자. 모델과 토크나이저를 초기화하고, 하드웨어 및 최대 토큰 수를 설정합니다.
        
        :param config: 모델 경로 및 최대 토큰 수를 포함하는 설정 정보
        :param all_config: 하드웨어 설정 포함
        """
        super().__init__()  # 부모 클래스(BaseChatModel)의 초기화 호출
        
        path = config.get("path")  # 설정에서 모델 경로 가져오기
        device = all_config["hardware"]["cuda_devices"]  # CUDA 장치 설정
        max_tokens = config.get("max_tokens")  # 최대 토큰 수 가져오기
        
        # 모델과 토크나이저 초기화
        self._model = AutoModelForCausalLM.from_pretrained(path, device_map='auto')
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        
        self._device = device  # 디바이스 저장
        self._max_tokens = max_tokens  # 최대 토큰 수 저장
        
        self.invoke("test")  # "test" 메시지 전송 (주석: 활용처 불명)

    def set_max_tokens(self, max_tokens):
        """
        최대 토큰 수를 설정합니다.
        
        :param max_tokens: 최대 토큰 수
        """
        self._max_tokens = max_tokens
            
    def set_template(self, path):
        """
        대화 템플릿을 설정합니다.
        
        :param path: 템플릿 파일 경로
        """
        with open(path, "r", encoding="utf-8") as f:
            self._tokenizer.chat_template = f.read()  # 템플릿 파일 내용을 읽어 설정

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        주어진 메시지를 기반으로 텍스트 생성을 수행합니다.
        
        :param messages: 사용자 및 AI 메시지의 리스트
        :param stop: 생성을 중지할 토큰의 리스트 (선택적)
        :param run_manager: 콜백 관리자의 인스턴스 (선택적)
        :return: 생성된 응답 결과
        """
        _messages = []
        for message in messages:
            if message.type == 'human':
                # 추가적인 시스템 메시지(주석 처리)
                _messages.append({"role": "user", "content": message.content})  # 사용자 메시지 추가

        # 토크나이저를 사용하여 입력 텍스트 생성
        text = self._tokenizer.apply_chat_template(
            _messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # 모델을 통해 텍스트 생성
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self._max_tokens,
        )
        
        # 입력 토큰 길이를 제외한 생성된 토큰 IDs를 얻기
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 생성된 텍스트를 디코딩
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # AI의 응답 추출
        response_text = response.split("AI:")[-1].strip()
        message = AIMessage(content=response_text)  # AI 메시지 객체 생성
        generation = ChatGeneration(message=message)  # ChatGeneration 객체 생성
        
        return ChatResult(generations=[generation])  # 결과 반환

    def _llm_type(self) -> str:
        """
        모델 유형을 반환합니다. (여기서는 'huggingface'로 설정)
        
        :return: 하드웨어 관련 문자열
        """
        return "huggingface"  # 모델 유형 반환