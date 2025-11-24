"""
LLM Client for Agent System
OpenAI와 Mock 모드 지원
"""
from typing import Dict, Any, List, Optional, Iterator
import json
import re
import requests
from abc import ABC, abstractmethod

from src.utils.logger import log
from config.settings import settings


class BaseLLMClient(ABC):
    """LLM Client 기본 클래스"""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """프롬프트 완성"""
        pass

    @abstractmethod
    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """JSON 형식으로 응답"""
        pass
    
    def stream_complete(self, prompt: str, **kwargs) -> Iterator[str]:
        """스트리밍 프롬프트 완성 (선택적 구현)"""
        # 기본 구현: 일반 완료를 시뮬레이션 스트리밍으로
        import time
        full_response = self.complete(prompt, **kwargs)
        words = full_response.split()
        for i in range(1, len(words) + 1):
            partial = ' '.join(words[:i])
            yield partial
            time.sleep(0.05)


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT Client"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenAI Client

        Args:
            api_key: OpenAI API key
            model: 모델명 (기본값은 settings.OPENAI_CHAT_MODEL)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_CHAT_MODEL

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            log.info(f"Initialized OpenAI client with model: {self.model}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        프롬프트 완성

        Args:
            prompt: 입력 프롬프트
            temperature: 생성 온도 (None이면 settings에서 가져옴)
            max_tokens: 최대 토큰 수 (None이면 settings에서 가져옴)

        Returns:
            생성된 텍스트
        """
        # 기본값을 settings에서 가져오기
        temperature = temperature if temperature is not None else getattr(settings, 'OPENAI_TEMPERATURE', 0.7)
        max_tokens = max_tokens if max_tokens is not None else getattr(settings, 'OPENAI_MAX_TOKENS', 1000)
        system_message = getattr(settings, 'OPENAI_SYSTEM_MESSAGE', 'You are a helpful assistant for analyzing vehicle quality data.')
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            log.error(f"OpenAI API error: {e}")
            raise

    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        JSON 형식으로 응답

        Args:
            prompt: 입력 프롬프트

        Returns:
            파싱된 JSON 객체
        """
        # JSON 응답을 명시적으로 요청
        json_suffix = getattr(settings, 'OPENAI_JSON_SUFFIX', '\n\nRespond with valid JSON only.')
        json_prompt = prompt + json_suffix
        
        json_temperature = getattr(settings, 'OPENAI_JSON_TEMPERATURE', 0.3)
        response_text = self.complete(json_prompt, temperature=json_temperature, **kwargs)

        # JSON 파싱 시도
        try:
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON response: {e}")
            log.debug(f"Response was: {response_text}")
            raise





class HChatClient(BaseLLMClient):
    """
    H-Chat API Client (Multi-Model Support)
    현대자동차 자체 LLM 서비스 - Gemini, OpenAI, Claude 모델 지원
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize H-Chat Client

        Args:
            api_key: H-Chat API key
            model: 모델명 (기본값은 settings.HCHAT_MODEL)
        """
        self.api_key = api_key or settings.HCHAT_API_KEY
        self.model = model or settings.HCHAT_MODEL
        self.base_url = settings.HCHAT_BASE_URL
        
        if not self.api_key:
            raise ValueError("H-Chat API key not provided")
        
        # 모델 계열 자동 감지
        self.model_type = self._detect_model_type(self.model)
        
        log.info(f"Initialized H-Chat client with model: {self.model} (type: {self.model_type})")
        log.info(f"API Key (마지막 8자리): ...{self.api_key[-8:] if len(self.api_key) > 8 else '***'}")
    
    def _detect_model_type(self, model: str) -> str:
        """
        모델명으로부터 모델 계열 자동 감지
        
        Args:
            model: 모델명
            
        Returns:
            모델 계열 ('gemini', 'openai', 'claude')
        """
        model_lower = model.lower()
        
        if 'gemini' in model_lower:
            return 'gemini'
        elif any(name in model_lower for name in ['gpt', 'openai']):
            return 'openai'
        elif 'claude' in model_lower:
            return 'claude'
        else:
            # 기본값: gemini로 처리
            log.warning(f"Unknown model type for {model}, defaulting to gemini")
            return 'gemini'

    def complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        프롬프트 완성 (모델 계열에 따라 자동 분기)

        Args:
            prompt: 입력 프롬프트
            temperature: 생성 온도 (0.0-1.0, None이면 settings에서 가져옴)
            max_tokens: 최대 토큰 수

        Returns:
            생성된 텍스트
        """
        # 기본값 설정
        temperature = temperature if temperature is not None else getattr(settings, 'HCHAT_TEMPERATURE', 0.7)
        max_tokens = max_tokens if max_tokens is not None else 1000
        
        # 모델 계열에 따라 분기
        if self.model_type == 'openai':
            return self._complete_openai(prompt, temperature, max_tokens)
        elif self.model_type == 'claude':
            return self._complete_claude(prompt, temperature, max_tokens)
        else:  # gemini
            return self._complete_gemini(prompt, temperature, max_tokens)
    
    def _complete_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        OpenAI 계열 모델 API 호출
        """
        try:
            url = f"{self.base_url}/openai/deployments/{self.model}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            system_message = getattr(settings, 'HCHAT_SYSTEM_MESSAGE', 'You are a helpful assistant for analyzing vehicle quality data. Respond in Korean when the input is in Korean.')
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            log.debug(f"H-Chat OpenAI API request: {url}")
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=getattr(settings, 'HCHAT_TIMEOUT', 60)
            )
            
            if response.status_code != 200:
                log.error(f"H-Chat OpenAI API error: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            result = response.json()
            
            # OpenAI 응답 구조에서 텍스트 추출
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content']
            
            log.error(f"Unexpected OpenAI response structure: {result}")
            raise ValueError("Invalid response structure from H-Chat OpenAI API")
            
        except requests.exceptions.RequestException as e:
            log.error(f"H-Chat OpenAI API request error: {e}")
            raise
        except Exception as e:
            log.error(f"H-Chat OpenAI API error: {e}")
            raise
    
    def _complete_claude(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Claude 계열 모델 API 호출
        """
        try:
            url = f"{self.base_url}/claude/messages"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            system_message = getattr(settings, 'HCHAT_SYSTEM_MESSAGE', 'You are a helpful assistant for analyzing vehicle quality data. Respond in Korean when the input is in Korean.')
            
            payload = {
                "max_tokens": max_tokens,
                "model": self.model,
                "stream": False,
                "system": system_message,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            log.debug(f"H-Chat Claude API request: {url}")
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=getattr(settings, 'HCHAT_TIMEOUT', 60)
            )
            
            if response.status_code != 200:
                log.error(f"H-Chat Claude API error: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            result = response.json()
            
            # Claude 응답 구조에서 텍스트 추출
            if 'content' in result and len(result['content']) > 0:
                if 'text' in result['content'][0]:
                    return result['content'][0]['text']
            
            log.error(f"Unexpected Claude response structure: {result}")
            raise ValueError("Invalid response structure from H-Chat Claude API")
            
        except requests.exceptions.RequestException as e:
            log.error(f"H-Chat Claude API request error: {e}")
            raise
        except Exception as e:
            log.error(f"H-Chat Claude API error: {e}")
            raise
    
    def _complete_gemini(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Gemini 계열 모델 API 호출 (기존 방식)
        """
        try:
            url = f"{self.base_url}/models/{self.model}:generateContent"
            params = {"key": self.api_key}
            headers = {"Content-Type": "application/json"}
            
            system_message = getattr(settings, 'HCHAT_SYSTEM_MESSAGE', 'You are a helpful assistant for analyzing vehicle quality data. Respond in Korean when the input is in Korean.')
            
            payload = {
                "systemInstruction": {
                    "parts": [{
                        "text": system_message
                    }]
                },
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature
                }
            }
            
            log.debug(f"H-Chat Gemini API request: {url}")
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                data=json.dumps(payload),
                timeout=getattr(settings, 'HCHAT_TIMEOUT', 60)
            )
            
            if response.status_code != 200:
                log.error(f"H-Chat Gemini API error: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            result = response.json()
            
            # Gemini 응답 구조에서 텍스트 추출
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']
            
            log.error(f"Unexpected Gemini response structure: {result}")
            raise ValueError("Invalid response structure from H-Chat Gemini API")
            
        except requests.exceptions.RequestException as e:
            log.error(f"H-Chat Gemini API request error: {e}")
            raise
        except Exception as e:
            log.error(f"H-Chat Gemini API error: {e}")
            raise

    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        JSON 형식으로 응답 (모델 계열에 따라 자동 분기)

        Args:
            prompt: 입력 프롬프트

        Returns:
            파싱된 JSON 객체
        """
        # JSON 응답을 명시적으로 요청
        json_suffix = getattr(settings, 'HCHAT_JSON_SUFFIX', '\n\n반드시 유효한 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.')
        json_prompt = prompt + json_suffix
        
        json_temperature = getattr(settings, 'HCHAT_JSON_TEMPERATURE', 0.3)
        response_text = self.complete(json_prompt, temperature=json_temperature, **kwargs)

        # JSON 파싱 시도
        try:
            # JSON 부분 추출 (마크다운 코드 블록 제거)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # JSON 파싱
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON response from H-Chat ({self.model_type}): {e}")
            log.debug(f"Response was: {response_text}")
            
            # 정규표현식으로 JSON 부분 추출 시도
            try:
                json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"No JSON found in H-Chat {self.model_type} response")
            except json.JSONDecodeError:
                log.error(f"Final JSON parsing attempt failed for {self.model_type}")
                raise
    
    def stream_complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Iterator[str]:
        """
        스트리밍 프롬프트 완성 - ChatGPT처럼 실시간 타이핑 효과
        
        Args:
            prompt: 입력 프롬프트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            
        Yields:
            str: 스트리밍 텍스트 청크들
        """
        # 기본값 설정
        temperature = temperature if temperature is not None else getattr(settings, 'HCHAT_TEMPERATURE', 0.7)
        max_tokens = max_tokens if max_tokens is not None else 1000
        
        # OpenAI 모델만 스트리밍 지원 (실제 H-Chat 테스트 결과)
        if self.model_type == 'openai':
            yield from self._stream_openai(prompt, temperature, max_tokens)
        else:
            # Gemini, Claude는 시뮬레이션 스트리밍
            log.info(f"Model {self.model_type} uses simulated streaming")
            import time
            full_response = self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            words = full_response.split()
            for i in range(1, len(words) + 1):
                partial = ' '.join(words[:i])
                yield partial
                time.sleep(0.05)
    
    def _stream_openai(self, prompt: str, temperature: float, max_tokens: int) -> Iterator[str]:
        """
        OpenAI 모델 실제 스트리밍 (검증된 구현)
        """
        try:
            url = f"{self.base_url}/openai/deployments/{self.model}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream"  # 핵심 헤더
            }
            
            system_message = getattr(settings, 'HCHAT_SYSTEM_MESSAGE', 
                                   'You are a helpful assistant for analyzing vehicle quality data. Respond in Korean when the input is in Korean.')
            
            payload = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True  # 스트리밍 활성화
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=getattr(settings, 'HCHAT_TIMEOUT', 60)
            )
            
            if response.status_code != 200:
                log.error(f"H-Chat streaming error: {response.status_code}")
                yield f"오류: HTTP {response.status_code}"
                return
            
            # 검증된 스트리밍 파싱 로직
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    # H-Chat 형식: data:{...} (공백 없음)
                    if line_str.startswith('data:'):
                        data_str = line_str[5:]  # "data:" 5글자 제거
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                choice = chunk_data['choices'][0]
                                
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        yield content
                                        
                        except (json.JSONDecodeError, KeyError):
                            continue
                            
        except Exception as e:
            log.error(f"Streaming error: {e}")
            yield f"스트리밍 오류: {e}"


class LLMClientFactory:
    """LLM Client Factory"""

    @staticmethod
    def create(client_type: Optional[str] = None, **kwargs) -> BaseLLMClient:
        """
        LLM Client 생성

        Args:
            client_type: "openai", "h-chat", 또는 "mock" (기본값은 settings.AGENT_LLM_TYPE)
            **kwargs: 클라이언트별 추가 파라미터

        Returns:
            LLM Client 인스턴스
        """
        # client_type이 지정되지 않으면 settings에서 가져오기
        if client_type is None:
            client_type = settings.AGENT_LLM_TYPE
        
        if client_type == "openai":
            return OpenAIClient(**kwargs)
        elif client_type == "h-chat":
            return HChatClient(**kwargs)
        else:
            raise ValueError(f"Unknown client type: {client_type}. Supported: 'openai', 'h-chat'")


def test_llm_clients():
    """LLM Client 테스트"""

    print("=" * 70)
    print("Testing LLM Clients")
    print("=" * 70)

    # 1. H-Chat 클라이언트 우선 테스트
    print("\n1. Testing H-Chat Client")
    print("-" * 30)

    # 2. H-Chat Client 상세 테스트
    if settings.HCHAT_API_KEY:
        print("\n2. Testing H-Chat Client (Gemini 2.5 Pro)")
        print("-" * 30)

        try:
            hchat_client = LLMClientFactory.create("h-chat")

            test_prompt = "IQS(Initial Quality Study)에 대해 간단히 설명해주세요."
            response = hchat_client.complete(test_prompt)
            print(f"H-Chat response: {response[:200]}...")

            # JSON 테스트
            json_prompt = """
다음 실행 계획을 JSON으로 작성하세요:
1. 타이어 문제 검색
2. 결과 집계
"""
            json_response = hchat_client.complete_json(json_prompt)
            print(f"\nH-Chat JSON response: {json.dumps(json_response, ensure_ascii=False, indent=2)[:200]}...")

        except Exception as e:
            print(f"H-Chat test failed: {e}")
    else:
        print("\n2. H-Chat Client - Skipped (no API key)")

    # 3. OpenAI Client 테스트 (API 키가 있는 경우만)
    if settings.OPENAI_API_KEY:
        print("\n3. Testing OpenAI Client")
        print("-" * 30)

        try:
            openai_client = LLMClientFactory.create("openai")

            test_prompt = "Briefly describe what IQS (Initial Quality Study) is."
            response = openai_client.complete(test_prompt, max_tokens=100)
            print(f"OpenAI response: {response[:200]}...")

        except Exception as e:
            print(f"OpenAI test failed: {e}")
    else:
        print("\n3. OpenAI Client - Skipped (no API key)")

    print("\n" + "=" * 70)
    print("LLM Client test complete!")


if __name__ == "__main__":
    test_llm_clients()