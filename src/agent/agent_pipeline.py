"""Agent-based IQS Search Pipeline
전체 시스템 통합 및 실행 (Kiwi Preprocessor 통합됨)
"""
from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path

# 기존 모듈 임포트

from src.agent.langgraph_agent import LangGraphIQSAgent
from src.agent.llm_client import LLMClientFactory
from src.data_pipeline.excel_loader import ExcelDataLoader
from src.utils.logger import log
from config.settings import settings

# Preprocessor 임포트

try:
    from src.agent.preprocessor import preprocessor
except ImportError:
    log.warning("KiwiPreprocessor not found. Skipping preprocessing step.")
    preprocessor = None

class AgentPipeline:
    """
    LangGraph Agent의 편의 래퍼 클래스 + 전처리(Preprocessing) 통합
    """

    def __init__(self,
                 llm_type: str = "h-chat",
                 use_cache: bool = True):
        self.llm_type = llm_type
        self.use_cache = use_cache

        # LangGraph Agent 초기화
        self.agent = LangGraphIQSAgent(llm_type=llm_type)

        # Pipeline 전용 기능
        self.execution_history = []
        self.data_loaded = False
        self.documents = []

        log.info(f"Initialized AgentPipeline with Kiwi Preprocessor support")

    def load_data(self, excel_path: Optional[str] = None, force_reload: bool = False):
        """데이터 로드 및 인덱싱 (기존과 동일)"""
        if self.data_loaded and not force_reload:
            log.info("Data already loaded, skipping")
            return

        log.info("Loading data for Agent pipeline")
        loader = ExcelDataLoader(excel_path)
        loader.load_excel()
        loader.clean_data()
        documents = loader.process_for_indexing()

        # 문서 인덱싱 위임
        if hasattr(self.agent, 'build_search_index'):
            self.agent.build_search_index(documents)

        self.data_loaded = True
        self.documents = documents
        log.info(f"Loaded {len(documents)} documents")

    def process_query(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        쿼리 처리 - Kiwi 전처리 + Agent 실행
        """
        start_time = time.time()
        log.info(f"Processing query: {query}")

        # 🆕 1. Kiwi 전처리 및 엔티티 추출
        # 사용자의 질문을 분석해 '시스템 컨텍스트'를 생성합니다.
        enhanced_query = query
        extracted_entities = {}

        if preprocessor:
            try:
                # 엔티티(차종, 연식, 코드) 추출
                extracted_entities = preprocessor.extract_entities(query)
                keywords = preprocessor.extract_keywords(query)

                log.info(f"Kiwi extracted: {extracted_entities}")

                # LLM에게 힌트를 주기 위한 시스템 컨텍스트 구성
                system_hint = "\\n\\n[System Context from Database]\\n"
                has_hint = False

                if extracted_entities.get('model'):
                    system_hint += f"- Target Vehicle Model: {extracted_entities['model']}\\n"
                    has_hint = True
                if extracted_entities.get('year'):
                    system_hint += f"- Target Model Year: {extracted_entities['year']}\\n"
                    has_hint = True
                if extracted_entities.get('codes'):
                    codes_str = ", ".join(extracted_entities['codes'])
                    system_hint += f"- Detected Trouble Codes: {codes_str}\\n"
                    has_hint = True

                # 힌트가 있을 경우에만 쿼리에 추가
                if has_hint:
                    system_hint += "- Use these detected entities for filtering and searching."
                    enhanced_query = f"{query}{system_hint}"
                    log.info("Query enhanced with system hints")

            except Exception as e:
                log.warning(f"Preprocessing failed: {e}")

        try:
            # 2. Agent 실행 (향상된 쿼리 전달)
            if not thread_id:
                thread_id = f"pipeline_{int(time.time())}"

            log.info(f"Delegating to Agent with query len={len(enhanced_query)}")

            agent_result = self.agent.process_query(
                query=enhanced_query,  # 힌트가 포함된 쿼리 전달
                thread_id=thread_id
            )

            # 3. 결과 후처리 (Pipeline 기능)
            final_output = self._enhance_agent_result(agent_result)
            execution_time = time.time() - start_time

            response = {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query if enhanced_query != query else None,
                "extracted_entities": extracted_entities, # 추출된 엔티티 정보 포함
                "thread_id": thread_id,
                "result": final_output,
                "execution_time": round(execution_time, 2),
                "agent_info": {
                    "steps_executed": agent_result.get("steps_executed", 0),
                    "tools_used": agent_result.get("tools_used", [])
                },
                "trace": self._create_pipeline_trace(agent_result)
            }

            self._add_to_history(response)
            self._last_result = response

            return response

        except Exception as e:
            log.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "execution_time": round(time.time() - start_time, 2)
            }

    # ... (나머지 메서드 _enhance_agent_result, _create_pipeline_trace 등은 기존 코드 유지)
    # 기존 코드의 나머지 부분은 그대로 두셔도 됩니다.

    def _enhance_agent_result(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """(기존 코드 유지)"""
        if not agent_result or not isinstance(agent_result, dict):
            return {"error": "Invalid agent result"}

        if agent_result.get("success"):
            return {
                "agent_result": agent_result,
                "final_answer": agent_result.get("final_synthesis") or agent_result.get("search_results")
            }
        else:
            return {"error": agent_result.get("error"), "agent_result": agent_result}

    def _create_pipeline_trace(self, agent_result: Dict[str, Any]) -> List[Dict]:
        """(기존 코드 유지)"""
        # ... (생략, 기존과 동일)
        return []

    def _add_to_history(self, response: Dict[str, Any]):
        """(기존 코드 유지)"""
        self.execution_history.append({
            "timestamp": time.time(),
            "query": response.get("query"),
            "success": response.get("success")
        })

    def get_statistics(self) -> Dict[str, Any]:
        """(기존 코드 유지)"""
        return {"total": len(self.execution_history)}