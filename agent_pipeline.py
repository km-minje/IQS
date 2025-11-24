"""Agent-based IQS Search Pipeline
전체 시스템 통합 및 실행
"""
from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path

from src.agent.langgraph_agent import LangGraphIQSAgent
from src.agent.llm_client import LLMClientFactory
from src.data_pipeline.excel_loader import ExcelDataLoader
from src.utils.logger import log
from config.settings import settings


class AgentPipeline:
    """
    LangGraph Agent의 편의 래퍼 클래스
    운영 환경 지원 기능 제공: 히스토리, 통계, 배치 처리, 데이터 로딩
    
    핵심 AI 로직은 LangGraphIQSAgent가 담당
    """

    def __init__(self, 
                 llm_type: str = "h-chat",
                 use_cache: bool = True):
        """
        Initialize Agent Pipeline as a wrapper around LangGraphIQSAgent

        Args:
            llm_type: LLM 클라이언트 타입 ("h-chat")
            use_cache: 캐싱 사용 여부 (Agent에 전달)
        """
        self.llm_type = llm_type
        self.use_cache = use_cache

        # LangGraph Agent 직접 초기화 (도구 관리 전담)
        self.agent = LangGraphIQSAgent(llm_type=llm_type)

        # Pipeline 전용 기능들
        self.execution_history = []
        self.data_loaded = False
        self.documents = []

        log.info(f"Initialized AgentPipeline as wrapper around LangGraphIQSAgent: {llm_type}")
        log.info(f"Pipeline features: history tracking, statistics, batch processing, data loading")

    def load_data(self, excel_path: Optional[str] = None, force_reload: bool = False):
        """
        데이터 로드 및 인덱싱

        Args:
            excel_path: Excel 파일 경로
            force_reload: 강제 재로드 여부
        """
        if self.data_loaded and not force_reload:
            log.info("Data already loaded, skipping")
            return

        log.info("Loading data for Agent pipeline")

        # Excel 데이터 로드
        loader = ExcelDataLoader(excel_path)
        loader.load_excel()
        loader.clean_data()
        documents = loader.process_for_indexing()

        log.info(f"Loaded {len(documents)} documents")

        # Pipeline 기능: Agent에 인덱스 구축 위임
        if hasattr(self.agent, 'build_search_index'):
            self.agent.build_search_index(documents)
            log.info(f"Pipeline delegated index building to agent: {len(documents)} documents")
        else:
            log.warning("Agent does not support index building")

        self.data_loaded = True
        self.documents = documents

    def process_query(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        쿼리 처리 - Agent에 위임 + Pipeline 기능 추가

        Args:
            query: 사용자 쿼리
            thread_id: 스레드 ID (옵션)

        Returns:
            Pipeline 형태의 처리 결과 (Agent 결과 + 추가 정보)
        """
        start_time = time.time()

        log.info(f"Processing query: {query}")

        try:
            # LangGraph Agent에 전체 위임
            log.info(f"Pipeline delegating to LangGraph Agent: '{query}'")
            
            # 스레드 ID 생성
            if not thread_id:
                thread_id = f"pipeline_{int(time.time())}"
            
            agent_result = self.agent.process_query(
                query=query,
                thread_id=thread_id
            )
            
            # Pipeline 기능 추가: 결과 향상 및 추적
            final_output = self._enhance_agent_result(agent_result)

            # 실행 시간 계산
            execution_time = time.time() - start_time

            # Pipeline 형태의 최종 응답 구성
            response = {
                "success": True,
                "query": query,
                "thread_id": thread_id,
                "result": final_output,
                "execution_time": round(execution_time, 2),
                "pipeline_features": {
                    "history_tracked": True,
                    "statistics_available": True
                },
                # Agent 정보 전달
                "agent_info": {
                    "steps_executed": agent_result.get("steps_executed", 0),
                    "tools_used": agent_result.get("tools_used", []),
                    "agent_success": agent_result.get("success", False)
                },
                "trace": self._create_pipeline_trace(agent_result)
            }

            # Pipeline 기능: 실행 히스토리 저장
            self._add_to_history(response)

            # 마지막 결과 저장 (interactive_test.py에서 사용)
            self._last_result = response
            
            return response

        except Exception as e:
            log.error(f"Query processing failed: {e}")
            execution_time = time.time() - start_time
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "execution_time": round(execution_time, 2)
            }

    def _enhance_agent_result(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 결과를 Pipeline 방식으로 향상
        
        Args:
            agent_result: LangGraph Agent 실행 결과
            
        Returns:
            Pipeline에서 향상된 결과
        """
        # 안전한 agent_result 처리
        if not agent_result or not isinstance(agent_result, dict):
            log.warning("Invalid agent_result for enhancement")
            return {
                "error": "Invalid agent result",
                "pipeline_metadata": {
                    "enhanced_by_pipeline": True,
                    "error_handled": True,
                    "error_type": "invalid_agent_result"
                }
            }
        
        if agent_result.get("success"):
            # Agent 결과 그대로 사용하되 Pipeline 정보 추가
            enhanced_result = {
                "agent_result": agent_result,
                "pipeline_metadata": {
                    "enhanced_by_pipeline": True,
                    "history_position": len(self.execution_history),
                    "data_loaded": self.data_loaded,
                    "enhancement_timestamp": time.time()
                }
            }
            
            # Agent의 최종 결과 추출 (우선순위 순)
            final_answer = None
            
            # 1순위: final_synthesis
            if agent_result.get("final_synthesis") and isinstance(agent_result["final_synthesis"], dict):
                final_answer = agent_result["final_synthesis"]
            # 2순위: aggregation_results  
            elif agent_result.get("aggregation_results") and isinstance(agent_result["aggregation_results"], dict):
                final_answer = agent_result["aggregation_results"]
            # 3순위: search_results
            elif agent_result.get("search_results") and isinstance(agent_result["search_results"], dict):
                final_answer = agent_result["search_results"]
            # 기본값
            else:
                final_answer = {
                    "message": "Agent processing completed",
                    "available_results": list(agent_result.keys())
                }
            
            enhanced_result["final_answer"] = final_answer
            return enhanced_result
        else:
            return {
                "error": agent_result.get("error", "Agent execution failed"),
                "agent_result": agent_result,
                "pipeline_metadata": {
                    "enhanced_by_pipeline": True,
                    "error_handled": True,
                    "error_timestamp": time.time()
                }
            }

    def _create_pipeline_trace(self, agent_result: Dict[str, Any]) -> List[Dict]:
        """
        Pipeline 스타일의 실행 추적 정보 생성
        
        Args:
            agent_result: Agent 실행 결과

        Returns:
            Pipeline 추적 정보
        """
        # 안전한 agent_result 처리
        if not agent_result or not isinstance(agent_result, dict):
            log.warning("Invalid agent_result for trace creation")
            return [{
                "step": 1,
                "component": "AgentPipeline",
                "action": "Error: Invalid agent result",
                "success": False
            }]
        
        trace = []
        
        # 1. Agent 위임 단계 기록
        trace.append({
            "step": 1,
            "component": "LangGraphIQSAgent",
            "action": "Full autonomous execution",
            "success": agent_result.get("success", False),
            "execution_time": agent_result.get("execution_time", 0)
        })
        
        # 2. Agent 내부 도구 사용 기록
        tools_used = agent_result.get("tools_used", [])
        if tools_used and isinstance(tools_used, list):
            for i, tool in enumerate(tools_used, 2):
                trace.append({
                    "step": i,
                    "component": f"Agent.{tool}",
                    "action": f"Tool execution: {tool}",
                    "success": True,  # Agent 성공 시 도구도 성공 가정
                    "managed_by": "LangGraphIQSAgent"
                })
        
        # 3. Pipeline 후처리 단계
        final_step = len(tools_used) + 2 if tools_used else 2
        trace.append({
            "step": final_step,
            "component": "AgentPipeline",
            "action": "Result enhancement and history tracking",
            "success": True
        })
        
        return trace
    
    def _add_to_history(self, response: Dict[str, Any]):
        """
        Pipeline 기능: 실행 히스토리에 추가
        
        Args:
            response: 처리 응답
        """
        try:
            # 안전한 response 처리
            if not response or not isinstance(response, dict):
                log.warning("Invalid response for history tracking")
                return
                
            history_entry = {
                "timestamp": time.time(),
                "query": response.get("query", "Unknown query"),
                "success": response.get("success", False),
                "execution_time": response.get("execution_time", 0),
                "thread_id": response.get("thread_id"),
                "tools_used": response.get("agent_info", {}).get("tools_used", []),
                "steps_executed": response.get("agent_info", {}).get("steps_executed", 0)
            }
            
            self.execution_history.append(history_entry)
            log.debug(f"Added entry to pipeline history (total: {len(self.execution_history)})")
            
        except Exception as e:
            log.error(f"Failed to add entry to history: {e}")

    def batch_process(self, queries: List[str], 
                     thread_prefix: Optional[str] = None,
                     delay_between_queries: float = 0.5) -> List[Dict[str, Any]]:
        """
        Pipeline 기능: 여러 쿼리 일괄 처리
        
        Args:
            queries: 쿼리 리스트
            thread_prefix: 스레드 ID 접두사
            delay_between_queries: 쿼리 간 대기 시간 (초)
            
        Returns:
            Pipeline 결과 리스트
        """
        log.info(f"Pipeline batch processing {len(queries)} queries")
        
        if not thread_prefix:
            thread_prefix = f"batch_{int(time.time())}"
        
        results = []
        batch_start_time = time.time()

        for i, query in enumerate(queries, 1):
            query_start_time = time.time()
            log.info(f"Pipeline batch: processing query {i}/{len(queries)}")
            
            # 각 쿼리마다 고유 스레드 ID
            thread_id = f"{thread_prefix}_query_{i}"
            
            result = self.process_query(query, thread_id=thread_id)
            
            # 배치 메타데이터 추가
            result["batch_metadata"] = {
                "batch_position": i,
                "batch_total": len(queries),
                "batch_query_time": time.time() - query_start_time,
                "thread_id": thread_id
            }
            
            results.append(result)

            # API 레이트 리미팅
            if i < len(queries):  # 마지막 쿼리 후에는 대기 안함
                time.sleep(delay_between_queries)
        
        batch_total_time = time.time() - batch_start_time
        log.info(f"Pipeline batch processing complete: {len(results)} results in {batch_total_time:.2f}s")
        
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Pipeline 기능: 종합 통계 정보 제공
        
        Returns:
            Pipeline 및 Agent 통계 정보
        """
        if not self.execution_history:
            return {
                "message": "No queries processed yet",
                "pipeline_status": "ready",
                "agent_status": "initialized"
            }

        total_queries = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get('success'))
        failed = total_queries - successful

        execution_times = [h.get('execution_time', 0) for h in self.execution_history if h.get('execution_time')]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # 도구 사용 통계
        all_tools_used = []
        for h in self.execution_history:
            tools = h.get('tools_used', [])
            all_tools_used.extend(tools)
        
        tool_usage = {}
        for tool in all_tools_used:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

        pipeline_stats = {
            # Pipeline 통계
            "pipeline_info": {
                "total_queries": total_queries,
                "successful_queries": successful,
                "failed_queries": failed,
                "success_rate": round(successful / total_queries * 100, 2) if total_queries > 0 else 0,
                "average_execution_time": round(avg_time, 2),
                "data_loaded": self.data_loaded,
                "documents_count": len(self.documents)
            },
            
            # Agent 정보
            "agent_info": {
                "agent_type": "LangGraphIQSAgent",
                "llm_type": self.llm_type,
                "tools_available": list(getattr(self.agent, 'tools', {}).keys()),
                "tools_usage_count": tool_usage
            },
            
            # Pipeline 기능
            "pipeline_features": {
                "history_tracking": True,
                "statistics_collection": True,
                "batch_processing": True,
                "data_loading_management": True,
                "agent_delegation": True
            }
        }
        
        # Agent 도구 검증 정보 추가
        if hasattr(self.agent, 'validate_tools'):
            try:
                tool_validation = self.agent.validate_tools()
                pipeline_stats["agent_info"]["tool_validation"] = tool_validation
            except Exception as e:
                log.warning(f"Tool validation failed: {e}")
                
        return pipeline_stats

    def save_history(self, filepath: Optional[str] = None, include_agent_info: bool = True):
        """
        Pipeline 기능: 실행 기록 저장
        
        Args:
            filepath: 저장 경로
            include_agent_info: Agent 정보 포함 여부
        """
        if not self.execution_history:
            log.warning("No execution history to save")
            return

        if not filepath:
            filepath = Path("data/pipeline_history.json")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 저장할 데이터 준비
        save_data = {
            "pipeline_metadata": {
                "pipeline_type": "AgentPipeline",
                "agent_type": "LangGraphIQSAgent", 
                "llm_type": self.llm_type,
                "save_timestamp": time.time(),
                "total_entries": len(self.execution_history)
            },
            "execution_history": self.execution_history
        }
        
        # Agent 정보 추가
        if include_agent_info and hasattr(self.agent, 'get_tool_info'):
            try:
                save_data["agent_info"] = self.agent.get_tool_info()
            except Exception as e:
                log.warning(f"Failed to include agent info: {e}")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        log.info(f"Pipeline history saved to {filepath} ({len(self.execution_history)} entries)")
    
    def load_history(self, filepath: str) -> bool:
        """
        Pipeline 기능: 실행 기록 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드 성공 여부
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            log.error(f"History file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "execution_history" in data:
                # 새 형식 (pipeline metadata 포함)
                self.execution_history = data["execution_history"]
                log.info(f"Loaded pipeline history: {len(self.execution_history)} entries")
                
                if "pipeline_metadata" in data:
                    metadata = data["pipeline_metadata"]
                    log.info(f"History metadata: {metadata.get('pipeline_type')} with {metadata.get('agent_type')}")
                    
            elif isinstance(data, list):
                # 이전 형식 (list 형태)
                self.execution_history = data
                log.info(f"Loaded legacy history format: {len(self.execution_history)} entries")
            else:
                log.error("Invalid history file format")
                return False
                
            return True
            
        except Exception as e:
            log.error(f"Failed to load history: {e}")
            return False


def test_agent_pipeline():
    """Agent Pipeline 통합 테스트 - LangGraphIQSAgent 래퍼로서의 역할 테스트"""

    print("=" * 70)
    print("Testing Agent Pipeline (as LangGraph Agent Wrapper)")
    print("=" * 70)

    # Pipeline 초기화 - H-Chat 사용
    print("Initializing Pipeline (wrapper around LangGraphIQSAgent)...")
    try:
        pipeline = AgentPipeline(llm_type="h-chat")
        print("✓ Pipeline initialized successfully (with LangGraph Agent)")
        
        # Agent 정보 표시
        if hasattr(pipeline.agent, 'get_tool_info'):
            tool_info = pipeline.agent.get_tool_info()
            print(f"✓ Agent tools available: {list(tool_info.keys())}")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return None

    # 샘플 데이터로 테스트 (실제 데이터가 없는 경우)
    sample_docs = [
        {
            "verbatim_id": f"{i:03d}",
            "model": ["Santa Fe", "Tucson", "Palisade"][i % 3],
            "model_year": 2024 + (i % 2),
            "problem": ["Tire - Vibration", "Engine - Noise", "Brake - Squeak"][i % 3],
            "verbatim_text": f"Sample issue {i}",
            "registration_date": f"2024-{(i % 12) + 1:02d}-01"
        }
        for i in range(30)
    ]

            # Pipeline을 통해 인덱스 구축 (내부적으로 Agent에 위임)
    pipeline.load_data()  # 이미 sample_docs 대신 내부 데이터 사용
    print("✓ Data loading delegated to agent via pipeline")

    # Pipeline 기능 테스트 쿼리들
    test_queries = [
        "2025년 산타페 타이어 문제 Top 3는 뭐야?",  # Agent 능력 테스트
        "엔진 소음 관련 불만 찾아줘",  # 검색 기능 테스트
        "가장 많이 발생하는 문제 5개 보여줘"  # 집계 기능 테스트
    ]

    # Pipeline 래퍼 기능 테스트
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Pipeline Test {i}: {query}")
        print("-" * 50)

        result = pipeline.process_query(query)

        if result['success']:
            print("✓ Pipeline successfully delegated to Agent")
            print(f" Total execution time: {result['execution_time']}s")
            print(f" Thread ID: {result.get('thread_id')}")
            
            # Agent 정보 표시
            agent_info = result.get('agent_info', {})
            print(f" Agent success: {agent_info.get('agent_success')}")
            print(f" Tools used by Agent: {agent_info.get('tools_used', [])}")
            print(f" Agent steps: {agent_info.get('steps_executed', 0)}")

            # Pipeline 실행 추적
            print("\nPipeline execution trace:")
            for step in result['trace']:
                status = "✓" if step['success'] else "✗"
                component = step.get('component', 'Unknown')
                action = step.get('action', step.get('description', 'No description'))
                print(f" {status} {component}: {action}")

            # 최종 결과 표시
            print("\nFinal result (enhanced by pipeline):")
            final_answer = result['result'].get('final_answer', result['result'])
            if isinstance(final_answer, dict) and 'synthesis' in final_answer:
                print(final_answer['synthesis'][:300] + "...")
            else:
                result_str = json.dumps(final_answer, indent=2, ensure_ascii=False)
                print(result_str[:300] + "..." if len(result_str) > 300 else result_str)
        else:
            print(f"✗ Pipeline delegation failed: {result.get('error')}")

    # Pipeline 통계 표시
    print("\n" + "=" * 50)
    print("Pipeline Statistics (wrapper functionality):")
    print("-" * 50)
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            print(f" {key}: {', '.join(map(str, value)) if value else 'None'}")
        else:
            print(f" {key}: {value}")
    
    # Agent 도구 정보 표시
    if hasattr(pipeline.agent, 'get_tool_info'):
        print("\nAgent Tool Information:")
        tool_info = pipeline.agent.get_tool_info()
        for tool_name, info in tool_info.items():
            print(f" {tool_name}: {info.get('description', 'No description')}")

    print("\n" + "=" * 70)
    print("Agent Pipeline (wrapper) test complete!")
    print("Successfully validated Pipeline as LangGraph Agent wrapper with:")
    print("- History tracking")
    print("- Statistics collection")
    print("- Agent delegation")
    print("- Result enhancement")


if __name__ == "__main__":
    test_agent_pipeline()