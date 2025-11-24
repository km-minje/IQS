"""
LangGraph-based Agent Implementation
StateGraph와 Node/Edge를 사용한 LangGraph 시스템
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Any, List, Optional, Annotated, TypedDict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import json
import time

from src.utils.logger import log
from src.agent.tools import (
    HybridSearchTool, AggregatorTool, RerankerTool, 
    SynthesizerTool, GlossaryTool
)
from src.agent.llm_client import LLMClientFactory

class AgentState(TypedDict):
    """LangGraph Agent State Definition"""
    # Core state
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    current_goal: str
    
    # Execution tracking
    step_count: int
    max_steps: int
    
    # Results accumulation
    search_results: Optional[Dict[str, Any]]
    aggregation_results: Optional[Dict[str, Any]]
    reranked_results: Optional[Dict[str, Any]]
    final_synthesis: Optional[Dict[str, Any]]
    
    # Execution metadata
    execution_start_time: float
    step_execution_times: List[float]
    tools_used: List[str]
    
    # Decision making (기존 호환성 유지)
    next_action: Optional[str]
    confidence_score: float
    requires_search: bool
    requires_aggregation: bool
    requires_reranking: bool
    
    # 🆕 Dynamic Planning Support
    execution_plan: Optional[List[Dict[str, Any]]]  # LLM이 생성한 실행 계획
    current_step_index: int                         # 현재 실행 중인 단계
    user_intent: Optional[str]                      # LLM이 파악한 사용자 의도
    complexity_level: Optional[str]                 # 쿼리 복잡도 (simple/moderate/complex)
    plan_reasoning: Optional[str]                   # 계획 수립 근거
    
    # Error handling
    errors: List[str]
    retry_count: int

class LangGraphIQSAgent:
    
    def __init__(self, llm_type: str = "h-chat"):
        """
        Initialize LangGraph Agent
        
        Args:
            llm_type: LLM type for the agent
        """
        self.llm_client = LLMClientFactory.create(llm_type)
        
        # Initialize tools
        self.tools = {
            'glossary': GlossaryTool(),
            'search': HybridSearchTool(), 
            'aggregator': AggregatorTool(),
            'reranker': RerankerTool(),
            'synthesizer': SynthesizerTool(llm_client=self.llm_client)
        }
        
        # Create the StateGraph
        self.graph = self._create_graph()
        
        # Memory for conversation
        self.memory = MemorySaver()
        
        # Compile the graph
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        log.info(f"Initialized LangGraph IQS Agent with {len(self.tools)} tools")
        log.info(f"Available tools: {list(self.tools.keys())}")
    
    def _create_graph(self) -> StateGraph:
        """
        Create the LangGraph StateGraph with Nodes and Edges
        
        Returns:
            Configured StateGraph
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("glossary_translator", self._glossary_node)
        workflow.add_node("search_executor", self._search_node)
        workflow.add_node("aggregator_executor", self._aggregator_node)
        workflow.add_node("reranker_executor", self._reranker_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("decision_maker", self._decision_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add conditional edges from planner
        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "glossary": "glossary_translator",
                "search": "search_executor", 
                "aggregation": "aggregator_executor",
                "end": END
            }
        )
        
        # Add edges from glossary
        workflow.add_edge("glossary_translator", "search_executor")
        
        # Add conditional edges from search
        workflow.add_conditional_edges(
            "search_executor",
            self._route_from_search,
            {
                "aggregation": "aggregator_executor",
                "reranking": "reranker_executor",
                "synthesis": "synthesizer",
                "decision": "decision_maker"
            }
        )
        
        # Add conditional edges from aggregator
        workflow.add_conditional_edges(
            "aggregator_executor", 
            self._route_from_aggregator,
            {
                "reranking": "reranker_executor",
                "synthesis": "synthesizer",
                "decision": "decision_maker"
            }
        )
        
        # Add edges from reranker
        workflow.add_edge("reranker_executor", "decision_maker")
        
        # Add conditional edges from decision maker
        workflow.add_conditional_edges(
            "decision_maker",
            self._route_from_decision,
            {
                "search": "search_executor",
                "aggregation": "aggregator_executor", 
                "reranking": "reranker_executor",
                "synthesis": "synthesizer",
                "end": END
            }
        )
        
        # Add edge from synthesizer to end
        workflow.add_edge("synthesizer", END)
        
        return workflow
    
    # ========== NODE IMPLEMENTATIONS ==========
    
    def _planner_node(self, state: AgentState) -> AgentState:

        log.info("ENHANCED PLANNER NODE: Dynamic LLM-based planning")
        
        original_query = state["original_query"]
        
        # Dynamic planning prompt
        enhanced_planning_prompt = f"""
당신은 IQS 차량 품질 데이터 분석을 위한 AI 계획 수립 전문가입니다.

**사용자 요청**: "{original_query}"

**사용 가능한 도구들**:
- **glossary**: 한-영 용어 번역 및 동의어 확장
- **search**: 하이브리드 검색 (BGE-M3 의미검색 + 키워드)
- **aggregator**: 데이터 집계/통계 (GROUP BY, COUNT, TOP N)
- **reranker**: 결과 재정렬 및 필터링 
- **synthesizer**: 최종 분석 및 응답 생성

**역할**: 이 요청을 해결하기 위한 최적의 실행 계획을 수립하세요.

**중요 지침**:
- 복잡한 요청도 단계별로 분해하세요
- 도구를 여러 번 사용해도 됩니다 
- 창의적이고 효율적인 해결책을 제시하세요
- 사용자 의도를 정확히 파악하세요

JSON 형식으로 응답하세요:
{{
    "user_intent": "사용자가 원하는 것은 무엇인가?",
    "complexity_level": "simple|moderate|complex",
    "execution_plan": [
        {{
            "step": 1,
            "action": "tool_name", 
            "purpose": "이 단계의 목적",
            "parameters": {{}}
        }},
        {{
            "step": 2, 
            "action": "tool_name",
            "purpose": "다음 단계 목적", 
            "parameters": {{}}
        }}
    ],
    "reasoning": "전체적인 계획 수립 근거",
    "confidence": 0.0-1.0
}}

**예시**:
- "INFO12와 INFO13 문제를 각각 찾아서 비교해줘"
  → [search(INFO12) → search(INFO13) → aggregator(비교분석) → synthesizer]
- "2024년과 2025년 현대차 인포테인먼트 문제 비교"
  → [glossary(용어번역) → search(2024) → search(2025) → aggregator(비교) → synthesizer]
"""
        
        try:
            # LLM 기반 동적 계획 수립
            plan_result = self.llm_client.complete_json(enhanced_planning_prompt)
            
            # 동적 상태 업데이트
            execution_plan = plan_result.get("execution_plan", [])
            user_intent = plan_result.get("user_intent", "")
            complexity = plan_result.get("complexity_level", "moderate")
            reasoning = plan_result.get("reasoning", "")
            
            # 새로운 동적 필드들 설정
            state.update({
                # 기존 호환성 유지 (자동 감지)
                "requires_search": any(step.get("action") == "search" for step in execution_plan),
                "requires_aggregation": any(step.get("action") == "aggregator" for step in execution_plan),
                "requires_reranking": any(step.get("action") == "reranker" for step in execution_plan),
                
                # 동적 계획 필드들
                "execution_plan": execution_plan,
                "current_step_index": 0,
                "user_intent": user_intent,
                "complexity_level": complexity,
                "plan_reasoning": reasoning,
                "confidence_score": plan_result.get("confidence", 0.8),
                "step_count": state.get("step_count", 0) + 1
            })
            
            # 첫 번째 단계 결정
            if execution_plan and len(execution_plan) > 0:
                first_step = execution_plan[0]
                state["next_action"] = first_step.get("action", "search")
                
                # 계획 로깅
                log.info(f"Dynamic plan created: {len(execution_plan)} steps")
                log.info(f"User intent: {user_intent}")
                log.info(f"Complexity: {complexity}")
                log.info(f"First action: {state['next_action']}")
                
                # 계획 요약 생성
                plan_summary = f"{complexity} 복잡도 - {len(execution_plan)}단계 계획: {' → '.join([step.get('action', '') for step in execution_plan])}"
                
            else:
                # 폴백: 기본 검색
                state["next_action"] = "search"
                plan_summary = "기본 검색 계획으로 폴백"
                log.warning("No execution plan generated, falling back to search")
            
            # 메시지 추가
            state["messages"].append(
                AIMessage(content=f"동적 계획 수립 완료: {user_intent} - {plan_summary}")
            )
            
            log.info(f"Enhanced planning complete: next_action={state['next_action']}")
            
        except Exception as e:
            log.error(f"Enhanced planning failed: {e}")
            # 폴백: 기존 방식으로 단순화
            state.update({
                "next_action": "search",
                "execution_plan": [{"step": 1, "action": "search", "purpose": "기본 검색"}],
                "current_step_index": 0,
                "user_intent": "검색 요청",
                "complexity_level": "simple",
                "plan_reasoning": "계획 수립 실패로 인한 기본 계획",
                "errors": state.get("errors", []) + [f"Enhanced planning error: {e}"]
            })
        
        return state
    
    def _glossary_node(self, state: AgentState) -> AgentState:

        log.info("GLOSSARY NODE: Translating Korean terms")
        
        try:
            glossary_result = self.tools['glossary'].execute(
                query=state["original_query"],
                include_synonyms=True
            )
            
            # Update query if translation applied
            if glossary_result.get("translation_applied", False):
                enhanced_query = glossary_result.get("enhanced_query", state["original_query"])
                state["current_goal"] = enhanced_query
                log.info(f"Query enhanced: '{state['original_query']}' → '{enhanced_query}'")
            
            state.update({
                "step_count": state.get("step_count", 0) + 1,
                "tools_used": state.get("tools_used", []) + ["glossary"]
            })
            
            state["messages"].append(
                AIMessage(content=f"용어 번역 완료: {glossary_result.get('enhanced_query', '')}")
            )
            
        except Exception as e:
            log.error(f"Glossary translation failed: {e}")
            state["errors"] = state.get("errors", []) + [f"Glossary error: {e}"]
        
        return state
    
    def _search_node(self, state: AgentState) -> AgentState:
        """
        Search Executor Node: Perform hybrid search
        """
        log.info("SEARCH NODE: Executing hybrid search")
        
        try:
            query = state.get("current_goal", state["original_query"])
            
            search_result = self.tools['search'].execute(
                query=query,
                limit=3000,
                step_description="Comprehensive data search",
                original_query=state["original_query"]
            )
            
            state.update({
                "search_results": search_result,
                "step_count": state.get("step_count", 0) + 1,
                "tools_used": state.get("tools_used", []) + ["search"]
            })
            
            total_hits = search_result.get("total_hits", 0)
            state["messages"].append(
                AIMessage(content=f"검색 완료: {total_hits:,}개 문서 발견")
            )
            
            log.info(f"Search completed: {total_hits:,} documents found")
            
        except Exception as e:
            log.error(f"Search failed: {e}")
            state["errors"] = state.get("errors", []) + [f"Search error: {e}"]
        
        return state
    
    def _aggregator_node(self, state: AgentState) -> AgentState:
        """
        Aggregator Executor Node: Perform data aggregation
        """
        log.info("AGGREGATOR NODE: Performing data aggregation")
        
        try:
            # Use search results if available
            documents = None
            if state.get("search_results"):
                documents = state["search_results"].get("documents", [])
            
            agg_result = self.tools['aggregator'].execute(
                documents=documents,
                aggregation="terms",
                field="problem",
                size=20,
                step_description="Statistical analysis",
                original_query=state["original_query"]
            )
            
            state.update({
                "aggregation_results": agg_result,
                "step_count": state.get("step_count", 0) + 1,
                "tools_used": state.get("tools_used", []) + ["aggregator"]
            })
            
            bucket_count = len(agg_result.get("buckets", []))
            state["messages"].append(
                AIMessage(content=f"집계 완료: {bucket_count}개 항목 분석")
            )
            
            log.info(f"Aggregation completed: {bucket_count} buckets")
            
        except Exception as e:
            log.error(f"Aggregation failed: {e}")
            state["errors"] = state.get("errors", []) + [f"Aggregation error: {e}"]
        
        return state
    
    def _reranker_node(self, state: AgentState) -> AgentState:
        """
        Reranker Executor Node: Rerank and filter results
        """
        log.info("RERANKER NODE: Reranking results")
        
        try:
            # Use search or aggregation results
            documents = None
            if state.get("search_results"):
                documents = state["search_results"].get("documents", [])
            elif state.get("aggregation_results"):
                documents = state["aggregation_results"].get("documents", [])
            
            if documents:
                rerank_result = self.tools['reranker'].execute(
                    documents=documents,
                    query=state.get("current_goal", state["original_query"]),
                    top_k=15
                )
                
                state.update({
                    "reranked_results": rerank_result,
                    "step_count": state.get("step_count", 0) + 1,
                    "tools_used": state.get("tools_used", []) + ["reranker"]
                })
                
                reranked_count = rerank_result.get("total_reranked", 0)
                state["messages"].append(
                    AIMessage(content=f"재순위화 완료: {reranked_count}개 문서 정렬")
                )
                
                log.info(f"Reranking completed: {reranked_count} documents")
            else:
                log.warning("No documents to rerank")
            
        except Exception as e:
            log.error(f"Reranking failed: {e}")
            state["errors"] = state.get("errors", []) + [f"Reranking error: {e}"]
        
        return state
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """
        Synthesizer Node: Generate final answer (standard mode)
        """
        log.info("SYNTHESIZER NODE: Generating final synthesis (standard mode)")
        
        try:
            # Collect all results
            results = []
            if state.get("search_results"):
                results.append(state["search_results"])
            if state.get("aggregation_results"):
                results.append(state["aggregation_results"])
            if state.get("reranked_results"):
                results.append(state["reranked_results"])
            
            synthesis_result = self.tools['synthesizer'].execute(
                results=results,
                original_query=state["original_query"]
            )
            
            state.update({
                "final_synthesis": synthesis_result,
                "step_count": state.get("step_count", 0) + 1,
                "tools_used": state.get("tools_used", []) + ["synthesizer"]
            })
            
            state["messages"].append(
                AIMessage(content=synthesis_result.get("synthesis", "분석 완료"))
            )
            
            log.info("Final synthesis completed")
            
        except Exception as e:
            log.error(f"Synthesis failed: {e}")
            state["errors"] = state.get("errors", []) + [f"Synthesis error: {e}"]
        
        return state
    
    def _synthesizer_node_streaming(self, state: AgentState):
        """
        Synthesizer Node: Generate final answer (streaming mode)
        LangGraph astream() 전용 - 스트리밍 청크 yield
        """
        log.info("SYNTHESIZER NODE: Starting streaming synthesis")
        
        try:
            # Collect all results
            results = []
            if state.get("search_results"):
                results.append(state["search_results"])
            if state.get("aggregation_results"):
                results.append(state["aggregation_results"])
            if state.get("reranked_results"):
                results.append(state["reranked_results"])
            
            # SynthesizerTool 스트리밍 실행
            accumulated_text = ""
            for chunk in self.tools['synthesizer'].execute_streaming(
                results=results,
                original_query=state["original_query"]
            ):
                if chunk:
                    accumulated_text += chunk
                    # LangGraph astream에서 사용할 수 있도록 yield
                    yield {
                        "type": "synthesis_chunk",
                        "content": chunk,
                        "accumulated": accumulated_text
                    }
            
            # 스트리밍 완료 후 state 업데이트
            synthesis_result = {
                "query": state["original_query"],
                "synthesis": accumulated_text,
                "source_count": len(results)
            }
            
            state.update({
                "final_synthesis": synthesis_result,
                "step_count": state.get("step_count", 0) + 1,
                "tools_used": state.get("tools_used", []) + ["synthesizer"]
            })
            
            # 최종 완료 신호
            yield {
                "type": "synthesis_complete",
                "content": accumulated_text,
                "state": state
            }
            
            log.info("Streaming synthesis completed")
            
        except Exception as e:
            log.error(f"Streaming synthesis failed: {e}")
            yield {
                "type": "synthesis_error",
                "content": f"Synthesis error: {e}",
                "state": state
            }
    
    def _decision_node(self, state: AgentState) -> AgentState:
        """
        🚀 Enhanced Dynamic Decision Node: Plan-aware intelligent decision making
        """
        log.info("ENHANCED DECISION NODE: Dynamic plan-based decision making")
        
        # 🆕 동적 계획 정보 가져오기
        execution_plan = state.get("execution_plan", [])
        current_step_index = state.get("current_step_index", 0)
        user_intent = state.get("user_intent", "")
        
        # 기존 상태 체크
        has_search = state.get("search_results") is not None
        has_aggregation = state.get("aggregation_results") is not None
        has_reranking = state.get("reranked_results") is not None
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 10)
        tools_used = state.get("tools_used", [])
        
        # 🆕 동적 계획이 있는 경우 - 계획 방식 대신 LLM 자율 판단
        if execution_plan and len(execution_plan) > 0:
            log.info(f"Following dynamic execution plan: step {current_step_index + 1}/{len(execution_plan)}")
            
            # 현재 지점 및 다음 단계 체크
            next_step_index = current_step_index + 1
            
            # LLM 기반 동적 의사결정 (더 유연함)
            decision_context = {
                "original_plan": execution_plan,
                "current_step": current_step_index,
                "next_planned_step": next_step_index,
                "user_intent": user_intent,
                "has_search_results": has_search,
                "has_aggregation_results": has_aggregation, 
                "has_rerank_results": has_reranking,
                "step_count": step_count,
                "tools_used": tools_used,
                "errors": state.get("errors", [])
            }
            
            decision_prompt = f"""
당신은 IQS 데이터 분석 실행 제어 전문가입니다.

**원본 요청**: "{state.get('original_query', '')}"
**사용자 의도**: "{user_intent}"
**원래 계획**: {json.dumps(execution_plan, ensure_ascii=False, indent=2)}

**현재 실행 상황**:
{json.dumps(decision_context, ensure_ascii=False, indent=2)}

**역할**: 다음 실행 단계를 동적으로 결정하세요.

고려사항:
1. 원래 계획을 따를 것인가, 아니면 적응할 것인가?
2. 현재까지의 결과가 충분한가?
3. 에러나 실패가 있었다면 어떻게 복구할 것인가?
4. 사용자 의도를 달성하기 위해 추가 단계가 필요한가?

JSON 형식으로 응답하세요:
{{
    "decision": "continue_plan|adapt_plan|skip_step|complete|synthesis",
    "next_action": "search|aggregator|reranker|glossary|synthesis|end",
    "reasoning": "결정 근거",
    "adaptation_applied": false,
    "confidence": 0.0-1.0
}}
"""
            
            try:
                # LLM 기반 동적 의사결정
                decision_result = self.llm_client.complete_json(decision_prompt)
                
                decision = decision_result.get("decision", "continue_plan")
                next_action = decision_result.get("next_action", "synthesis")
                reasoning = decision_result.get("reasoning", "")
                
                # 상태 업데이트
                state.update({
                    "next_action": next_action,
                    "step_count": step_count + 1
                })
                
                # 계획 진행 상황 업데이트
                if decision == "continue_plan" and next_step_index < len(execution_plan):
                    state["current_step_index"] = next_step_index
                elif decision in ["complete", "synthesis"]:
                    state["current_step_index"] = len(execution_plan)  # 계획 완료
                
                log.info(f"Enhanced decision: {decision} → {next_action}")
                log.info(f"Reasoning: {reasoning}")
                log.info(f"Plan progress: {state.get('current_step_index', 0)}/{len(execution_plan)}")
                
                return state
                
            except Exception as e:
                log.error(f"Enhanced decision making failed: {e}")
                # 폴백: 기존 로직으로
                pass
        
        # 🆕 폴백: 기존 로직 (계획이 없으거나 LLM 의사결정 실패 시)
        log.info("Falling back to legacy decision logic")
        
        # Count how many times each tool was used
        reranker_count = tools_used.count("reranker")
        search_count = tools_used.count("search")
        aggregator_count = tools_used.count("aggregator")
        
        # Enhanced decision logic with comprehensive loop prevention
        if step_count >= max_steps:
            state["next_action"] = "synthesis"
            log.info("Max steps reached, forcing synthesis")
        elif reranker_count >= 2:  # Prevent infinite reranking
            state["next_action"] = "synthesis"
            log.info(f"Reranking limit reached ({reranker_count} times), forcing synthesis")
        elif not has_search and search_count == 0:  # Need initial search
            state["next_action"] = "search"
            log.info("Starting with search")
        elif has_search and not has_aggregation and state.get("requires_aggregation", False) and aggregator_count == 0:
            state["next_action"] = "aggregation"
            log.info("Moving to aggregation")
        elif (has_search or has_aggregation) and state.get("requires_reranking", False) and reranker_count == 0:
            # Only rerank if we have actual documents
            search_docs = state.get("search_results", {}).get("documents", [])
            agg_docs = state.get("aggregation_results", {}).get("documents", [])
            if search_docs or agg_docs:
                state["next_action"] = "reranking"
                log.info("Starting reranking")
            else:
                state["next_action"] = "synthesis"
                log.info("No documents to rerank, forcing synthesis")
        else:
            state["next_action"] = "synthesis"
            log.info("All tasks complete, proceeding to synthesis")
        
        log.info(f"Legacy decision: next_action={state['next_action']} (reranker_count={reranker_count}, step_count={step_count})")
        
        return state
    
    # ========== 🆕 ENHANCED ROUTING FUNCTIONS ==========

    def _route_from_planner(self, state: AgentState) -> str:
        """🆕 Enhanced routing from planner - supports dynamic actions"""
        next_action = state.get("next_action", "search")
        
        # 동적 라우팅 맵
        routing_map = {
            "glossary": "glossary_translator",
            "search": "search_executor", 
            "aggregator": "aggregator_executor",
            "aggregation": "aggregator_executor",  # 동의어 지원
            "reranker": "reranker_executor",
            "reranking": "reranker_executor",      # 동의어 지원
            "synthesizer": "synthesizer",
            "synthesis": "synthesizer",            # 동의어 지원
            "end": "END"
        }
        
        route = routing_map.get(next_action, "search_executor")
        log.info(f"Enhanced planner routing: {next_action} → {route}")
        return route
    
    def _route_from_search(self, state: AgentState) -> str:
        """🆕 Enhanced routing from search - plan-aware"""
        # 동적 계획이 있으면 decision으로 보내서 LLM이 결정
        execution_plan = state.get("execution_plan", [])
        if execution_plan:
            log.info("Search completed, routing to enhanced decision maker")
            return "decision"
        
        # 폴백: 기존 로직
        if state.get("requires_aggregation", False):
            return "aggregation"
        elif state.get("requires_reranking", False):
            return "reranking"
        else:
            return "decision"
        
    def _route_from_aggregator(self, state: AgentState) -> str:
        """🆕 Enhanced routing from aggregator - plan-aware"""
        # 동적 계획이 있으면 decision으로 보내서 LLM이 결정
        execution_plan = state.get("execution_plan", [])
        if execution_plan:
            log.info("Aggregation completed, routing to enhanced decision maker")
            return "decision"
            
        # 폴백: 기존 로직
        if state.get("requires_reranking", False):
            return "reranking" 
        else:
            return "decision"
        
    def _route_from_decision(self, state: AgentState) -> str:
        """🆕 Enhanced routing from decision maker - supports all tools"""
        next_action = state.get("next_action", "synthesis")
        
        # 확장된 라우팅 맵
        routing_map = {
            "search": "search",
            "aggregator": "aggregation",
            "aggregation": "aggregation", 
            "reranker": "reranking",
            "reranking": "reranking",
            "glossary": "glossary",
            "synthesizer": "synthesis",
            "synthesis": "synthesis",
            "end": "end"
        }
        
        route = routing_map.get(next_action, "synthesis")
        log.info(f"Enhanced decision routing: {next_action} → {route}")
        return route
    
    # ========== TOOL MANAGEMENT METHODS ==========
    
    def build_search_index(self, documents: List[Dict[str, Any]]):
        """
        Build search index for the hybrid search tool
        
        Args:
            documents: List of documents to index
        """
        search_tool = self.tools.get('search')
        if search_tool and hasattr(search_tool, 'build_index'):
            search_tool.build_index(documents)
            log.info(f"Search index built with {len(documents)} documents")
        else:
            log.warning("Search tool not available or does not support indexing")
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about available tools
        
        Returns:
            Tool information dictionary
        """
        tool_info = {}
        for name, tool in self.tools.items():
            tool_info[name] = {
                'name': getattr(tool, 'name', name),
                'description': getattr(tool, 'description', 'No description'),
                'available': True
            }
        return tool_info
    
    def validate_tools(self) -> Dict[str, bool]:
        """
        Validate that all tools are properly initialized
        
        Returns:
            Dictionary with tool validation results
        """
        validation_results = {}
        for name, tool in self.tools.items():
            try:
                # Basic validation - check if tool has required methods
                has_execute = hasattr(tool, 'execute')
                has_description = hasattr(tool, 'description') or hasattr(tool, 'get_description')
                validation_results[name] = has_execute and has_description
                
                if not validation_results[name]:
                    log.warning(f"Tool {name} failed validation: execute={has_execute}, description={has_description}")
                    
            except Exception as e:
                log.error(f"Tool {name} validation error: {e}")
                validation_results[name] = False
        
        return validation_results
    
    # ========== PUBLIC METHODS ==========
    
    def process_query(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Process query using LangGraph (standard mode)
        
        Args:
            query: User query
            thread_id: Optional thread ID for conversation
            
        Returns:
            Processing result
        """
        log.info(f"Processing query (standard mode): '{query}'")
        
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            original_query=query,
            current_goal=query, 
            step_count=0,
            max_steps=15,  # 동적 계획을 위해 증가
            execution_start_time=time.time(),
            step_execution_times=[],
            tools_used=[],
            errors=[],
            retry_count=0,
            confidence_score=0.7,
            # 기존 필드 (호환성)
            requires_search=True,
            requires_aggregation=False,
            requires_reranking=False,
            # 동적 계획 필드들
            execution_plan=None,
            current_step_index=0,
            user_intent=None,
            complexity_level=None,
            plan_reasoning=None
        )
        
        thread_config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": 50
        }
        
        try:
            # Execute the graph
            final_state = self.compiled_graph.invoke(initial_state, config=thread_config)
            
            execution_time = time.time() - final_state["execution_start_time"]
            
            # Prepare result
            result = {
                "success": True,
                "query": query,
                "execution_time": execution_time,
                "steps_executed": final_state["step_count"],
                "tools_used": final_state["tools_used"],
                "final_synthesis": final_state.get("final_synthesis"),
                "search_results": final_state.get("search_results"),
                "aggregation_results": final_state.get("aggregation_results"),
                "errors": final_state.get("errors", []),
                "langgraph_state": final_state
            }
            
            log.info(f"LangGraph execution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            log.error(f"LangGraph execution failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "execution_time": time.time() - initial_state["execution_start_time"]
            }
    
    def process_query_streaming(self, query: str, thread_id: str = None):
        """
        Process query using LangGraph (streaming mode)
        전략 1 적용: LangGraph astream() + Synthesizer 스트리밍
        
        Args:
            query: User query
            thread_id: Optional thread ID for conversation
            
        Yields:
            Streaming chunks from synthesis stage
        """
        log.info(f"Processing query (streaming mode): '{query}'")
        
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            original_query=query,
            current_goal=query, 
            step_count=0,
            max_steps=15,  # 동적 계획을 위해 증가
            execution_start_time=time.time(),
            step_execution_times=[],
            tools_used=[],
            errors=[],
            retry_count=0,
            confidence_score=0.7,
            # 기존 필드 (호환성)
            requires_search=True,
            requires_aggregation=False,
            requires_reranking=False,
            # 동적 계획 필드들
            execution_plan=None,
            current_step_index=0,
            user_intent=None,
            complexity_level=None,
            plan_reasoning=None
        )
        
        thread_config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": 50
        }
        
        try:
            # 비동기 스트리밍 대신 사용자 정의 스트리밍
            for chunk in self._custom_streaming_execution(initial_state, thread_config):
                yield chunk
                
        except Exception as e:
            log.error(f"LangGraph streaming failed: {e}")
            yield f"Streaming execution error: {e}"
    
    def _custom_streaming_execution(self, initial_state: AgentState, thread_config: dict):
        """
        사용자 정의 스트리밍 실행
        LangGraph astream() 대신 직접 제어
        """
        state = initial_state
        
        # Phase 1: 비 스트리밍 단계들 순차 실행
        yield "LangGraph Agent 시작 완료"
        
        # Planner node
        state = self._planner_node(state)
        yield "Planning 단계 완료"
        
        # Conditional execution based on plan
        if state.get("next_action") == "glossary":
            state = self._glossary_node(state)
            yield "Glossary 단계 완료"
        
        # Search node
        if state.get("requires_search", True):
            state = self._search_node(state)
            search_hits = state.get("search_results", {}).get("total_hits", 0)
            yield f"Search 단계 완료 ({search_hits:,}개 문서 발견)"
        
        # Aggregation node
        if state.get("requires_aggregation", False):
            state = self._aggregator_node(state)
            bucket_count = len(state.get("aggregation_results", {}).get("buckets", []))
            yield f"Aggregation 단계 완료 ({bucket_count}개 카테고리)"
        
        # Reranking node
        if state.get("requires_reranking", False):
            state = self._reranker_node(state)
            reranked_count = state.get("reranked_results", {}).get("total_reranked", 0)
            yield f"Reranking 단계 완료 ({reranked_count}개 문서 정렬)"
        
        # Phase 2: Synthesis 단계 스트리밍
        yield "\n\nSynthesis 단계 시작 - 실시간 답변 생성 중:\n\n"
        
        # 스트리밍 Synthesizer 실행
        for synthesis_chunk in self._synthesizer_node_streaming(state):
            if synthesis_chunk["type"] == "synthesis_chunk":
                yield synthesis_chunk["content"]
            elif synthesis_chunk["type"] == "synthesis_complete":
                # 종합 완료 - state 업데이트
                state = synthesis_chunk["state"]
                break
            elif synthesis_chunk["type"] == "synthesis_error":
                yield synthesis_chunk["content"]
                break
        
        log.info("Custom streaming execution completed")
    
    def visualize_graph(self, output_path: str = "langgraph_visualization.png"):
        """
        Visualize the LangGraph structure
        
        Args:
            output_path: Path to save visualization
        """
        try:
            # Create a mermaid representation
            mermaid_graph = self._generate_mermaid_graph()
            
            # Save to file
            with open(output_path.replace('.png', '.md'), 'w', encoding='utf-8') as f:
                f.write(f"# LangGraph IQS Agent Visualization\n\n```mermaid\n{mermaid_graph}\n```")
            
            log.info(f"Graph visualization saved to {output_path}")
            return mermaid_graph
            
        except Exception as e:
            log.error(f"Graph visualization failed: {e}")
            return None
    
    def _generate_mermaid_graph(self) -> str:
        """Generate Mermaid graph representation"""
        mermaid = """
graph TD
    START([User Query]) --> PLANNER{Planner<br/>Query Analysis}
    
    PLANNER -->|Korean detected| GLOSSARY[Glossary<br/>Translation]
    PLANNER -->|Direct search| SEARCH[Search<br/>Hybrid Search]
    PLANNER -->|Direct aggregation| AGGREGATOR[Aggregator<br/>Data Analysis]
    PLANNER -->|Complete| END([End])
    
    GLOSSARY --> SEARCH
    
    SEARCH -->|Need aggregation| AGGREGATOR
    SEARCH -->|Need reranking| RERANKER[Reranker<br/>Result Filtering]
    SEARCH -->|Check completion| DECISION{Decision<br/>Next Action?}
    
    AGGREGATOR -->|Need reranking| RERANKER
    AGGREGATOR -->|Check completion| DECISION
    
    RERANKER --> DECISION
    
    DECISION -->|More search needed| SEARCH
    DECISION -->|More aggregation| AGGREGATOR
    DECISION -->|More reranking| RERANKER
    DECISION -->|Ready for synthesis| SYNTHESIZER[Synthesizer<br/>Final Answer]
    DECISION -->|Complete| END
    
    SYNTHESIZER --> END
    
    style START fill:#e1f5fe
    style PLANNER fill:#f3e5f5
    style GLOSSARY fill:#fff3e0
    style SEARCH fill:#e8f5e8
    style AGGREGATOR fill:#fff8e1
    style RERANKER fill:#fce4ec
    style DECISION fill:#f1f8e9
    style SYNTHESIZER fill:#e3f2fd
    style END fill:#ffebee
"""
        return mermaid
    
    def get_graph_state_info(self) -> Dict[str, Any]:
        """Get information about the graph structure"""
        return {
            "nodes": [
                "planner", "glossary_translator", "search_executor",
                "aggregator_executor", "reranker_executor", "synthesizer", "decision_maker"
            ],
            "entry_point": "planner",
            "end_points": ["synthesizer"],
            "conditional_edges": [
                "planner -> {glossary, search, aggregation, end}",
                "search_executor -> {aggregation, reranking, synthesis, decision}",
                "aggregator_executor -> {reranking, synthesis, decision}",
                "decision_maker -> {search, aggregation, reranking, synthesis, end}"
            ],
            "tools_available": list(self.tools.keys()),
            "state_fields": list(AgentState.__annotations__.keys())
        }


### test
def test_langgraph_agent():
    """Test the LangGraph Agent"""
    print("=" * 70)
    print("Testing LangGraph IQS Agent")
    print("=" * 70)
    
    try:
        # Initialize agent
        agent = LangGraphIQSAgent(llm_type="h-chat")
        
        # Test graph visualization
        print("\n1. Generating graph visualization...")
        mermaid = agent.visualize_graph()
        if mermaid:
            print("Graph visualization generated")
            print("Check 'langgraph_visualization.md' for the graph")
        
        # Test query processing
        print("\n2. Testing query processing...")
        test_queries = [
            "인포테인먼트 문제 상위 3개",
            "2025년 산타페 타이어 문제",
            "엔진 소음 관련 불만 찾아줘"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Test {i}: {query}")
            result = agent.process_query(query, thread_id=f"test_{i}")
            
            if result["success"]:
                print(f"    Success: {result['steps_executed']} steps")
                print(f"    Tools used: {result['tools_used']}")
                print(f"    Time: {result['execution_time']:.2f}s")
            else:
                print(f"    Failed: {result['error']}")
        
        # Test graph info
        print("\n3. Graph structure info:")
        graph_info = agent.get_graph_state_info()
        print(f"    Nodes: {len(graph_info['nodes'])}")
        print(f"    Tools: {len(graph_info['tools_available'])}")
        print(f"    State fields: {len(graph_info['state_fields'])}")
        
        print("\n" + "=" * 70)
        print("LangGraph Agent test complete!")
        return True
        
    except Exception as e:
        print(f"LangGraph test failed: {e}")
        return False


if __name__ == "__main__":
    test_langgraph_agent()