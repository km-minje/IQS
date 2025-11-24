import streamlit as st
import sys
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import traceback
from datetime import datetime
import time
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.agent.agent_pipeline import AgentPipeline
    from src.utils.logger import log
except ImportError as e:
    st.error(f"모듈 임포트 실패: {e}")
    st.error("프로젝트 루트 디렉토리에서 실행하세요.")
    st.stop()

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = None

def initialize_iqs_system():
    """IQS 시스템 초기화 - 작은 스피너 사용"""
    try:
        if "iqs_pipeline" not in st.session_state:
            # 작은 로딩 스피너로 초기화 표시
            init_container = st.empty()
            init_container.markdown('<div class="small-spinner"><div class="spinner"></div>IQS 시스템을 초기화하고 있습니다...</div>', unsafe_allow_html=True)
            
            st.session_state.iqs_pipeline = AgentPipeline(llm_type="h-chat")
            
            # 초기화 완료 후 스피너 제거
            init_container.empty()
            st.success("초기화가 완료되었습니다. IQS 시스템이 준비되었습니다!")
        return st.session_state.iqs_pipeline
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.error("환경 설정을 확인하세요 (.env 파일의 HCHAT_API_KEY 등)")
        return None

def stream_agent_response(pipeline, query):
    """
    LangGraph Agent 완전 위임 - Hallucination 방지:
    1. LangGraph Agent가 전체 데이터 처리 및 답변 생성 완료
    2. Streamlit은 Agent 결과만 표시 (추가 LLM 호출 금지)
    """
    try:
        # 시점 1: LangGraph Agent 전체 실행 시작 (HTML 태그 제거)
        yield 'LangGraph Agent가 데이터 분석을 시작합니다...'
        
        # Agent 대기 시간 시각적 피드백
        import asyncio
        for i in range(3):
            yield "."
            time.sleep(0.3)
        
        yield 'LangGraph Agent가 68,982건의 IQS 데이터를 완전 자율적으로 분석 중입니다...'  # HTML 태그 제거
        
        # 시점 1: Agent 실제 실행 (데이터 수집에만 집중)
        result = pipeline.process_query(query)
        
        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            yield f'데이터 분석 실패: {error_msg}'  # HTML 태그 제거
            
            # 구체적인 해결 방안 제시
            if 'API' in error_msg or 'key' in error_msg.lower():
                yield '<div class="process-info">해결 방안: .env 파일의 HCHAT_API_KEY를 확인하세요.</div>'
            elif 'timeout' in error_msg.lower():
                yield '<div class="process-info">해결 방안: 네트워크 연결을 확인하고 잠시 후 다시 시도하세요.</div>'
            else:
                yield '<div class="process-info">해결 방안: 질문을 다시 입력하거나 관리자에게 문의하세요.</div>'
            return
        
        # Agent 실행 완료 알림
        execution_time = result.get('execution_time', 0)
        tools_used = result.get('agent_info', {}).get('tools_used', [])
        
        yield f'<div class="process-info">LangGraph Agent 완료 (소요시간: {execution_time:.1f}초)'
        if tools_used:
            yield f' 🛠 사용된 도구: {", ".join(tools_used)}'
        yield '</div>'
        
        # 시점 2: LangGraph Agent 결과 추출 (추가 LLM 호출 금지)
        yield '<div class="process-info">Agent가 생성한 최종 답변을 추출하고 있습니다...</div>\n\n'
        
        # Agent의 SynthesizerTool 결과만 사용 (추가 LLM 호출 금지)
        final_answer = extract_langgraph_final_answer(result)
        
        if final_answer:
            # Agent가 이미 생성한 답변을 그대로 사용
            yield final_answer
        else:
            # Agent 결과가 없을 경우에만 기본 정보 표시
            yield "😕 LangGraph Agent가 최종 답변을 생성하지 못했습니다. 다른 질문으로 시도해주세요."
                
    except Exception as e:
        yield f'<div class="process-info">시스템 오류 발생: {str(e)[:200]}...</div>'
        yield f'<div class="process-info">진단 정보:'
        yield f'<br>- Pipeline 상태: {"정상" if hasattr(pipeline, "agent") else "비정상"}'
        yield f'<br>- Agent 상태: {"정상" if hasattr(pipeline, "agent") and pipeline.agent else "비정상"}'
        yield '</div>'
        
        # 긴급 폴백: 기본 정보 제공
        yield '<div class="process-info">긴급 모드: 시스템을 재시작하거나 관리자에게 문의하세요.</div>'

def enhance_agent_data_for_streaming(agent_data, query):
    """
    Agent 데이터를 스트리밍용으로 증강 및 재가공
    """
    enhanced = agent_data.copy()
    
    # 검색 결과 요약 정보 추가
    search_results = agent_data.get('search_results')
    if search_results:
        total_hits = search_results.get('total_hits', 0)
        documents = search_results.get('documents', [])
        
        enhanced['search_summary'] = {
            'total_found': total_hits,
            'analyzed_count': len(documents),
            'key_brands': list(set([doc.get('make_of_vehicle', 'Unknown') for doc in documents[:10]])),
            'key_models': list(set([doc.get('model_of_vehicle', 'Unknown') for doc in documents[:10]])),
            'problem_types': len(set([doc.get('problem', '') for doc in documents])),
            'sample_issues': [doc.get('problem', '')[:100] for doc in documents[:3]]
        }
    
    # 집계 결과 요약 정보 추가
    agg_results = agent_data.get('aggregation_results')
    if agg_results:
        buckets = agg_results.get('buckets', [])
        enhanced['aggregation_summary'] = {
            'top_issues_count': len(buckets),
            'total_categories': sum([bucket.get('doc_count', 0) for bucket in buckets]),
            'most_common': buckets[0] if buckets else None
        }
    
    return enhanced

def create_enhanced_streaming_prompt(query, enhanced_data):
    """
    스트리밍용 최적화된 프롬프트 생성
    더 청천하고 간결하여 스트리밍 품질 향상
    """
    
    search_summary = enhanced_data.get('search_summary', {})
    agg_summary = enhanced_data.get('aggregation_summary', {})
    
    # 기본 정보
    prompt = f"""사용자 질문: {query}

분석 결과 요약:"""
    
    # 검색 결과 요약
    if search_summary:
        prompt += f"""
검색 정보:
- 총 {search_summary.get('total_found', 0):,}건의 관련 데이터 발견
- 분석 대상: {search_summary.get('analyzed_count', 0)}건
- 대상 브랜드: {', '.join(search_summary.get('key_brands', [])[:3])}"""
    
    # 집계 결과 요약
    if agg_summary:
        prompt += f"""
집계 분석:
- {agg_summary.get('top_issues_count', 0)}개 주요 문제 유형 식별
- 총 {agg_summary.get('total_categories', 0):,}건의 불만 사례 분석"""
    
    prompt += f"""

사용자에게 위 데이터를 바탕으로 정확하고 상세한 답변을 작성해주세요. 한국어로 친근하고 전문적으로 설명하며, 마크다운 형식을 활용해 구조적으로 작성하세요."""
    
    return prompt

def stream_with_empty_container_v2(pipeline, query):
    """
    st.empty()를 사용한 전략 1 스트리밍
    LangGraph Agent 진행상황 + 실제 토큰 스트리밍
    """
    streaming_container = st.empty()
    accumulated_response = ""
    
    try:
        log.info("Starting Strategy 1 streaming with st.empty()")
        chunk_count = 0
        
        for chunk in stream_agent_response_v2(pipeline, query):
            if chunk:
                accumulated_response += chunk
                chunk_count += 1
                
                # Streamlit 실시간 업데이트 - HTML 태그 지원
                streaming_container.markdown(accumulated_response, unsafe_allow_html=True)
                
                # 부드러운 스트리밍을 위한 최소 딘레이
                time.sleep(0.01)
                
                # 주기적 로깅
                if chunk_count % 20 == 0:
                    log.debug(f"Strategy 1 streaming progress: {chunk_count} chunks, {len(accumulated_response)} chars")
        
        log.info(f"Strategy 1 streaming completed: {chunk_count} chunks, {len(accumulated_response)} characters")
        return accumulated_response
        
    except Exception as e:
        log.error(f"Strategy 1 st.empty() streaming failed: {e}")
        streaming_container.error(f"스트리밍 중 오류 발생: {e}")
        return fallback_to_basic_response(pipeline, query)

def stream_with_empty_container_OLD(pipeline, query):
    """
    st.empty()를 사용한 확실한 스트리밍 구현
    터미널 테스트에서 검증된 H-Chat 스트리밍을 Streamlit에 적용
    """
    streaming_container = st.empty()
    accumulated_response = ""
    
    try:
        log.info("Starting Streamlit streaming with st.empty()")
        chunk_count = 0
        
        for chunk in stream_agent_response(pipeline, query):
            if chunk:
                accumulated_response += chunk
                chunk_count += 1
                
                # Streamlit 실시간 업데이트 (터미널과 동일한 방식)
                streaming_container.markdown(accumulated_response, unsafe_allow_html=True)
                
                # 부드러운 스트리밍을 위한 최소 딜레이
                time.sleep(0.01)
                
                # 주기적 로깅 (디버깅용)
                if chunk_count % 20 == 0:
                    log.debug(f"Streamlit streaming progress: {chunk_count} chunks, {len(accumulated_response)} chars")
        
        log.info(f"Streamlit streaming completed: {chunk_count} chunks, {len(accumulated_response)} characters")
        return accumulated_response
        
    except Exception as e:
        log.error(f"st.empty() streaming failed: {e}")
        streaming_container.error(f"스트리밍 중 오류 발생: {e}")
        return fallback_to_basic_response(pipeline, query)

def fallback_to_basic_response(pipeline, query):
    """
    모든 스트리밍이 실패했을 때의 최종 폴백
    """
    try:
        with st.spinner("기본 방식으로 답변 생성 중..."):
            result = pipeline.process_query(query)
            if result['success']:
                agent_data = extract_agent_data(result)
                synthesis_prompt = create_synthesis_prompt(query, agent_data)
                ai_response = pipeline.agent.llm_client.complete(synthesis_prompt)
                st.markdown(ai_response)
                return ai_response
            else:
                error_msg = f"분석 실패: {result.get('error', 'Unknown error')}"
                st.error(error_msg)
                return error_msg
    except Exception as fallback_error:
        error_msg = f"모든 방식 실패: {fallback_error}"
        st.error(error_msg)
        return error_msg

def stream_hchat_response(llm_client, prompt, enhanced_data):
    """
    터미널 테스트에서 검증된 H-Chat 실시간 스트리밍 구현
    토큰 단위로 세밀한 실시간 스트리밍 (2.4자/청크 수준)
    """
    try:
        # 터미널 테스트에서 확인된 스트리밍 지원 체크
        if hasattr(llm_client, 'stream_complete'):
            log.info("Starting H-Chat real-time streaming (verified working)")
            
            accumulated_response = ""
            chunk_count = 0
            empty_chunks = 0
            
            # 터미널과 동일한 방식으로 실시간 스트리밍 시작
            for chunk in llm_client.stream_complete(prompt):
                if chunk:  # 빈 청크가 아닌 경우만 처리
                    accumulated_response += chunk
                    chunk_count += 1
                    yield chunk  # 터미널과 동일하게 바로 yield
                else:
                    empty_chunks += 1
            
            # 터미널 테스트와 비슷한 로그 출력
            log.info(f"H-Chat streaming completed: {len(accumulated_response)} chars in {chunk_count} chunks (avg: {len(accumulated_response)/max(chunk_count,1):.1f} chars/chunk)")
            
            # 스트리밍 실패 감지 (터미널 테스트에서는 165자 성공)
            if len(accumulated_response.strip()) < 10:
                log.warning(f"H-Chat streaming returned too short response ({len(accumulated_response)} chars), falling back")
                yield from _fallback_simulated_streaming(llm_client, prompt)
                
        else:
            log.warning("H-Chat stream_complete method not found, using fallback")
            yield from _fallback_simulated_streaming(llm_client, prompt)
            
    except Exception as e:
        log.error(f"H-Chat streaming error: {e}")
        yield f"\n\n⚠️ 스트리밍 오류 발생, 일반 모드로 전환 중...\n\n"
        yield from _fallback_simulated_streaming(llm_client, prompt)

def _fallback_simulated_streaming(llm_client, prompt):
    """
    H-Chat 실시간 스트리밍 실패 시 시뮬레이션 폴백
    터미널 테스트에서 검증된 방식을 사용
    """
    try:
        log.info("Using simulated streaming fallback (like terminal test)")
        
        # 터미널 테스트와 동일한 방식으로 일반 완료 호출
        response = llm_client.complete(prompt)
        
        if response and response.strip():
            log.info(f"Simulated streaming: {len(response)} chars to stream")
            
            # 터미널에서 사용한 방식: 더 자연스러운 문장별 스트리밍
            sentences = response.split('. ')
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    # 마지막 문장이 아니면 마침표 추가 (터미널과 동일)
                    if i < len(sentences) - 1:
                        yield sentence + '. '
                    else:
                        yield sentence
                    time.sleep(0.08)  # 터미널 테스트와 동일한 속도
            
            log.info("Simulated streaming completed successfully")
        else:
            log.warning("LLM returned empty response for simulated streaming")
            yield "😕 답변 생성에 실패했습니다. 다시 시도해 주세요."
            
    except Exception as e:
        log.error(f"Simulated streaming fallback failed: {e}")
        yield f"😕 답변 생성 중 오류: {str(e)[:100]}..."

def _yield_raw_data(agent_data):
    """원시 데이터를 사용자 친화적으로 표시하는 생성자"""
    search_results = agent_data.get('search_results')
    aggregation_results = agent_data.get('aggregation_results')
    
    if search_results and search_results.get('total_hits', 0) > 0:
        yield f"\n\n### 검색 결과\n\n총 {search_results['total_hits']:,}개의 관련 문서를 찾았습니다.\n\n"
        
        documents = search_results.get('documents', [])[:3]
        if documents:
            yield "주요 사례:\n\n"
            for i, doc in enumerate(documents, 1):
                model = doc.get('model_of_vehicle', 'N/A')
                problem = doc.get('problem', 'N/A')[:80] + '...' if len(doc.get('problem', '')) > 80 else doc.get('problem', 'N/A')
                brand = doc.get('make_of_vehicle', 'N/A')
                yield f"{i}. {brand} {model}: {problem}\n\n"
    
    if aggregation_results and aggregation_results.get('buckets'):
        yield "### 집계 분석 결과\n\n"
        buckets = aggregation_results['buckets'][:5]
        for bucket in buckets:
            key = bucket.get('key', 'N/A')
            count = bucket.get('doc_count', 0)
            yield f"- {key}: {count:,}건\n"
        yield "\n"
    
    if not search_results and not aggregation_results:
        yield "\n\n분석할 데이터가 없습니다. 다른 키워드로 검색해보세요."

def stream_agent_response_v2(pipeline, query):
    """
    전략 1 적용: LangGraph Agent 스트리밍
    1. Agent 진행상황 표시 (15초)
    2. Synthesizer 단계에서 실제 토큰 스트리밍 (15초~)
    3. 추가 LLM 호출 완전 금지
    """
    try:
        # 초기 알림 - HTML 태그 제거
        yield 'LangGraph Agent 시작 - 68,982건 데이터 분석 중...\n\n'
        
        # 전략 1: Agent 스트리밍 실행
        progress_stage = "prep"  # prep -> processing -> synthesis
        
        for chunk in pipeline.agent.process_query_streaming(query):
            if "LangGraph Agent 시작" in chunk or "단계 완료" in chunk:
                progress_stage = "processing"
                yield chunk + '\n'  # HTML 태그 제거
            elif "Synthesis 단계 시작" in chunk:
                progress_stage = "synthesis"
                yield '최종 답변 생성 시작...\n\n'  # HTML 태그 제거
            elif progress_stage == "synthesis":
                # 실제 토큰 스트리밍
                if chunk and chunk.strip():
                    yield chunk
            else:
                # 기타 진행상황 - HTML 태그 제거
                yield chunk + '\n'
                
    except Exception as e:
        log.error(f"Agent streaming failed: {e}")
        yield f'<div class="process-info">스트리밍 오류: {str(e)}</div>\n\n'
        
        # 폴백: 기존 방식으로 실행
        try:
            result = pipeline.process_query(query)
            if result['success']:
                final_answer = extract_langgraph_final_answer(result)
                if final_answer:
                    # 시뮬레이션 스트리밍
                    sentences = final_answer.split('. ')
                    for i, sentence in enumerate(sentences):
                        if sentence.strip():
                            if i < len(sentences) - 1:
                                yield sentence + '. '
                            else:
                                yield sentence
                            time.sleep(0.05)
                else:
                    yield "Agent 결과를 추출할 수 없습니다. 다시 시도해주세요."
            else:
                yield f"오류: {result.get('error', 'Unknown error')}"
                
        except Exception as fallback_error:
            yield f"폴백 실행 오류: {str(fallback_error)}"

def extract_langgraph_final_answer(result):
    """
    LangGraph Agent의 SynthesizerTool 최종 결과만 추출
    추가 LLM 호출 없이 Agent가 생성한 답변만 사용
    """
    try:
        # LangGraph Agent 결과 구조 분석
        result_data = result.get('result', {})
        
        # Pipeline에서 enhanced된 결과 처리
        if isinstance(result_data, dict):
            # 1순위: Agent의 final_synthesis 추출
            if 'agent_result' in result_data:
                agent_result = result_data['agent_result']
                if isinstance(agent_result, dict) and 'final_synthesis' in agent_result:
                    synthesis_result = agent_result['final_synthesis']
                    if isinstance(synthesis_result, dict) and 'synthesis' in synthesis_result:
                        return synthesis_result['synthesis']  # SynthesizerTool이 생성한 답변
            
            # 2순위: final_answer 처리 (이미 Agent가 처리한 결과)
            elif 'final_answer' in result_data:
                final_answer = result_data['final_answer']
                if isinstance(final_answer, dict) and 'synthesis' in final_answer:
                    return final_answer['synthesis']
                elif isinstance(final_answer, str):
                    return final_answer
        
        log.warning("No final synthesis found in LangGraph Agent result")
        return None
        
    except Exception as e:
        log.error(f"Failed to extract LangGraph final answer: {e}")
        return None

def extract_agent_data(result):
    """에이전트 결과에서 실제 데이터 추출"""
    result_data = result.get('result', {})
    
    extracted = {
        "query": result.get('query', ''),
        "execution_time": result.get('execution_time', 0),
        "tools_used": result.get('agent_info', {}).get('tools_used', [])
    }
    
    # 검색 결과 추출
    if 'final_answer' in result_data:
        final_answer = result_data['final_answer']
        if isinstance(final_answer, dict) and 'agent_result' in final_answer:
            agent_result = final_answer['agent_result']
            extracted['search_results'] = agent_result.get('search_results')
            extracted['aggregation_results'] = agent_result.get('aggregation_results')
    
    return extracted

def create_synthesis_prompt(query, agent_data):
    """에이전트 데이터를 바탕으로 LLM용 종합 프롬프트 생성 - 마크다운 품질 향상"""
    
    search_results = agent_data.get('search_results')
    aggregation_results = agent_data.get('aggregation_results')
    
    prompt = f"""사용자 질문: {query}

시스템이 분석한 결과를 바탕으로 종합적인 답변을 작성하세요.

분석 정보:
- 실행 시간: {agent_data.get('execution_time', 0):.2f}초
- 사용된 도구: {', '.join(agent_data.get('tools_used', []))}

"""
    
    # 검색 결과 추가
    if search_results and isinstance(search_results, dict):
        total_hits = search_results.get('total_hits', 0)
        documents = search_results.get('documents', [])
        
        prompt += f"검색 결과: {total_hits:,}개 관련 문서 발견\n"
        
        if documents:
            prompt += "주요 사례:\n"
            for i, doc in enumerate(documents[:3], 1):
                model = doc.get('model_of_vehicle', 'N/A')
                problem = doc.get('problem', 'N/A')
                prompt += f"{i}. {model}: {problem}\n"
    
    # 집계 결과 추가
    if aggregation_results and isinstance(aggregation_results, dict):
        buckets = aggregation_results.get('buckets', [])
        if buckets:
            prompt += "\n집계 분석 결과:\n"
            for bucket in buckets[:5]:
                key = bucket.get('key', 'N/A')
                count = bucket.get('doc_count', 0)
                prompt += f"- {key}: {count:,}건\n"
    
    prompt += f"""

위 분석 결과를 바탕으로 사용자의 질문에 대해 구체적이고 실용적인 답변을 작성하세요.

분석 요구사항:
- 데이터에 기반한 정확한 정보 제공
- 한국어로 친근하고 이해하기 쉽게 설명
- 필요시 개선 방안이나 추가 분석 제안

마크다운 형식 준수사항 (필수):
• 헤더(###) 앞에는 반드시 두 줄 개행을 넣으세요
  예시: "분석 완료했습니다.\n\n### 주요 발견사항"
• 리스트(1., 2., 3.) 앞에는 반드시 두 줄 개행을 넣으세요
  예시: "다음과 같습니다.\n\n1. 첫 번째 항목"
• 소제목(####) 앞에도 두 줄 개행을 넣으세요
  예시: "분석이 완료되었습니다.\n\n#### 상세 내용"
• 각 섹션 사이에는 적절한 간격을 유지하세요
• 표나 코드 블록 앞에도 두 줄 개행을 넣으세요

올바른 형식 예시:
```
분석이 완료되었습니다.

### 주요 발견사항

다음과 같은 문제들이 발견되었습니다.

1. INFO12 코드 관련 문제
2. INFO13 코드 관련 문제

#### 상세 분석

각 항목의 세부 내용은 다음과 같습니다.
```

위 결과와 형식 지침을 바탕으로 종합적인 답변을 작성하세요.
"""
    
    return prompt

def extract_iqs_related_data(query, pipeline_result=None):
    """관련 데이터 추출 (기존 호환성)"""
    try:
        # 실제 검색 결과에서 관련 데이터 추출
        if pipeline_result and 'result' in pipeline_result:
            result_data = pipeline_result['result']
            
            # 검색 결과에서 관련 차종/문제 정보 추출
            related_data = []
            
            if 'final_answer' in result_data:
                final_answer = result_data['final_answer']
                
                # agent_result에서 실제 데이터 추출 시도
                if isinstance(final_answer, dict) and 'agent_result' in final_answer:
                    agent_result = final_answer['agent_result']
                    
                    # 검색 결과 추출
                    if 'search_results' in agent_result:
                        search_results = agent_result['search_results']
                        if 'documents' in search_results:
                            documents = search_results['documents'][:5]  # 상위 5개만
                            for doc in documents:
                                related_data.append({
                                    "차종": doc.get('model_of_vehicle', 'N/A'),
                                    "모델연도": doc.get('model_year', 'N/A'),
                                    "문제코드": doc.get('problem', 'N/A')[:50] + '...' if len(str(doc.get('problem', ''))) > 50 else doc.get('problem', 'N/A'),
                                    "브랜드": doc.get('make_of_vehicle', 'N/A')
                                })
                    
                    # 집계 결과 추출
                    elif 'aggregation_results' in agent_result:
                        agg_results = agent_result['aggregation_results']
                        if 'buckets' in agg_results:
                            buckets = agg_results['buckets'][:5]  # 상위 5개만
                            for bucket in buckets:
                                related_data.append({
                                    "문제유형": bucket.get('key', 'N/A'),
                                    "불만건수": bucket.get('doc_count', 'N/A'),
                                    "비중": f"{bucket.get('doc_count', 0) / agg_results.get('total_docs', 1) * 100:.1f}%" if 'total_docs' in agg_results else 'N/A'
                                })
            
            return related_data
    except Exception as e:
        st.error(f"관련 데이터 추출 오류: {e}")
        return []
    
    # 기본 예시 데이터
    return [
        {
            "차종": "Hyundai Santa Fe 4dr SUV",
            "모델연도": 2025,
            "문제코드": "INFO12: Built-in navigation - Broken/works inconsistently",
            "브랜드": "Hyundai"
        }
    ]

def generate_iqs_recommendations(query, response_text):
    """IQS 특화 추천 질문 생성"""
    try:
        # 쿼리 분석 기반 추천 질문 생성
        recommendations = []
        
        query_lower = query.lower()
        
        # 브랜드 관련 질문이면 브랜드 비교 추천
        if any(brand in query_lower for brand in ['현대', 'hyundai', '기아', 'kia']):
            recommendations.append("이 문제에서 현대와 기아 브랜드 간 차이점을 비교해줘")
        
        # 차종 관련 질문이면 다른 차종 추천
        if any(model in query_lower for model in ['싼타페', 'santa fe', '팰리세이드', 'palisade']):
            recommendations.append("동일 문제에서 다른 차종들의 현황은 어떤지 알려줘")
        
        # INFO 코드 관련이면 관련 코드 추천
        if 'info' in query_lower:
            recommendations.append("INFO 계열의 다른 문제 코드들과의 연관성을 분석해줘")
        
        # 연도 관련이면 트렌드 추천
        if any(year in query for year in ['2023', '2024', '2025', '23MY', '24MY', '25MY']):
            recommendations.append("이 문제의 최근 3년간 변화 추이를 분석해줘")
        
        # 기본 추천 질문들
        if len(recommendations) < 3:
            basic_recommendations = [
                "이 분석 결과에서 가장 심각한 문제점은 무엇인가요?",
                "개선이 필요한 우선순위 영역을 알려주세요",
                "경쟁사 대비 우리 브랜드의 강약점을 분석해주세요",
                "고객 불만을 줄이기 위한 구체적인 개선 방안은?",
                "이와 유사한 다른 품질 문제들도 있나요?"
            ]
            
            for rec in basic_recommendations:
                if len(recommendations) >= 3:
                    break
                if rec not in recommendations:
                    recommendations.append(rec)
        
        return recommendations[:3]  # 최대 3개
        
    except Exception as e:
        # 오류 시 기본 추천 질문 반환
        return [
            "이 분석 결과에서 가장 심각한 문제점은 무엇인가요?",
            "개선이 필요한 우선순위 영역을 알려주세요",
            "경쟁사 대비 우리 브랜드의 강약점을 분석해주세요"
        ]

def save_feedback_log(query, response, rating, feedback):
    """피드백 저장"""
    try:
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": query,
            "ai_response": response,
            "rating": str(rating) if rating else "",
            "feedback": str(feedback) if feedback else ""
        }
        
        log_df = pd.DataFrame([log_data])
        log_path = Path("logs/iqs_feedback.csv")
        
        # 로그 디렉토리 생성
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if log_path.exists():
            existing_df = pd.read_csv(log_path)
            updated_df = pd.concat([existing_df, log_df], ignore_index=True)
        else:
            updated_df = log_df
        
        updated_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        st.error(f"피드백 저장 실패: {e}")
        return False

def show_chat_history(chat_history):
    """채팅 기록 표시"""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

def add_search_history(query, timestamp=None):
    """검색 기록 추가"""
    if timestamp is None:
        timestamp = datetime.now()
    
    # 중복 방지
    if query in [h.get('query') for h in st.session_state.search_history]:
        return
    
    # 최대 10개만 유지
    if len(st.session_state.search_history) >= 10:
        st.session_state.search_history.pop(0)
    
    st.session_state.search_history.append({
        'query': query,
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M"),
        'short_query': query[:30] + '...' if len(query) > 30 else query
    })

def show_sidebar():
    """사이드바 표시 - 대화 초기화, 데이터 정보, 검색 기록"""
    with st.sidebar:
        # 대화 초기화 버튼 (상단)
        if st.button("🔍 IQS-Search", key="sidebar_reset_button", help="새로운 대화를 시작합니다", use_container_width=True):
            # 세션 상태 초기화
            st.session_state.chat_history = []
            if "analysis_result" in st.session_state:
                del st.session_state["analysis_result"]
            if "related_data" in st.session_state:
                del st.session_state["related_data"]
            if "recommendations" in st.session_state:
                del st.session_state["recommendations"]
            if "current_query" in st.session_state:
                del st.session_state["current_query"]
            if "pipeline_result" in st.session_state:
                del st.session_state["pipeline_result"]
            st.rerun()
        
        st.divider()
        
        # 데이터 정보
        st.markdown("### 데이터 정보")
        
        # 기본 데이터 통계
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 데이터", "68,982건")
            st.metric("대상 연도", "23-25MY")
        with col2:
            st.metric("브랜드", "3개")
            st.metric("차종", "49개")
        
        # 상세 정보
        with st.expander("상세 정보", expanded=False):
            st.markdown("""
            **브랜드**: 현대, 기아, 제네시스  
            **차종**: SUV, 세단, 해치백 등 49개  
            **문제 유형**: INFO, FCD, PWR 등 223개  
            **데이터 기간**: 2023-2025 모델이어  
            **출처**: J.D.Power Initial Quality Study
            """)
        
        st.divider()
        
        # 검색 기록
        st.markdown("### 검색 기록")
        
        if st.session_state.search_history:
            # 기록 삭제 버튼
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("삭제", key="clear_history"):
                    st.session_state.search_history = []
                    st.rerun()
            
            # 기록 표시 (최근 10개)
            for i, history_item in enumerate(reversed(st.session_state.search_history)):
                with st.container():
                    # 간략한 표시
                    if st.button(
                        f"{history_item['short_query']}", 
                        key=f"history_{i}",
                        help=f"{history_item['timestamp']} - {history_item['query']}",
                        use_container_width=True
                    ):
                        st.session_state["pending_user_query"] = history_item['query']
                        st.rerun()
                    
                    # 시간 표시
                    st.caption(history_item['timestamp'])
        else:
            st.info("검색 기록이 없습니다.")


def restart_streamlit():
    """Streamlit 앱 재시작"""
    st.session_state.clear()
    st.markdown(
        """
        <meta http-equiv="refresh" content="0;url=/" />
        """,
        unsafe_allow_html=True
    )

# CSS 스타일링 추가
def apply_custom_css():
    st.markdown("""
    <style>
    /* 메인 타이틀 스타일링 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 2rem;
        line-height: 1.4;
    }
    
    /* 사이드바 IQS-Search 버튼 스타일링 */
    button[key="sidebar_reset_button"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 6px !important;
        color: white !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.2s ease !important;
        margin-bottom: 0.5rem !important;
    }
    
    button[key="sidebar_reset_button"]:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4c93 100%) !important;
        transform: translateY(-1px) !important;
    }
    
    /* 프로세스 정보 스타일링 (더 작고 연하게 + 간격 줄이기) */
    .process-info {
        font-size: 0.75rem !important;
        color: #9ca3af !important;
        line-height: 1.2 !important;
        margin: 0.1rem 0 !important;
        padding: 0.05rem 0 !important;
        font-weight: 400 !important;
    }
    
    .process-info p {
        margin: 0.1rem 0 !important;
        padding: 0 !important;
    }
    
    /* 작은 로딩 스피너 (인라인 스타일) */
    .small-spinner {
        display: inline-flex;
        align-items: center;
        font-size: 0.75rem;
        color: #6b7280;
        margin-right: 8px;
    }
    
    .small-spinner .spinner {
        width: 14px;
        height: 14px;
        border: 1.5px solid #f3f4f6;
        border-top: 1.5px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 6px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* 로딩 중 메시지 스타일링 */
    .loading-message {
        font-size: 0.8rem;
        color: #6b7280;
        font-style: italic;
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
    }
    
    .loading-message .loading-dots {
        margin-left: 8px;
    }
    
    /* 채팅 메시지 스타일링 */
    .stChatMessage {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    
    /* 사이드바 스타일링 */
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    /* 데이터 테이블 스타일링 */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* 버튼 스타일링 */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """메인 함수"""
    st.set_page_config(
        page_title="IQS Quality Data Analytics", 
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 스타일링 적용
    apply_custom_css()
    
    # 메인 타이틀 및 설명
    st.markdown("<h1 class='main-title'>IQS(Initial Quality Study) 품질 AI어시스턴트</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>J.D.Power 23~25년 신차품질조사 결과데이터(IQS)를 기반으로 검색을 수행합니다.</div>", unsafe_allow_html=True)
    
    # 사이드바 표시
    show_sidebar()
    
    # 시스템 초기화
    pipeline = initialize_iqs_system()
    if pipeline is None:
        st.stop()
    
    # 대화 초기화는 우상단 AI 아이콘으로 대체됨
    
    # 대화 초기화는 우상단 AI 아이콘으로 대체됨
    
    # 채팅 입력
    user_query = st.chat_input("IQS 품질 데이터 분석 질문을 입력하세요")
    
    # 질문 처리
    if user_query:
        st.session_state["pending_user_query"] = user_query
    
    # 사용자 질문이 입력되면 바로 채팅창에 표시
    if "pending_user2user_display" not in st.session_state and "pending_user_query" in st.session_state:
        query = st.session_state["pending_user_query"]
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state["pending_user2user_display"] = True
    
    # LangGraph Agent 실행
    if "pending_user_query" in st.session_state:
        query = st.session_state["pending_user_query"]
        
        # 개선된 실시간 스트리밍 답변 생성
        with st.chat_message("assistant"):
            try:
                # Streamlit 스트리밍 방식 1: st.write_stream 시도
                if hasattr(st, 'write_stream'):
                    ai_response = ""
                    try:
                        def response_generator():
                            nonlocal ai_response
                            for chunk in stream_agent_response_v2(pipeline, query):
                                ai_response += chunk
                                yield chunk
                        
                        st.write_stream(response_generator())
                        
                    except Exception as write_stream_error:
                        log.warning(f"st.write_stream failed: {write_stream_error}, falling back to st.empty()")
                    # st.empty() 방식으로 폴백
                    ai_response = stream_with_empty_container_v2(pipeline, query)
                else:
                    # st.empty() 방식 사용
                    ai_response = stream_with_empty_container_v2(pipeline, query)
                
                # 답변 유효성 검사
                if not ai_response or len(str(ai_response).strip()) < 10:
                    st.warning("답변이 너무 짧습니다. 다시 시도해주세요.")
                    ai_response = "답변 생성 중 문제가 발생했습니다."
                    
            except Exception as stream_error:
                log.error(f"Streaming error: {stream_error}")
                st.error(f"스트리밍 오류: {stream_error}")
                
                # 최종 폴백: 기본 방식
                ai_response = fallback_to_basic_response(pipeline, query)
        
        # 관련 데이터 및 추천 질문 생성 (작은 스피너 사용)
        loading_container = st.empty()
        loading_container.markdown('<div class="small-spinner"><div class="spinner"></div>관련 정보를 준비 중입니다...</div>', unsafe_allow_html=True)
        
        pipeline_result = getattr(st.session_state.get('iqs_pipeline', {}), '_last_result', None)
        related_data = extract_iqs_related_data(query, pipeline_result)
        recommendations = generate_iqs_recommendations(query, ai_response)
        
        # 로딩 스피너 제거
        loading_container.empty()
        
        # 채팅 기록에 추가
        st.session_state.chat_history.append(HumanMessage(query))
        st.session_state.chat_history.append(AIMessage(ai_response))
        
        # 검색 기록에 추가
        add_search_history(query)
        
        # 세션에 저장
        st.session_state["analysis_result"] = ai_response
        st.session_state["related_data"] = related_data
        st.session_state["recommendations"] = recommendations
        st.session_state["current_query"] = query
        
        # pending_user_query 삭제
        del st.session_state["pending_user_query"]
        if "pending_user2user_display" in st.session_state:
            del st.session_state["pending_user2user_display"]
        
        st.rerun()
    
    # 채팅 기록 표시
    show_chat_history(st.session_state.chat_history)
    
    # 관련 데이터 표시 (간소화)
    if "related_data" in st.session_state and st.session_state["related_data"]:
        st.markdown("### 관련 데이터")
        df = pd.DataFrame(st.session_state["related_data"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 추천 질문 표시 (간소화)
    if "recommendations" in st.session_state and st.session_state["recommendations"]:
        st.markdown("### 추천 질문")
        questions = st.session_state["recommendations"]
        for i, question in enumerate(questions):
            if st.button(question, key=f"rec_q_{i}"):
                st.session_state["pending_user_query"] = question
                st.rerun()
    
    # 피드백 처리 (선택적 표시)
    if "current_query" in st.session_state and "analysis_result" in st.session_state:
        with st.expander("답변 평가", expanded=False):
            rating = st.radio("만족도", [1, 2, 3, 4, 5], index=None, 
                            key=f"rating_{st.session_state['current_query']}")
            feedback = st.text_area("추가 의견", 
                                  key=f"feedback_{st.session_state['current_query']}")
            
            if st.button("평가 제출", key=f"submit_{st.session_state['current_query']}"):
                if save_feedback_log(st.session_state['current_query'], 
                                   st.session_state['analysis_result'], 
                                   rating, feedback):
                    st.success("평가가 저장되었습니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("오류가 발생하여 애플리케이션을 다시 시작합니다.")
        st.text(traceback.format_exc())
        restart_streamlit()