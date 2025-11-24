#!/usr/bin/env python3
"""
IQS Agent 대화형 테스트 시스템
한 번 초기화하고 반복적으로 쿼리 테스트
"""
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent.agent_pipeline import AgentPipeline

def initialize_system():
    """시스템 한 번 초기화 - 단순화된 Pipeline로"""
    print("IQS Agent 시스템 초기화 (Pipeline + LangGraph Agent)")
    print("=" * 65)
    
    print("1. LangGraph Agent 초기화 중...")
    print("   - H-Chat GPT-4o 연결")
    print("   - 도구들 로딩 (Glossary, Search, Aggregator, Reranker, Synthesizer)")
    print("   - StateGraph 워크플로 구성")
    print("2. Pipeline 래퍼 초기화 중...")
    print("   - 히스토리 관리 시스템")
    print("   - 통계 수집 시스템")
    print("   - 배치 처리 시스템")
    
    start_time = time.time()
    
    try:
        # Pipeline이 내부적으로 LangGraph Agent를 초기화
        pipeline = AgentPipeline(llm_type="h-chat")
        
        init_time = time.time() - start_time
        
        print(f"\n초기화 완료! (소요 시간: {init_time:.2f}초)")
        
        # 시스템 정보 표시
        if hasattr(pipeline.agent, 'tools'):
            agent_tools = list(pipeline.agent.tools.keys())
            print(f"Agent 도구: {agent_tools}")
        
        print(f"Pipeline 기능: 히스토리, 통계, 배치처리, 데이터로딩")
        print(f"아키텍처: Pipeline(Wrapper) -> LangGraphIQSAgent(Core)")
        print(f"시스템 준비 완료!")
        
        return pipeline
        
    except Exception as e:
        print(f"초기화 실패: {e}")
        print("해결 방법: .env 파일에서 HCHAT_API_KEY 확인")
        return None

def process_query_standard(pipeline, query):
    """기존 방식 테스트 (호환성 검증용)"""
    print(f"\n[STANDARD MODE] 쿼리 처리 중: {query}")
    print("-" * 60)
    
    try:
        result = pipeline.process_query(query)
        
        if result['success']:
            execution_time = result.get('execution_time', 0)
            print(f"처리 성공! (총 실행 시간: {execution_time:.2f}초)")
            
            # Agent 실행 정보
            agent_info = result.get('agent_info', {})
            print(f"\n[AGENT INFO]")
            print(f"Agent 성공: {agent_info.get('agent_success', False)}")
            print(f"Agent 단계 수: {agent_info.get('steps_executed', 0)}")
            print(f"Agent 사용 도구: {agent_info.get('tools_used', [])}")
            
            return True
        else:
            print(f"처리 실패: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def process_query_streaming(pipeline, query):
    """새로운 스트리밍 방식 테스트 - 전략 1 적용"""
    print(f"\n[STREAMING MODE] 쿼리 처리 중: {query}")
    print("-" * 60)
    print("\n=== 스트리밍 시작 ===")
    
    try:
        chunk_count = 0
        accumulated_response = ""
        synthesis_started = False
        
        # LangGraph Agent 스트리밍 실행
        for chunk in pipeline.agent.process_query_streaming(query):
            chunk_count += 1
            
            # 진행상황 vs 실제 답변 구분
            if "단계 완료" in chunk or "시작 완료" in chunk:
                print(f"[PROGRESS] {chunk}")
            elif "Synthesis 단계 시작" in chunk:
                synthesis_started = True
                print(f"[SYNTHESIS START] {chunk}")
                print("\n=== 실시간 토큰 스트리밍 시작 ===")
            elif synthesis_started:
                # 실제 답변 토큰들
                print(chunk, end='', flush=True)
                accumulated_response += chunk
            else:
                print(f"[INFO] {chunk}")
        
        print("\n\n=== 스트리밍 완료 ===")
        print(f"총 청크: {chunk_count}개")
        print(f"답변 길이: {len(accumulated_response)}자")
        print(f"첫 100자: {accumulated_response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n스트리밍 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def process_query(pipeline, query):
    """메인 쿼리 처리 - 스트리밍 방식만 테스트 (효율성)"""
    print(f"\n{'='*80}")
    print(f"쿼리 처리: {query}")
    print(f"{'='*80}")
    
    # 전략 1: 스트리밍 방식만 테스트 (중복 실행 방지)
    print("\n🚀 LangGraph Agent 스트리밍 테스트:")
    streaming_success = process_query_streaming(pipeline, query)
    
    if streaming_success:
        print("\n\u2705 스트리밍 방식 정상 작동")
    else:
        print("\n\u274c 스트리밍 방식 실패")
        
    return streaming_success

def process_query_comparison(pipeline, query):
    """비교 테스트 - 두 방식 모두 테스트 (선택적 사용)"""
    print(f"\n{'='*80}")
    print(f"비교 테스트: {query}")
    print(f"{'='*80}")
    print("⚠️ 주의: 같은 쿼리를 두 번 실행합니다 (비효율적)")
    
    # 1. 기존 방식 테스트
    print("\n1️⃣ 기존 방식 (Standard Mode) 테스트:")
    standard_success = process_query_standard(pipeline, query)
    
    # 2. 새로운 스트리밍 방식 테스트
    print("\n\n2️⃣ 새로운 스트리밍 방식 (Strategy 1) 테스트:")
    streaming_success = process_query_streaming(pipeline, query)
    
    # 3. 결과 비교
    print("\n\n3️⃣ 테스트 결과 비교:")
    print(f"기존 방식: {'성공' if standard_success else '실패'}")
    print(f"스트리밍 방식: {'성공' if streaming_success else '실패'}")
    
    if standard_success and streaming_success:
        print("✅ 모든 방식 정상 작동")
    elif standard_success:
        print("⚠️ 스트리밍 방식에 문제 있음")
    elif streaming_success:
        print("⚠️ 기존 방식에 문제 있음")
    else:
        print("❌ 모든 방식 실패")
        
    return standard_success or streaming_success

def show_stats(query_history, pipeline=None):
    """세션 통계 표시 - Pipeline 버전"""
    if not query_history:
        print("아직 처리된 쿼리가 없습니다.")
        return
    
    total = len(query_history)
    successful = sum(1 for h in query_history if h['success'])
    
    print(f"""
세션 통계 (Interactive Test):
  - 총 쿼리: {total}개
  - 성공: {successful}개  
  - 성공률: {successful/total*100:.1f}%
""")
    
    if successful > 0:
        avg_time = sum(h['execution_time'] for h in query_history if h['success']) / successful
        print(f"  - 평균 실행 시간: {avg_time:.2f}초")
    
    # Pipeline 통계 추가 표시
    if pipeline:
        try:
            pipeline_stats = pipeline.get_statistics()
            agent_info = pipeline_stats.get('agent_info', {})
            
            print(f"\nPipeline 시스템 정보:")
            print(f"  - Agent 타입: {agent_info.get('agent_type')}")
            print(f"  - LLM 타입: {agent_info.get('llm_type')}")
            print(f"  - 사용 가능 도구: {', '.join(agent_info.get('tools_available', []))}")
            
            tool_usage = agent_info.get('tools_usage_count', {})
            if tool_usage:
                print(f"  - 도구 사용 통계: {tool_usage}")
                
        except Exception as e:
            print(f"  Pipeline 통계 수집 오류: {e}")
    
    print(f"\n최근 쿼리 기록:")
    for i, record in enumerate(query_history[-5:], 1):  # 최근 5개만
        status = "SUCCESS" if record['success'] else "FAILED"
        query_short = record['query'][:40] + "..." if len(record['query']) > 40 else record['query']
        print(f"  {i}. {status} {query_short} ({record['execution_time']:.1f}초)")

def test_agent_architecture():
    """LangGraph Agent 아키텍처 상세 테스트"""
    print("\n" + "="*80)
    print("LangGraph Agent 아키텍처 상세 분석")
    print("="*80)
    
    try:
        pipeline = initialize_system()
        if not pipeline:
            return False
        
        agent = pipeline.agent
        
        print("\n📋 Agent 구조 정보:")
        print(f"Agent 타입: {type(agent).__name__}")
        print(f"사용 가능한 도구: {list(agent.tools.keys())}")
        
        # 도구별 상세 정보
        print("\n🔧 도구별 상세 정보:")
        for tool_name, tool in agent.tools.items():
            print(f"  {tool_name}: {type(tool).__name__}")
            if hasattr(tool, 'execute_streaming'):
                print(f"    ✅ 스트리밍 지원")
            else:
                print(f"    ❌ 스트리밍 미지원")
        
        # LLM 클라이언트 정보
        print("\n💬 LLM 클라이언트 정보:")
        llm_client = agent.llm_client
        print(f"LLM 타입: {type(llm_client).__name__}")
        if hasattr(llm_client, 'stream_complete'):
            print("✅ H-Chat 스트리밍 지원")
        else:
            print("❌ H-Chat 스트리밍 미지원")
            
        # StateGraph 정보
        print("\n🔄 StateGraph 정보:")
        graph_info = agent.get_graph_state_info()
        print(f"노드 수: {len(graph_info['nodes'])}")
        print(f"노드 목록: {graph_info['nodes']}")
        
        return True
        
    except Exception as e:
        print(f"아키텍처 테스트 실패: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """메인 실행 함수 - 아키텍처 + 스트리밍 테스트"""
    print("IQS 품질 데이터 검색 시스템 종합 테스트")
    print("=" * 60)
    
    # 0. 아키텍처 상세 분석
    print("\n0️⃣ LangGraph Agent 아키텍처 분석...")
    if not test_agent_architecture():
        print("아키텍처 테스트 실패. 계속 진행합니다.")
    
    # 1. 시스템 한 번 초기화
    pipeline = initialize_system()
    if not pipeline:
        print("시스템 초기화에 실패했습니다.")
        return
    
    # 2. 세션 시작
    query_history = []
    
    print(f"\n테스트 세션 시작!")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.")
    
    # 3. REPL 루프
    while True:
        try:
            # 사용자 입력 받기
            print("\n명령어:")
            print("  - 질문 입력: 스트리밍 방식으로 질문 처리")
            print("  - 'test': 샘플 쿼리 스트리밍 테스트")
            print("  - 'compare': 기존 vs 스트리밍 비교 테스트 (비효율적)")
            print("  - 'arch': 아키텍처 재분석")
            print("  - 'stats': 통계 보기")
            print("  - 'exit': 종료")
            
            user_input = input(f"\n입력 > ").strip()
            
            # 종료 명령
            if user_input.lower() in ['exit', 'quit', '종료']:
                print("테스트를 종료합니다.")
                show_stats(query_history)
                break
            
            # 빈 입력 무시
            if not user_input:
                continue
            
            # 명령어 처리
            if user_input.lower() == 'stats':
                show_stats(query_history, pipeline)
                continue
            elif user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif user_input.lower() == 'arch':
                test_agent_architecture()
                continue
            elif user_input.lower() == 'test':
                # 샘플 쿼리 스트리밍 테스트
                sample_queries = [
                    "INFO12 코드 관련 문제 찾아줘",
                    "2025년 산타페 타이어 문제",
                    "현대브랜드 인포테인먼트 불만 상위 3개"
                ]
                for i, sample in enumerate(sample_queries, 1):
                    print(f"\n=== 샘플 스트리밍 테스트 {i}/3 ===")
                    success = process_query(pipeline, sample)  # 스트리밍만 테스트
                    
                    # 기록 저장
                    query_history.append({
                        'query': sample,
                        'success': success,
                        'execution_time': 0,
                        'timestamp': time.strftime("%H:%M:%S")
                    })
                    
                    if i < len(sample_queries):
                        input("\nEnter를 눌러서 다음 테스트 진행...")
                continue
            elif user_input.lower() == 'compare':
                # 비교 테스트 (선택적)
                print("\n비교 테스트를 위한 쿼리를 입력하세요:")
                compare_query = input("> ").strip()
                if compare_query:
                    success = process_query_comparison(pipeline, compare_query)
                    query_history.append({
                        'query': f"[COMPARE] {compare_query}",
                        'success': success,
                        'execution_time': 0,
                        'timestamp': time.strftime("%H:%M:%S")
                    })
                continue
            
            # 일반 쿼리 처리
            success = process_query(pipeline, user_input)
            
            # 실행 시간은 통합 테스트에서 측정하지 않음 (개별 측정됨)
            execution_time = 0
            
            # 기록 저장
            query_history.append({
                'query': user_input,
                'success': success,
                'execution_time': execution_time,
                'timestamp': time.strftime("%H:%M:%S")
            })
            
        except KeyboardInterrupt:
            print(f"\n\nCtrl+C로 중단되었습니다.")
            show_stats(query_history, pipeline)
            break
        except Exception as e:
            print(f"\n예상치 못한 오류: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"프로그램 실행 중 오류: {e}")
        sys.exit(1)