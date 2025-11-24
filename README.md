# IQS Agent 시스템

현대자동차 IQS 품질 데이터 68,982건에 대한 LangGraph 기반 지능형 에이전트 시스템입니다.
Plan-and-Execute 패턴으로 동적이고 유연한 자연어 쿼리 처리를 제공합니다.

## 시스템 개요

### 핵심 특징
- **동적 Plan-and-Execute**: LLM이 쿼리별로 최적의 실행 계획 수립
- **유연한 도구 선택**: 상황에 따른 지능적 도구 조합 및 실행 순서 결정
- **실제 데이터 연동**: H-Chat GPT-4o, Elasticsearch, Ollama BGE-M3 통합
- **고성능 의미 검색**: BGE-M3 1024차원 벡터로 정확한 의미 기반 검색
- **한-영 자동 번역**: 용어 사전 기반 쿼리 최적화
- **이중 저장 시스템**: Elasticsearch(메인) + 로컬 파일(폴백) 안정성
- **대화형 테스트**: REPL 인터페이스로 빠른 반복 테스트

### 지원 쿼리 유형
- 집계 분석: "인포테인먼트 문제 상위 3개는?"
- 비교 분석: "2024년과 2025년 엔진 소음 문제 비교"
- 코드 해석: "FCD35 코드의 의미는?"
- 원인 분석: "타이어 진동 문제의 주요 원인은?"

## 에이전틱 시스템 아키텍처

### LangGraph 기반 Plan-and-Execute 패턴

```
사용자 쿼리
    ↓
┌─────────────────────────────────────────────┐
│              PLANNING PHASE                 │
│                                             │
│  LLM (H-Chat GPT-4o)이 쿼리 분석:           │
│  • 의도 파악 (검색? 집계? 비교?)             │
│  • 필요한 단계 결정                          │
│  • 각 단계별 최적 도구 선택                  │
│  • 동적 실행 순서 생성                       │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│             EXECUTION PHASE                 │
│                                             │
│  LLM이 상황별 도구 호출:                     │
│  • A → B → C (선형)                         │
│  • A → C (단계 생략)                        │
│  • A → A → B (도구 반복)                    │
│  • B → A → C (순서 변경)                    │
│                                             │
│  실행 중 동적 판단:                          │
│  • 이전 결과 기반 다음 도구 선택              │
│  • 실패 시 대안 전략 수립                    │
│  • 필요시 추가 단계 삽입                     │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│            SYNTHESIS PHASE                  │
│                                             │
│  모든 결과를 종합하여 자연어 답변 생성        │
└─────────────────────────────────────────────┘
```

### 동적 도구 선택 메커니즘

LLM은 매 순간 현재 상태와 목표를 비교하여 최적의 다음 행동을 결정합니다:

**상황 1: 한글 쿼리**
```
LLM 판단: "한글이네? glossary_lookup으로 번역할까, 아니면 내가 직접 번역할까?"
→ 도구 description 비교 후 최적 선택
```

**상황 2: 검색 결과 부족**
```
LLM 판단: "결과가 10개뿐이네. hybrid_search를 다른 키워드로 다시 호출해야겠다"
→ 같은 도구 재호출 결정
```

**상황 3: 집계 후 상세 정보 필요**
```
LLM 판단: "Top 3는 나왔는데 각각의 사례가 필요하네. 각 문제별로 다시 검색하자"
→ 계획에 없던 추가 단계 동적 추가
```

## 파일 구조

```
IQS_Search_3_agent/
├── src/
│   ├── agent/                    # 에이전트 시스템 핵심
│   │   ├── agent_pipeline.py     # 전체 파이프라인 통합
│   │   ├── orchestrator.py       # Plan-and-Execute 오케스트레이터
│   │   ├── tools.py              # 도구 모음 (검색/집계/재순위화/종합)
│   │   ├── llm_client.py         # LLM 클라이언트 (H-Chat)
│   │   └── glossary_tool.py      # 한-영 용어 번역 도구
│   │
│   ├── search/                   # 검색 엔진
│   │   ├── semantic_search.py    # 시맨틱 검색 (Faiss + ES)
│   │   ├── embeddings/
│   │   │   └── embedding_model.py # BGE-M3 임베딩 (Ollama)
│   │   └── elasticsearch/
│   │       └── es_client.py      # Elasticsearch 클라이언트
│   │
│   ├── reranker/
│   │   └── reranker.py           # 검색 결과 재순위화
│   │
│   └── utils/
│       └── logger.py             # 로깅 시스템
│
├── data/
│   ├── glossary/
│   │   └── kor_eng_automotive_terms.json  # 한-영 용어 사전
│   └── processed/                # 처리된 IQS 데이터
│
├── config/
│   └── settings.py               # 시스템 설정
│
├── interactive_test.py           # 대화형 테스트 인터페이스
└── README.md
```

### 핵심 구성 요소 역할

**Agent Pipeline** (`src/agent/agent_pipeline.py`)
- 전체 시스템의 진입점
- LLM 클라이언트, 도구들, 오케스트레이터 통합 관리

**Tools** (`src/agent/tools.py`)
- GlossaryTool: 한-영 용어 번역
- HybridSearchTool: Elasticsearch + 시맨틱 검색
- AggregatorTool: 데이터 집계 및 통계
- RerankerTool: 결과 재순위화
- SynthesizerTool: 최종 답변 생성

**LLM Client** (`src/agent/llm_client.py`)
- H-Chat GPT-4o API 연동
- JSON 구조화 응답 처리
- 오류 처리 및 재시도 로직

**Semantic Search** (`src/search/semantic_search.py`)
- Elasticsearch와 Faiss 통합 검색
- 68,982개 문서 BGE-M3 임베딩 인덱스 관리
- 하이브리드 검색 (키워드 + 시맨틱)
- 이중 저장: ES 메인 + 로컬 폴백

## 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 클론 및 가상환경 생성
python -m venv venv-iqs3
venv-iqs3\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
copy .env.example .env
# .env 파일에서 HCHAT_API_KEY 설정
```

### 2. 시스템 검증

```bash
# 1. Elasticsearch 서버 확인 (Docker)
docker ps --filter "name=es"

# 2. Elasticsearch 데이터 확인
curl -X GET "localhost:9200/_cat/indices?v" | findstr iqs

# 3. H-Chat 연결 테스트
python -c "from src.agent.llm_client import LLMClientFactory; client = LLMClientFactory.create('h-chat'); print('H-Chat 연결 성공!')"

# 4. Elasticsearch 내 IQS 데이터 로딩 확인
python -c "from src.search.elasticsearch.es_client import ElasticSearchClient; client = ElasticSearchClient(); result = client.client.count(index='iqs_verbatim'); print(f'IQS 데이터: {result[\"count\"]:,}개 문서 로드됨')"

# 5. BGE-M3 임베딩 모델 로딩 테스트 (Ollama 필요)
python -c "from src.search.embeddings.embedding_model import EmbeddingModel; model = EmbeddingModel(model_type='ollama'); print(f'BGE-M3 모델 로드됨 (차원: {model.get_dimension()})')"

# 6. 용어 사전 로딩 확인
python -c "from src.agent.glossary_tool import GlossaryLookupTool; tool = GlossaryLookupTool(); result = tool.execute('인포테인먼트 문제'); print(f'용어 번역: {result[\"original_query\"]} -> {result[\"enhanced_query\"]}')"

# 7. 전체 시스템 통합 테스트
python -c "from src.agent.agent_pipeline import AgentPipeline; pipeline = AgentPipeline(llm_type='h-chat'); print('전체 시스템 초기화 성공!')"
```

### 3. 대화형 테스트 실행

```bash
python interactive_test.py
```

### 4. Streamlit 실행

```bash
python run_iqs_app.py
```

## BGE-M3 임베딩 시스템

### 현재 구성
- **임베딩 모델**: Ollama BGE-M3 (1024차원)
- **총 데이터**: 68,982개 문서 + BGE-M3 벡터
- **저장 위치**: 이중 저장 시스템
  - **Elasticsearch**: 1,599.2 MB (메인 저장소)
  - **로컬 파일**: 2.1 GB (폴백 + 캐시)

### 저장 구조

#### Elasticsearch (메인)
```
iqs_verbatim 인덱스:
├── 문서 텍스트 (verbatim_text)
├── 메타데이터 (make, model, problem 등)
└── BGE-M3 벡터 (verbatim_vector: 1024차원)
```

#### 로컬 파일 (폴백)
```
C:/Users/.../AppData/Local/Temp/iqs_embeddings/:
├── documents.json      (488.1 MB) - 문서 메타데이터
├── embeddings.npy      (750.0 MB) - BGE-M3 벡터 배열
├── faiss.index         (375.0 MB) - Faiss 검색 인덱스
└── embedding_cache.pkl (497.3 MB) - 임베딩 캐시
```

### BGE-M3 특징
- **다국어 지원**: 한글/영어 동시 지원
- **고품질 벡터**: 1024차원 의미 벡터
- **빈 텍스트 처리**: 자동 플레이스홀더 또는 제로 벡터
- **캐싱 시스템**: 63,000+ 벡터 캐시로 빠른 응답

### 캐시 관리

임베딩 캐시 문제가 발생할 경우 `find_cache.py` 유틸리티 사용:

```bash
# 캐시 위치 및 내용 확인
python find_cache.py

# 모든 캐시 삭제 (차원 불일치 해결)
# y를 입력하여 캐시 삭제 선택
```

**캐시 문제 증상:**
- 임베딩 차원 불일치 (1536 vs 1024)
- "inhomogeneous shape" 오류
- BGE-M3 모델 변경 후 이전 벡터 반환


## 개발 가이드

### 새로운 도구 추가

```python
# src/agent/tools.py에 새 도구 클래스 추가
class NewTool(BaseTool):
    def __init__(self):
        self.name = "new_tool"
        self.description = "새로운 도구의 기능 설명"
    
    def execute(self, **kwargs):
        # 도구 실행 로직
        return {"result": "실행 결과"}

# src/agent/orchestrator.py에 도구 등록
tools['new_tool'] = NewTool()
```

## 설정 관리

### 환경 변수 (.env)

```bash
# LLM 설정
AGENT_LLM_TYPE=h-chat
HCHAT_API_KEY=your_api_key_here
HCHAT_MODEL=gpt-4o
HCHAT_BASE_URL=https://h-chat-api.autoever.com/v2/api

# 검색 엔진 설정  
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=iqs_verbatim
ES_USE_ELASTICSEARCH=true

# BGE-M3 임베딩 설정
EMBEDDING_MODEL_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
EMBEDDINGS_DIR=data/embeddings

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=logs/iqs_agent.log
```

### 성능 튜닝

```python
# config/settings.py에서 조정 가능한 설정들
ES_VECTOR_DIMS = 1024           # BGE-M3 임베딩 차원
DEFAULT_SEARCH_LIMIT = 50       # 기본 검색 결과 수
RERANKER_TOP_K = 10            # 재순위화 대상 수
LLM_TIMEOUT = 30               # LLM 응답 타임아웃
OLLAMA_BATCH_SIZE = 8          # BGE-M3 배치 크기
OLLAMA_TIMEOUT = 30            # Ollama API 타임아웃
```

## 트러블슈팅

### 임베딩 모델 관련

**문제**: Ollama BGE-M3 연결 실패
```bash
# Ollama 서버 상태 확인
ollama list
ollama serve

# BGE-M3 모델 다운로드
ollama pull bge-m3
```

**문제**: 임베딩 차원 불일치
```bash
# 캐시 삭제 후 재시작
python find_cache.py  # y 선택하여 캐시 삭제
python interactive_test.py
```

**문제**: 빈 텍스트 오류
- BGE-M3 시스템은 빈 텍스트를 자동으로 `[EMPTY_TEXT]` 플레이스홀더로 처리
- 오류 발생 시 제로 벡터(1024차원)로 안전하게 처리

### Elasticsearch 관련

**문제**: 인덱스 없음 또는 벡터 필드 없음
```bash
# 인덱스 재구축 (주의: 기존 데이터 삭제됨)
python rebuild_bge_index.py
```

**문제**: 검색 결과 없음
```bash
# 인덱스 상태 확인
curl -X GET "localhost:9200/iqs_verbatim/_count"
curl -X GET "localhost:9200/iqs_verbatim/_mapping"
```
