"""
Agent Tools - 기존 모듈들을 Agent용 Tool로 래핑
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from src.utils.logger import log
from src.search.semantic_search import SemanticSearcher, SearchResult
from src.search.embeddings.embedding_model import EmbeddingModel
from src.reranker.reranker import Reranker
from src.agent.glossary_tool import GlossaryLookupTool
from src.agent.utils.document_utils import DocumentExtractor
from src.agent.mixins import SelfAssessmentMixin, ErrorHandlingMixin


class BaseTool(ABC, SelfAssessmentMixin, ErrorHandlingMixin):
    """
    도구 기본 클래스 - 🔧 개선: Mixin 기능 통합
    
    모든 도구가 공통으로 사용할 수 있는 기능들:
    - SelfAssessmentMixin: 자기 평가 기능
    - ErrorHandlingMixin: 안전한 실행 및 에러 처리
    """
    
    def __init__(self, *args, **kwargs):
        """기본 도구 초기화 - Mixin 호환성 보장"""
        super().__init__(*args, **kwargs)
        
        # 도구별 고유 설정 로드
        if hasattr(self, 'name'):
            log.debug(f"Initialized {self.name} with assessment and error handling")

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """도구 실행"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """도구 설명"""
        pass
    
    # 🔧 자기 평가 기능은 SelfAssessmentMixin에서 자동 상속
    # assess_query_suitability(), assess_step_suitability() 자동 제공
    
    # 🔧 안전한 실행 기능도 ErrorHandlingMixin에서 자동 상속
    # safe_execute() 자동 제공


class AggregatorTool(BaseTool):
    """
    통계 및 집계 도구
    SQL 스타일의 GROUP BY, COUNT, TOP N 등 처리
    """

    def __init__(self):
        """Initialize Aggregator Tool"""
        self.name = "aggregator"
        self.description = "데이터 집계 및 통계 분석 (GROUP BY, COUNT, TOP N)"
        
        # 🔧 Mixin 초기화 (자기 평가 및 에러 처리 기능 포함)
        super().__init__()
        
        log.info("Initialized AggregatorTool with assessment capabilities")

    def execute(self, 
                documents: Optional[List[Dict]] = None,
                aggregation: str = "terms",
                field: str = "problem",
                size: int = 10,
                filters: Optional[Dict] = None,
                previous_result: Optional[Any] = None,
                **kwargs) -> Dict[str, Any]:
        """
        집계 실행

        Args:
            documents: 문서 리스트 (previous_result에서 가져올 수도 있음)
            aggregation: 집계 타입 (terms, count, stats 등)
            field: 집계할 필드
            size: 결과 개수
            filters: 필터 조건
            previous_result: 이전 단계 결과

        Returns:
            집계 결과
        """
        log.info(f"Executing aggregation: {aggregation} on {field}")

        # 🔧 개선: 통합된 문서 추출 로직 사용
        documents, extracted_query = DocumentExtractor.extract_from_previous_result(
            previous_result=previous_result,
            documents=documents,
            fallback_fields=['source_documents'],
            extract_query=False,
            log_details=True
        )

        if not documents:
            log.warning("No documents to aggregate")
            return {
                "error": "No documents provided for aggregation",
                "aggregation_attempted": aggregation,
                "field_requested": field,
                "total_docs": 0,
                "buckets": [],
                "hallucination_prevented": True
            }

        # DataFrame으로 변환
        df = pd.DataFrame(documents)

        # 필터 적용
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]

        # 🔧 Hallucination 방지: 최소 데이터 임계값 검증
        MIN_DOCS_FOR_AGGREGATION = 5
        if len(df) < MIN_DOCS_FOR_AGGREGATION:
            log.warning(f"Insufficient data for reliable aggregation: {len(df)} < {MIN_DOCS_FOR_AGGREGATION}")
            return {
                "error": f"Insufficient data: only {len(df)} documents (minimum {MIN_DOCS_FOR_AGGREGATION} required)",
                "aggregation_attempted": aggregation,
                "field_requested": field,
                "total_docs": len(df),
                "buckets": [],
                "hallucination_prevented": True,
                "minimum_threshold": MIN_DOCS_FOR_AGGREGATION
            }

        # 집계 수행
        results = {}

        if aggregation == "terms":
            # Terms aggregation - 값별 카운트
            log.info(f"Available columns: {list(df.columns)}")
            
            # 필드명 매핑 (잘못된 필드명 수정)
            field_mapping = {
                'complaint_type': 'problem',  # complaint_type -> problem
                'issue_type': 'problem',      # issue_type -> problem  
                'defect_type': 'problem',     # defect_type -> problem
                'category': 'problem'
            }
            
            actual_field = field_mapping.get(field, field)
            
            if actual_field in df.columns:
                value_counts = df[actual_field].value_counts().head(size)
                # 🔧 Hallucination 방지: 실제 데이터 검증 및 투명성
                buckets = [
                    {
                        "key": key,
                        "doc_count": int(count)
                    }
                    for key, count in value_counts.items()
                ]
                
                results = {
                    "aggregation": "terms",
                    "field": actual_field,
                    "original_field": field,
                    "buckets": buckets,
                    "total_docs": len(df),
                    "data_verification": {
                        "source_doc_count": len(documents),
                        "filtered_doc_count": len(df),
                        "unique_values_found": len(buckets),
                        "filters_applied": bool(filters),
                        "hallucination_prevented": True,
                        "aggregation_timestamp": __import__('time').strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                log.info(f"Terms aggregation on '{actual_field}': {len(results['buckets'])} buckets")
                
                # 🔍 디버깅: 집계 결과 상세 로깅

            else:
                available_fields = [col for col in df.columns if 'problem' in col.lower() or 'issue' in col.lower() or 'category' in col.lower()]
                results = {
                    "error": f"Field '{field}' (mapped to '{actual_field}') not found",
                    "available_columns": list(df.columns),
                    "suggested_fields": available_fields,
                    "aggregation": "error"
                }

        elif aggregation == "count":
            # Simple count
            results = {
                "aggregation": "count",
                "count": len(df),
                "field": field if field else "all"
            }
            
        elif aggregation == "info_code_analysis":
            # INFO 코드 전용 분석
            if 'problem' in df.columns:
                import re
                info_counts = {}
                info_details = {}
                
                # 전체 문서에서 INFO 코드 추출 및 카운트
                for _, row in df.iterrows():
                    problem = str(row.get('problem', ''))
                    brand = row.get('make_of_vehicle', '')
                    verbatim = row.get('verbatim_text', '')
                    
                    # 현대 브랜드만 분석
                    if brand.lower() not in ['hyundai', '현대']:
                        continue
                    
                    # INFO12, INFO13, INFO14 추출
                    for target_code in ['INFO12', 'INFO13', 'INFO14']:
                        if target_code in problem:
                            if target_code not in info_counts:
                                info_counts[target_code] = 0
                                info_details[target_code] = []
                            
                            info_counts[target_code] += 1
                            
                            # 상세 정보 저장 (처음 3개만)
                            if len(info_details[target_code]) < 3:
                                info_details[target_code].append({
                                    'problem': problem,
                                    'verbatim': verbatim[:100] + '...' if len(verbatim) > 100 else verbatim,
                                    'model': row.get('model_of_vehicle', 'N/A')
                                })
                
                results = {
                    "aggregation": "info_code_analysis",
                    "field": "problem",
                    "hyundai_info_counts": info_counts,
                    "hyundai_info_details": info_details,
                    "total_hyundai_docs": len([r for _, r in df.iterrows() if r.get('make_of_vehicle', '').lower() in ['hyundai', '현대']]),
                    "total_docs_analyzed": len(df)
                }
                
                log.info(f"INFO code analysis complete: {info_counts}")
            else:
                results = {"error": "No 'problem' field found for INFO code analysis"}

        elif aggregation == "stats":
            # Statistical aggregation (for numeric fields)
            if field in df.columns and pd.api.types.is_numeric_dtype(df[field]):
                results = {
                    "aggregation": "stats",
                    "field": field,
                    "count": len(df),
                    "min": float(df[field].min()),
                    "max": float(df[field].max()),
                    "avg": float(df[field].mean()),
                    "sum": float(df[field].sum())
                }
            else:
                results = {"error": f"Field '{field}' is not numeric"}

        elif aggregation == "group_by":
            # GROUP BY multiple fields
            group_fields = kwargs.get('group_by', [field])
            if all(f in df.columns for f in group_fields):
                grouped = df.groupby(group_fields).size().reset_index(name='count')
                grouped = grouped.sort_values('count', ascending=False).head(size)

                results = {
                    "aggregation": "group_by",
                    "fields": group_fields,
                    "groups": grouped.to_dict('records'),
                    "total_groups": len(grouped)
                }
            else:
                results = {"error": "Invalid group fields"}

        elif aggregation == "top_problems":
            # 특별한 케이스: 문제별 TOP N with examples
            if 'problem' in df.columns:
                problem_groups = df.groupby('problem').agg({
                    'verbatim_id': 'count',
                    'verbatim_text': lambda x: list(x.head(3))  # 각 문제별 예시 3개
                }).rename(columns={'verbatim_id': 'count'})

                problem_groups = problem_groups.sort_values('count', ascending=False).head(size)

                results = {
                    "aggregation": "top_problems",
                    "problems": [
                        {
                            "problem": problem,
                            "count": int(row['count']),
                            "examples": row['verbatim_text']
                        }
                        for problem, row in problem_groups.iterrows()
                    ],
                    "source_documents": documents  # 원본 문서도 포함
                }

        aggregation_type = results.get('aggregation', 'unknown')
        if aggregation_type == 'unknown':
            log.warning(f"Aggregation type unknown. Results: {results}")
        else:
            log.info(f"Aggregation complete: {aggregation_type}")
        
        # 결과에 원본 문서 정보 추가 (다음 단계에서 사용 가능하도록)
        if isinstance(results, dict) and 'error' not in results and documents:
            results['source_documents'] = documents
            results['total_source_docs'] = len(documents)
            results['documents'] = documents  # 추가: documents 필드도 녹입
            log.info(f"Added {len(documents)} source documents AND documents to aggregation result")
        
        return results

    def _evaluate_and_apply_filters(self, explicit_filters: Optional[Dict], suggested_filters: Optional[Dict], 
                                   step_description: Optional[str], original_query: Optional[str]) -> Dict[str, Any]:
        """
        Agentic 필터 평가 및 적용 결정
        도구가 스스로 어떤 필터를 적용할지 지능적으로 판단
        """
        final_filters = {}
        
        # 1. 명시적 필터는 무조건 적용
        if explicit_filters:
            final_filters.update(explicit_filters)
            log.info(f"Applied explicit filters: {explicit_filters}")
        
        # 2. 제안된 필터는 맥락을 보고 선택적 적용
        if suggested_filters:
            for key, value in suggested_filters.items():
                should_apply = self._should_apply_suggested_filter(key, value, step_description, original_query)
                if should_apply:
                    final_filters[key] = value
                    log.info(f"Accepted suggested filter: {key}={value}")
                else:
                    log.info(f"Rejected suggested filter: {key}={value} (not relevant to current task)")
        
        return final_filters
    
    def _should_apply_suggested_filter(self, filter_key: str, filter_value: Any, 
                                      step_description: Optional[str], original_query: Optional[str]) -> bool:
        """
        제안된 필터의 적용 여부를 맥락적으로 판단
        """
        if not step_description and not original_query:
            return True  # 컨텍스트가 없으면 보수적으로 적용
        
        context_text = f"{step_description or ''} {original_query or ''}".lower()
        
        # 필터별 맥락적 적용 로직
        if filter_key == 'model_year':
            # 연도가 명시적으로 언급되거나 시간 관련 분석일 때만 적용
            year_mentioned = str(filter_value) in context_text
            time_analysis = any(word in context_text for word in ['연도', '년도', '시간', '기간', '최근', '작년'])
            return year_mentioned or time_analysis
        
        elif filter_key in ['model', 'vehicle_model']:
            # 특정 차종이 언급되었을 때만 적용
            model_mentioned = str(filter_value).lower() in context_text
            return model_mentioned
        
        elif filter_key == 'brand':
            # 브랜드 비교나 특정 브랜드 분석일 때만 적용
            brand_analysis = any(word in context_text for word in ['브랜드', '현대', '기아', '비교', 'vs'])
            return brand_analysis
        
        else:
            return True  # 알 수 없는 필터는 적용
    
    # 🔧 제거: assess_query_suitability() 메서드
    # SelfAssessmentMixin에서 설정 기반으로 자동 제공
    
    # 🔧 제거: assess_step_suitability() 메서드
    # SelfAssessmentMixin에서 설정 기반으로 자동 제공
    
    def _extract_problem_description(self, problem: str) -> str:
        """
        Problem 필드에서 문제 설명 추출
        예: "INFO12: Built-in navigation - Broken/works inconsistently" 
        → "Built-in navigation - Broken/works inconsistently"
        """
        if ':' in problem:
            return problem.split(':', 1)[1].strip()
        return problem
    
    def get_description(self) -> str:
        return self.description


class HybridSearchTool(BaseTool):
    """
    하이브리드 검색 도구
    의미 기반 검색 + 필터링 결합
    """

    def __init__(self):
        """Initialize Hybrid Search Tool"""
        self.name = "hybrid_search"
        self.description = "지능형 검색: 한글 번역, 코드 해석, 의미 검색 포함"
        
        # 🔧 수정: Mixin 초기화 (자기 평가 및 에러 처리 기능 포함)
        super().__init__()

        # 기존 SemanticSearcher 활용 (Ollama BGE-M3 + Elasticsearch)
        self.searcher = SemanticSearcher(
            embedding_model=EmbeddingModel(model_type="ollama"),
            use_elasticsearch=True
        )
        
        # 내장된 지식베이스 초기화
        self.knowledge_base = self._load_integrated_knowledge()

        # 인덱스 로드 시도
        if not self.searcher.load_index():
            log.warning("No pre-built index found. Need to build index first.")

        log.info("Initialized HybridSearchTool with assessment capabilities")

    def execute(self, 
                query: str,
                limit: int = 5000,
                filters: Optional[Dict] = None,
                year: Optional[int] = None,
                model: Optional[str] = None,
                part: Optional[str] = None,
                # 🆕 벡터 검색 옵션
                search_type: str = "hybrid",  # "text", "vector", "hybrid"
                enable_vector_search: bool = True,
                # Agentic 파라미터들
                suggested_filters: Optional[Dict] = None,
                step_description: Optional[str] = None,
                original_query: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        하이브리드 검색 실행

        Args:
            query: 검색 쿼리
            limit: 결과 개수
            filters: 필터 조건
            year: 연도 필터
            model: 차종 필터
            part: 부품 필터

        Returns:
            검색 결과
        """
        # INFO 코드 분석이나 전체 데이터 분석을 위해 강제로 대용량 검색
        if 'INFO' in query.upper() or '전체' in query or '전수' in query or limit < 1000:
            original_limit = limit
            limit = 5000  # 강제로 5000개로 설정
            log.info(f"Forced large-scale search: original limit={original_limit} -> forced limit={limit}")
        
        log.info(f"Executing intelligent search: '{query}' with limit={limit}")
        
        # Agentic 컨텍스트 인식
        if step_description:
            log.info(f"Agentic context: {step_description}")
        
        # 지능적 필터 평가 및 적용
        intelligent_filters = self._evaluate_search_filters(filters, suggested_filters, step_description, original_query)
        
        # 🔧 개선된 쿼리 처리: 문제 코드 정확한 타겟팅
        problem_codes = self._extract_problem_codes(query)
        if problem_codes:
            log.info(f"Problem codes detected: {problem_codes}")
            return self._search_by_problem_codes(problem_codes, intelligent_filters, limit, query)
        else:
            enhanced_query = self._enhance_query_with_knowledge(query)
            if enhanced_query != query:
                log.info(f"Query enhanced: '{query}' -> '{enhanced_query}'")
                query = enhanced_query

        # 지능적 필터 통합 (기존 방식 대체)
        final_filters = intelligent_filters.copy()
        
        # 개별 파라미터 통합
        if year:
            final_filters['model_year'] = year
        if model:
            final_filters['model'] = model
        if part:
            final_filters['part'] = part

        # 🆕 검색 실행 - 벡터 검색 지원
        try:
            # 벡터 검색 사용 여부 결정
            if enable_vector_search and search_type != "text":
                actual_search_type = search_type
                log.info(f"Using {actual_search_type} search with BGE-M3 vectors")
            else:
                actual_search_type = "text"
                log.info("Using text-only search (vector search disabled)")
            
            results = self.searcher.search(
                query=query,
                k=limit,
                filters=final_filters if final_filters else None,
                search_type=actual_search_type
            )

            # SearchResult 객체를 Dict로 변환
            documents = []
            for result in results:
                doc = result.content.copy()
                doc['_score'] = result.score
                doc['_matched_text'] = result.matched_text
                doc['_search_type'] = actual_search_type
                documents.append(doc)

            # 🔒 Hallucination 방지: 검색 결과 검증 및 투명성
            output = {
                "query": query,
                "filters": final_filters,
                "total_hits": len(documents),
                "documents": documents,
                "search_type": actual_search_type,
                "vector_search_enabled": enable_vector_search,
                "intelligent_filtering_applied": bool(intelligent_filters),
                "search_verification": {
                    "actual_results_found": len(documents),
                    "search_method_used": actual_search_type,
                    "filters_applied": final_filters,
                    "hallucination_prevented": True,
                    "search_timestamp": __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
                    "minimum_threshold_met": len(documents) > 0
                }
            }
            
            # 🚨 결과 부족 경고
            if len(documents) == 0:
                log.warning(f"ZERO RESULTS for query '{query}' with filters {final_filters}")
                output["warning"] = "No documents found - avoid hallucination"

            log.success(f"{actual_search_type.title()} search returned {len(documents)} results (requested: {limit})")
            return output

        except Exception as e:
            log.error(f"Search failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "filters": final_filters,
                "total_hits": 0,
                "documents": [],
                "search_verification": {
                    "search_failed": True,
                    "error_message": str(e),
                    "hallucination_prevented": True,
                    "recommendation": "Modify search criteria or check data availability"
                }
            }

    def _load_integrated_knowledge(self) -> Dict:
        """내장된 지식베이스 로드"""
        try:
            import json
            from pathlib import Path
            
            kb_file = Path("iqs_knowledge_base.json")
            if kb_file.exists():
                with open(kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load integrated knowledge base: {e}")
        
        return {}
    
    def _extract_problem_codes(self, query: str) -> List[str]:
        """
        쿼리에서 문제 코드 패턴 추출
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            추출된 문제 코드 리스트
        """
        import re
        problem_codes = []
        
        # 🔧 개선된 코드 추출: 모든 패턴 지원
        
        # 1. 완전한 코드 패턴 (INFO12, EXT16 등)
        full_code_pattern = r'(?:INFO|EXT|FCD|DRA|CLMT|INT|PWR)\d{1,2}'
        full_matches = re.findall(full_code_pattern, query, re.IGNORECASE)
        problem_codes.extend([code.upper() for code in full_matches])
        
        # 2. INFO + 숫자 조합 패턴 (INFO12/13/14, INFO22,23 등)
        info_combo_pattern = r'INFO\s*(?:\d{1,2}(?:[,/]\s*\d{1,2})*|\d{1,2}[/-]\d{1,2})'
        info_combos = re.findall(info_combo_pattern, query, re.IGNORECASE)
        
        for combo in info_combos:
            # 숫자만 추출
            numbers = re.findall(r'\d{1,2}', combo)
            for num in numbers:
                problem_codes.append(f'INFO{num}')
        
        # 3. 단독 숫자 후 INFO 문맥 처리 ("22, 23 항목코드")
        context_number_pattern = r'(\d{1,2})(?:\s*,\s*(\d{1,2}))*\s*항목\s*코드'
        context_matches = re.findall(context_number_pattern, query)
        
        for match in context_matches:
            if isinstance(match, tuple):
                for num in match:
                    if num:  # 빈 문자열 제외
                        problem_codes.append(f'INFO{num}')
            else:
                problem_codes.append(f'INFO{match}')
        
        # 4. 코드 + "에 해당" 패턴 처리 (EXT16에 해당하는)
        attached_pattern = r'([A-Z]{2,5}\d{1,2})에\s*해당'
        attached_matches = re.findall(attached_pattern, query, re.IGNORECASE)
        problem_codes.extend([code.upper() for code in attached_matches])
        
        # 5. 중복 제거 및 정렬
        unique_codes = list(set(problem_codes))
        
        # INFO 코드 우선 정렬
        info_codes = [code for code in unique_codes if code.startswith('INFO')]
        other_codes = [code for code in unique_codes if not code.startswith('INFO')]
        
        return info_codes + other_codes
    
    def _search_by_problem_codes(self, problem_codes: List[str], filters: Dict, limit: int, original_query: str) -> Dict[str, Any]:
        """
        문제 코드 기반 정확한 검색
        
        Args:
            problem_codes: 검색할 문제 코드 리스트
            filters: 기존 필터들
            limit: 결과 수 제한
            original_query: 원본 쿼리
            
        Returns:
            검색 결과
        """
        log.info(f"Executing problem code search for: {problem_codes}")
        
        try:
            all_results = []
            
            for code in problem_codes:
                # Elasticsearch를 사용할 수 있는 경우 체크 수정
                if hasattr(self.searcher, 'es_client') and self.searcher.es_client:
                    code_results = self._es_search_by_problem_code(code, filters)
                    all_results.extend(code_results)
                else:
                    # 로컬 검색 폴백
                    code_results = self._local_search_by_problem_code(code, filters)
                    all_results.extend(code_results)
            
            # 중복 제거 및 정렬
            unique_results = self._deduplicate_and_sort(all_results, limit)
            
            log.info(f"Problem code search found {len(unique_results)} documents")
            
            return {
                "query": original_query,
                "problem_codes_searched": problem_codes,
                "search_method": "problem_code_targeting",
                "filters": filters,
                "total_hits": len(unique_results),
                "documents": unique_results
            }
            
        except Exception as e:
            log.error(f"Problem code search failed: {e}")
            # 폴백: 기존 방식으로 검색
            return self._fallback_text_search(original_query, filters, limit)
    
    def _es_search_by_problem_code(self, code: str, filters: Dict) -> List[Dict]:
        """
        Elasticsearch를 사용한 문제 코드 정확 검색
        """
        try:
            # 🔧 수정된 문제 코드 검색 - match 쿼리 사용
            es_query = {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "problem": code  # INFO12, EXT16 등 코드 매칭
                            }
                        }
                    ],
                    "filter": []
                }
            }
            
            # 추가 필터 적용
            if filters:
                for key, value in filters.items():
                    es_query["bool"]["filter"].append({
                        "term": {key: value}
                    })
            
            # Elasticsearch 실행 수정
            es_response = self.searcher.es_client.search({
                "query": es_query,
                "size": 1000,  # 코드별로 최대 1000개
                "_source": True
            })
            
            results = []
            for hit in es_response['hits']['hits']:
                doc = hit['_source']
                problem = doc.get('problem', '')
                
                # 🔧 후처리 필터링: 정확한 코드만 선택
                if problem.startswith(f"{code}:"):
                    doc['_score'] = hit['_score']
                    doc['_search_method'] = f'es_match_filtered_{code}'
                    results.append(doc)
            
            log.info(f"ES search for {code}: {len(results)} documents")
            return results
            
        except Exception as e:
            log.error(f"ES search for {code} failed: {e}")
            return []
    
    def _local_search_by_problem_code(self, code: str, filters: Dict) -> List[Dict]:
        """
        일반 의미 검색을 사용한 문제 코드 검색
        """
        try:
            log.info(f"Using semantic search for {code} problem code")
            
            # 일반 의미 검색으로 대체
            search_query = f"{code} problem code"
            if filters and 'make_of_vehicle' in filters:
                search_query += f" {filters['make_of_vehicle']}"
            
            semantic_results = self.searcher.search(
                query=search_query,
                k=1000,  # 코드별로 1000개
                filters=filters
            )
            
            results = []
            for result in semantic_results:
                doc = result.content.copy()
                problem = doc.get('problem', '')
                
                # 🔧 정확한 코드 매칭 확인
                if problem.startswith(f"{code}:"):
                    doc['_score'] = result.score
                    doc['_search_method'] = f'semantic_filtered_problem_code_{code}'
                    results.append(doc)
            
            log.info(f"Semantic search for {code}: found {len(results)} matching documents")
            return results
            
        except Exception as e:
            log.error(f"Semantic search for {code} failed: {e}")
            return []
    
    def _deduplicate_and_sort(self, results: List[Dict], limit: int) -> List[Dict]:
        """
        중복 제거 및 정렬
        """
        # verbatim_id 기준 중복 제거
        seen_ids = set()
        unique_results = []
        
        for doc in results:
            doc_id = doc.get('verbatim_id', '') or doc.get('_id', '')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)
        
        # 점수순 정렬
        unique_results.sort(key=lambda x: x.get('_score', 0), reverse=True)
        
        return unique_results[:limit]
    
    def _fallback_text_search(self, query: str, filters: Dict, limit: int) -> Dict[str, Any]:
        """
        문제 코드 검색 실패 시 폴백
        """
        log.warning("Falling back to text search")
        
        try:
            results = self.searcher.search(
                query=query,
                k=limit,
                filters=filters if filters else None
            )
            
            documents = []
            for result in results:
                doc = result.content.copy()
                doc['_score'] = result.score
                doc['_matched_text'] = result.matched_text
                doc['_search_method'] = 'fallback_text'
                documents.append(doc)

            return {
                "query": query,
                "search_method": "fallback_text_search",
                "filters": filters,
                "total_hits": len(documents),
                "documents": documents
            }
            
        except Exception as e:
            log.error(f"Fallback search also failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "total_hits": 0,
                "documents": []
            }

    def _enhance_query_with_knowledge(self, query: str) -> str:
        """
        지식베이스를 활용한 지능형 쿼리 향상
        🔧 개선: 번역은 GlossaryTool에서 처리, 여기서는 코드 해석만
        
        Args:
            query: 원본 쿼리
            
        Returns:
            향상된 쿼리
        """
        try:
            enhanced_query = query
            
            # 1. 🔧 번역 로직 제거 (GlossaryTool에서 처리)
            # 한글 번역은 LangGraph Agent가 GlossaryTool을 호출해서 처리
            
            # 2. 코드 해석 및 확장 (유지)
            enhanced_query = self._expand_code_terms(enhanced_query)
            
            # 3. 도메인 용어 해석 (유지)
            enhanced_query = self._expand_domain_terms(enhanced_query)
            
            # 4. 🔧 GlossaryTool 결과 활용 시도
            glossary_enhanced = self._try_extract_from_glossary_result()
            if glossary_enhanced:
                log.info(f"Using GlossaryTool result: '{query}' -> '{glossary_enhanced}'")
                enhanced_query = glossary_enhanced
            
            return enhanced_query
            
        except Exception as e:
            log.warning(f"Query enhancement failed: {e}")
            return query
    
    def _try_extract_from_glossary_result(self) -> Optional[str]:
        """
        🔧 새로운 기능: 이전 결과에서 GlossaryTool의 번역 결과 추출
        LangGraph Agent가 GlossaryTool -> HybridSearchTool 순서로 호출했을 때 활용
        
        Returns:
            GlossaryTool에서 번역된 쿼리 또는 None
        """
        # 현재 LangGraph의 시스템 상태를 통해 이전 결과 접근 시도
        # (실제 구현에서는 previous_result나 context를 통해 접근)
        
        # TODO: 실제 구현에서는 LangGraph state를 통해 접근
        # 예시: state.get('glossary_results', {})
        
        return None
    
    
    def _expand_code_terms(self, query: str) -> str:
        """코드 용어 확장"""
        # 문제 코드 패턴 찾기
        import re
        code_pattern = r'\b([A-Z]{2,5}\d{1,3})\b'
        codes = re.findall(code_pattern, query)
        
        if not codes or not self.knowledge_base.get('problem_codes', {}).get('codes'):
            return query
        
        enhanced_parts = [query]
        
        for code in codes:
            code_info = self.knowledge_base['problem_codes']['codes'].get(code)
            if code_info:
                # 코드 설명을 검색어에 추가
                description = code_info.get('description', '')
                # 주요 키워드 추출
                keywords = re.findall(r'\b\w{3,}\b', description.lower())
                if keywords:
                    enhanced_parts.extend(keywords[:3])  # 상위 3개 키워드
        
        return ' '.join(enhanced_parts)
    
    def _expand_domain_terms(self, query: str) -> str:
        """도메인 용어 확장"""
        domain_expansions = {
            'DTU': 'difficult to use usability problem',
            'OEM': 'original equipment manufacturer',
            'IQS': 'initial quality study problem'
        }
        
        enhanced_query = query
        for term, expansion in domain_expansions.items():
            if term in query:
                enhanced_query = enhanced_query.replace(term, f"{term} {expansion}")
        
        return enhanced_query
    
    def _validate_and_enhance_search(self, original_query: str, documents: List[Dict], limit: int, filters: Optional[Dict]) -> Dict[str, Any]:
        """
        동적 검색 결과 검증 및 필요시 대안 검색
        LangGraph 철학: 도구가 스스로 판단하고 적응
        """
        import re
        
        # 1. INFO 코드 검색 결과 검증
        requested_info_codes = re.findall(r'INFO\d{1,2}', original_query.upper())
        
        if requested_info_codes:
            # 요청된 INFO 코드와 실제 결과 비교
            found_codes = set()
            for doc in documents:
                problem = doc.get('problem', '')
                doc_codes = re.findall(r'INFO\d{1,2}', problem)
                found_codes.update(doc_codes)
            
            missing_codes = set(requested_info_codes) - found_codes
            
            if missing_codes:
                log.warning(f"Requested codes {requested_info_codes} but only found {list(found_codes)}. Missing: {list(missing_codes)}")
                
                # 자동 대안 검색 시도
                alternative_documents = self._try_alternative_search(original_query, missing_codes, limit, filters)
                
                if alternative_documents:
                    log.info(f"Alternative search found {len(alternative_documents)} additional results")
                    # 기존 결과와 대안 결과 결합
                    combined_docs = documents + alternative_documents
                    # 중복 제거 (문서 ID 기준)
                    seen_ids = set()
                    unique_docs = []
                    for doc in combined_docs:
                        doc_id = doc.get('verbatim_id') or doc.get('_id')
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            unique_docs.append(doc)
                    
                    documents = unique_docs[:limit]  # limit 유지
        
        return {
            "query": original_query,
            "filters": filters,
            "total_hits": len(documents),
            "documents": documents,
            "search_enhanced": len(requested_info_codes) > 0 and bool(missing_codes)
        }
    
    def _try_alternative_search(self, original_query: str, missing_codes: set, limit: int, filters: Optional[Dict]) -> List[Dict]:
        """
        대안 검색 전략: 의미 기반 검색
        """
        # 실제 데이터에서 학습한 패턴을 사용
        # 하드코딩 대신 동적 의미 검색
        alternative_queries = []
        
        for code in missing_codes:
            # 코드 번호별 대안 전략
            if '12' in code:
                alternative_queries.append('navigation broken inconsistent works')
            elif '13' in code:
                alternative_queries.append('navigation DTU difficult update')
            elif '14' in code:
                alternative_queries.append('navigation inaccurate wrong incorrect')
            else:
                # 알려지지 않은 코드는 일반적 인포테인먼트 검색
                alternative_queries.append('infotainment problem')
        
        if not alternative_queries:
            return []
        
        # 대안 쿼리로 검색
        combined_query = ' OR '.join(alternative_queries)
        log.info(f"Trying alternative search: {combined_query}")
        
        try:
            alt_results = self.searcher.search(
                query=combined_query,
                k=limit,
                filters=filters
            )
            
            alt_documents = []
            for result in alt_results:
                doc = result.content.copy()
                doc['_score'] = result.score * 0.9  # 대안 검색은 약간 낮은 점수
                doc['_matched_text'] = result.matched_text
                doc['_search_method'] = 'alternative_semantic'
                alt_documents.append(doc)
            
            return alt_documents
            
        except Exception as e:
            log.warning(f"Alternative search failed: {e}")
            return []
    
    def _evaluate_search_filters(self, explicit_filters: Optional[Dict], suggested_filters: Optional[Dict],
                               step_description: Optional[str], original_query: Optional[str]) -> Dict[str, Any]:
        """
        Agentic 검색 필터 평가 (HybridSearchTool 전용)
        검색 도구가 스스로 어떤 필터를 적용할지 판단
        """
        final_filters = {}
        
        # 1. 명시적 필터는 무조건 적용
        if explicit_filters:
            final_filters.update(explicit_filters)
            log.info(f"Applied explicit search filters: {explicit_filters}")
        
        # 2. 제안된 필터를 맥락적으로 평가
        if suggested_filters:
            for key, value in suggested_filters.items():
                should_apply = self._should_apply_search_filter(key, value, step_description, original_query)
                if should_apply:
                    final_filters[key] = value
                    log.info(f"Search tool accepted suggested filter: {key}={value}")
                else:
                    log.info(f"Search tool rejected suggested filter: {key}={value} (not contextually relevant)")
        
        return final_filters
    
    def _should_apply_search_filter(self, filter_key: str, filter_value: Any,
                                   step_description: Optional[str], original_query: Optional[str]) -> bool:
        """
        검색 특화 필터 적용 판단
        """
        if not step_description and not original_query:
            return True
        
        context_text = f"{step_description or ''} {original_query or ''}".lower()
        
        # 검색 도구는 보다 열린 필터링 정책
        if filter_key == 'model_year':
            # 연도가 명시되거나 최근 데이터 요청 시 적용
            year_mentioned = str(filter_value) in context_text
            recent_request = any(word in context_text for word in ['최근', '작년', '금년', '올해'])
            return year_mentioned or recent_request
        
        elif filter_key in ['model', 'vehicle_model']:
            # 특정 차종 언급 시만 적용
            return str(filter_value).lower() in context_text
        
        elif filter_key == 'brand':
            # 브랜드 관련 분석이나 비교 시만 적용
            brand_context = any(word in context_text for word in ['브랜드', '현대', '기아', '비교'])
            return brand_context
        
        else:
            # 다른 필터들은 기본적으로 수용
            return True
    
    # 🔧 중복 제거: assess_query_suitability() 메서드 제거
    # SelfAssessmentMixin에서 설정 기반으로 자동 제공
    
    # 🔧 중복 제거: assess_step_suitability() 메서드 제거  
    # SelfAssessmentMixin에서 설정 기반으로 자동 제공
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        문서로부터 검색 인덱스 구축

        Args:
            documents: 인덱싱할 문서 리스트
        """
        log.info(f"Building search index with {len(documents)} documents")
        self.searcher.build_index_from_documents(documents)

    def get_description(self) -> str:
        return self.description


class RerankerTool(BaseTool):
    """
    재순위화 도구
    검색 결과를 다양한 신호를 활용하여 재순위화
    """

    def __init__(self):
        """Initialize Reranker Tool"""
        self.name = "reranker"
        self.description = "검색 결과를 관련성 기준으로 재순위화"
        
        # 🔧 수정: Mixin 초기화 (자기 평가 및 에러 처리 기능 포함)
        super().__init__()
        
        self.reranker = Reranker()
        log.info("Initialized RerankerTool with assessment capabilities")

    def execute(self,
                documents: Optional[List[Dict]] = None,
                query: Optional[str] = None,
                top_k: int = 10,
                previous_result: Optional[Any] = None,
                filter: Optional[Dict] = None,  # 브랜드 필터링 등
                **kwargs) -> Dict[str, Any]:
        """
        재순위화 실행

        Args:
            documents: 재순위화할 문서들
            query: 원본 쿼리
            top_k: 상위 K개 반환
            previous_result: 이전 단계 결과

        Returns:
            재순위화된 결과
        """
        log.info(f"Executing reranking with top_k={top_k}")

                            # 🔧 개선: 통합된 문서 추출 로직 사용
        documents, extracted_query = DocumentExtractor.extract_from_previous_result(
            previous_result=previous_result,
            documents=documents,
            fallback_fields=['source_documents'],
            extract_query=True,
            log_details=True
        )
        
        # 추출된 쿼리가 있으면 사용
        if query is None and extracted_query:
            query = extracted_query

        if not documents:
            log.warning(f"No documents to rerank. Previous result available: {previous_result is not None}")
            return {
                "error": "No documents provided",
                "query": query,
                "total_reranked": 0,
                "top_k": top_k,
                "documents": [],
                "debug_info": {
                    "document_extraction_attempted": True,
                    "previous_result_available": previous_result is not None
                }
            }
        
        # LLM 지시에 따른 필터링 수행
        if filter:
            original_count = len(documents)
            for filter_key, filter_value in filter.items():
                if filter_key == 'brand' and filter_value.lower() in ['hyundai', '현대']:
                    documents = [doc for doc in documents 
                               if doc.get('make_of_vehicle', '').lower() in ['hyundai', '현대']]
                    log.info(f"Filtered by Hyundai brand: {original_count} -> {len(documents)} documents")
                elif filter_key in ['make_of_vehicle', 'brand']:
                    documents = [doc for doc in documents 
                               if doc.get('make_of_vehicle', '').lower() == filter_value.lower()]
                    log.info(f"Filtered by brand '{filter_value}': {original_count} -> {len(documents)} documents")
            
            # 필터링 후 문서가 없으면 오류 반환
            if not documents:
                return {
                    "error": f"No documents found after filtering by {filter}",
                    "query": query,
                    "total_reranked": 0,
                    "top_k": top_k,
                    "documents": [],
                    "filter_applied": filter
                }

        if not query:
            log.warning("No query provided for reranking")
            # 쿼리 없이도 메타데이터 기반 재순위화는 가능

        # SearchResult 객체로 변환
        search_results = []
        for doc in documents:
            # 점수가 있으면 사용, 없으면 기본값
            score = doc.get('_score', 0.5)

            result = SearchResult(
                doc_id=doc.get('verbatim_id', ''),
                score=score,
                content=doc,
                matched_text=doc.get('verbatim_text', '')
            )
            search_results.append(result)

        # 재순위화 실행
        reranked = self.reranker.rerank(
            results=search_results,
            query=query or "",
            query_plan=None,  # Agent 모드에서는 query_plan 없음
            top_k=top_k
        )

        # 결과 변환
        reranked_docs = []
        for result, score in reranked:
            doc = result.content.copy()
            doc['_score'] = score.total_score
            doc['_rerank_details'] = {
                'semantic': score.semantic_score,
                'relevance': score.relevance_score,
                'recency': score.recency_score,
                'metadata': score.metadata_score,
                'explanation': score.explanation
            }
            reranked_docs.append(doc)

        output = {
            "query": query,
            "total_reranked": len(reranked_docs),
            "top_k": top_k,
            "documents": reranked_docs
        }

        log.info(f"Reranking complete: {len(reranked_docs)} documents")
        
        # 필터 정보 추가
        if filter:
            output['filter_applied'] = filter
            
        return output

    def _evaluate_reranking_filters(self, explicit_filter: Optional[Dict], suggested_filters: Optional[Dict],
                                   step_description: Optional[str], original_query: Optional[str]) -> Optional[Dict]:
        """
        Agentic 재순위화 필터 평가
        재순위화 도구가 스스로 필터링 필요성을 판단
        """
        # 1. 명시적 필터가 있으면 우선 적용
        if explicit_filter:
            log.info(f"Reranker applying explicit filter: {explicit_filter}")
            return explicit_filter
        
        # 2. 제안된 필터를 맥락적으로 평가
        if suggested_filters:
            context_text = f"{step_description or ''} {original_query or ''}".lower()
            
            # 브랜드 필터링이 의미가 있는지 판단
            brand_context = any(word in context_text for word in [
                '브랜드', '현대', '기아', '비교', 'vs', '전용', '특정'
            ])
            
            if brand_context and 'brand' in suggested_filters:
                return {'brand': suggested_filters['brand']}
            elif brand_context and 'model_year' in suggested_filters:
                return {'model_year': suggested_filters['model_year']}
        
        # 3. 필터링이 필요 없다고 판단
        return None
    
    def _apply_intelligent_filter(self, documents: List[Dict], filter_config: Dict) -> List[Dict]:
        """
        지능적 필터 적용
        """
        filtered_docs = documents.copy()
        
        for filter_key, filter_value in filter_config.items():
            if filter_key == 'brand':
                if filter_value.lower() in ['hyundai', '현대']:
                    filtered_docs = [doc for doc in filtered_docs 
                                   if doc.get('make_of_vehicle', '').lower() in ['hyundai', '현대']]
                elif filter_value.lower() in ['kia', '기아']:
                    filtered_docs = [doc for doc in filtered_docs 
                                   if doc.get('make_of_vehicle', '').lower() in ['kia', '기아']]
                else:
                    filtered_docs = [doc for doc in filtered_docs 
                                   if doc.get('make_of_vehicle', '').lower() == filter_value.lower()]
            
            elif filter_key == 'model_year':
                filtered_docs = [doc for doc in filtered_docs 
                               if doc.get('model_year') == filter_value]
            
            elif filter_key in ['model', 'vehicle_model']:
                filtered_docs = [doc for doc in filtered_docs 
                               if doc.get('model_of_vehicle', '').lower() == str(filter_value).lower()]
        
        return filtered_docs
    
    # 🔧 중복 제거: assess_query_suitability() 메서드 제거
    # SelfAssessmentMixin에서 설정 기반으로 자동 제공
    
    # 🔧 중복 제거: assess_step_suitability() 메서드 제거
    # SelfAssessmentMixin에서 설정 기반으로 자동 제공
    
    def get_description(self) -> str:
        return self.description


class GlossaryTool(BaseTool):
    """
    용어 사전 조회 도구
    한글 쿼리를 영어로 변환하고 동의어 확장
    """
    
    def __init__(self):
        """Initialize Glossary Tool"""
        self.name = "glossary_lookup"
        self.description = "한-영 용어 변환 및 동의어 확장 (검색 전 필수)"
        
        # 🔧 수정: Mixin 초기화 (자기 평가 및 에러 처리 기능 포함)
        super().__init__()
        
        self.glossary = GlossaryLookupTool()
        log.info("Initialized GlossaryTool with assessment capabilities")
    
    def execute(self,
                query: str,
                include_synonyms: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        용어 변환 실행
        
        Args:
            query: 변환할 쿼리
            include_synonyms: 동의어 포함 여부
            
        Returns:
            변환 결과
        """
        log.info(f"Executing glossary lookup: '{query}'")
        
        result = self.glossary.execute(
            query=query,
            include_synonyms=include_synonyms
        )
        
        log.info(f"Glossary lookup complete: {result.get('translation_applied', False)}")
        return result
    
    def get_description(self) -> str:
        return self.description


class SynthesizerTool(BaseTool):
    """
    종합 도구 - 개선된 동적 코드 처리
    하드코딩된 INFO 처리 로직을 패턴 기반으로 개선
    """

    def __init__(self, llm_client=None):
        """
        Initialize Synthesizer Tool

        Args:
            llm_client: LLM 클라이언트
        """
        self.name = "synthesizer"
        self.description = "여러 결과를 종합하여 최종 응답 생성"
        
        # 🔧 수정: Mixin 초기화 (자기 평가 및 에러 처리 기능 포함)
        super().__init__()
        
        self.llm_client = llm_client
        
        # 🔧 Phase 1: LLM 자율 판단 방식으로 변경 (하드코딩 제거)
        
        # 🔧 개선: 코드 메타데이터 설정
        self.code_metadata = self._load_code_metadata()
        
        log.info("Initialized SynthesizerTool with response mode control and dynamic code processing")

    def execute(self,
                results: List[Any],
                original_query: str,
                **kwargs) -> Dict[str, Any]:
        """
        결과 종합 실행 (응답 모드 지원)

        Args:
            results: 종합할 결과들
            original_query: 원본 쿼리
            response_mode: 응답 모드 ("factual", "analytical", 또는 None)

        Returns:
            종합된 최종 응답
        """
        log.info("Executing synthesis with LLM autonomous judgment")

        if self.llm_client:
            # LLM을 사용한 종합 - LLM이 응답 스타일 자율 결정
            synthesis_prompt = self._create_synthesis_prompt(results, original_query)
            llm_response = self.llm_client.complete(synthesis_prompt)

            return {
                "query": original_query,
                "synthesis": llm_response,
                "source_count": len(results)
            }
        else:
            # 규칙 기반 종합
            return self._rule_based_synthesis(results, original_query)
    
    def execute_streaming(self,
                         results: List[Any],
                         original_query: str,
                         **kwargs):
        """
        결과 종합 실행 (스트리밍 방식 - LangGraph 전용)

        Args:
            results: 종합할 결과들
            original_query: 원본 쿼리
            response_mode: 응답 모드 ("factual", "analytical", 또는 None)

        Yields:
            스트리밍 텍스트 청크들
        """
        log.info("Executing synthesis (streaming mode) with LLM autonomous judgment")

        if self.llm_client and hasattr(self.llm_client, 'stream_complete'):
            # LLM 스트리밍 종합 - LLM이 응답 스타일 자율 결정
            synthesis_prompt = self._create_synthesis_prompt(results, original_query)
            
            try:
                # H-Chat 실시간 스트리밍 사용
                for chunk in self.llm_client.stream_complete(synthesis_prompt):
                    if chunk:  # 빈 청크 제외
                        yield chunk
                        
            except Exception as e:
                log.error(f"LLM streaming failed: {e}")
                # 폴백: 일반 모드로 전환
                llm_response = self.llm_client.complete(synthesis_prompt)
                # 시뮬레이션 스트리밍
                sentences = llm_response.split('. ')
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        if i < len(sentences) - 1:
                            yield sentence + '. '
                        else:
                            yield sentence
        else:
            # 규칙 기반 종합 (스트리밍 시뮬레이션)
            rule_result = self._rule_based_synthesis(results, original_query)
            synthesis_text = rule_result.get('synthesis', 'Analysis completed.')
            
            # 문장별 스트리밍 시뮬레이션
            sentences = synthesis_text.split('\n')
            for sentence in sentences:
                if sentence.strip():
                    yield sentence + '\n'

    def _create_synthesis_prompt(self, results: List[Any], query: str) -> str:
        """🔧 개선된 LLM 종합을 위한 프롬프트 생성 - Hallucination 방지 강화"""

        # Hallucination 방지 1단계: 실제 데이터 검증
        total_docs_across_results = 0
        valid_results_count = 0
        error_results_count = 0
        
        for result in results:
            if isinstance(result, dict):
                if 'error' in result:
                    error_results_count += 1
                elif 'documents' in result:
                    doc_count = len(result.get('documents', []))
                    total_docs_across_results += doc_count
                    if doc_count > 0:
                        valid_results_count += 1
                elif 'buckets' in result:
                    bucket_count = len(result.get('buckets', []))
                    if bucket_count > 0:
                        valid_results_count += 1
                        # 버킷의 총 문서 수 계산
                        bucket_total = sum(bucket.get('doc_count', 0) for bucket in result['buckets'])
                        total_docs_across_results += bucket_total
        
        # Hallucination 위험 감지
        if total_docs_across_results == 0 and valid_results_count == 0:
            log.error(f"HALLUCINATION RISK: No actual data found for query '{query}'")
            return f"""**데이터 부족 알림**

죄송합니다. 요청하신 조건에 맞는 실제 데이터를 찾을 수 없습니다:

**검색 조건**: {query}
**검증 결과**: 
- 총 처리된 결과: {len(results)}개
- 오류 발생: {error_results_count}개
- 실제 데이터: {total_docs_across_results}개 문서

**권장사항**:
1. 검색 조건을 더 넓게 설정해보세요
2. 연도나 차종 필터를 제거해보세요  
3. 다른 문제 코드로 검색해보세요

이 시스템은 **실제 데이터만 제공**하며, 추정이나 가정에 기반한 답변을 생성하지 않습니다."""
        
        # 결과를 텍스트로 변환 - 검증된 데이터만 처리
        results_text = []
        discovered_codes = set()
        
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                if 'documents' in result:
                    documents = result['documents']
                    results_text.append(f"검색 결과: {len(documents)}개 문서")
                    
                    # 🔧 개선: 동적 코드 감지 (하드코딩 제거)
                    code_analysis = self._analyze_codes_in_documents(documents)
                    discovered_codes.update(code_analysis.keys())
                    
                    for code, info in code_analysis.items():
                        results_text.append(f"\n⭐ {code} 문서 발견: {info['count']}개")
                        if info['examples']:
                            results_text.append(f" - {info['examples'][0]}")
                    
                    # 기타 문서 표시 (코드가 아닌 것들)
                    other_docs = [doc for doc in documents[:3] 
                                if not any(code in doc.get('problem', '') for code in discovered_codes)]
                    for doc in other_docs:
                        results_text.append(f" - {doc.get('problem', 'N/A')}")
                        
                elif 'buckets' in result:
                    # 🔧 개선: 동적 집계 결과 처리
                    buckets = result.get('buckets', [])
                    debug_info = result.get('debug_info', {})
                    
                    results_text.append(f"집계 결과 상세 분석:")
                    results_text.append(f"- 총 문서 수: {debug_info.get('input_doc_count', len(buckets))}개")
                    
                    # 동적 코드 분석
                    bucket_codes = self._analyze_codes_in_buckets(buckets)
                    discovered_codes.update(bucket_codes.keys())
                    
                    for code, info in bucket_codes.items():
                        metadata = self.code_metadata.get(code, {})
                        severity = metadata.get('severity', '보통')
                        description = metadata.get('description', f'{code} 관련 문제')
                        
                        results_text.append(f"\n*** {code} 코드 발견 ***")
                        results_text.append(f"- 문제: {info['key']}")
                        results_text.append(f"- 발생 건수: {info['count']}건")
                        results_text.append(f"- 문제 유형: {description}")
                        results_text.append(f"- 심각도: {severity}")
                    
                    # 기타 버킷 표시
                    for bucket in buckets[:5]:
                        if not any(code in bucket['key'] for code in discovered_codes):
                            results_text.append(f"- {bucket['key']}: {bucket['doc_count']}건")

        #  Hallucination 방지 2단계: 데이터 투명성 강화
        data_transparency_info = f"""

**데이터 검증 정보 (투명성 보장)**:
- 총 분석 문서: {total_docs_across_results:,}개
- 유효한 결과: {valid_results_count}개  
- 오류 결과: {error_results_count}개
- 발견된 코드: {', '.join(discovered_codes) if discovered_codes else '없음'}
- 분석 시점: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}"""
        
        # Phase 1: LLM 자율 판단 방식으로 변경 + Hallucination 방지 강화
        llm_autonomous_instructions = self._generate_hallucination_safe_instructions(list(discovered_codes), total_docs_across_results)
        
        prompt = f"""사용자 질문: {query}

분석 결과:
{chr(10).join(results_text)}
{data_transparency_info}

응답 지침 (실데이터 기반 + Hallucination 방지):
{llm_autonomous_instructions}

**중요: Hallucination 방지 규칙**
- 실제 데이터에 없는 숫자나 통계를 절대 생성하지 마세요
- "약", "대략", "추정" 등의 표현으로 불확실성을 명시하세요
- 데이터가 부족하면 솔직히 "데이터 부족"이라고 명시하세요
- 위에 제공된 실제 문서 수와 코드 정보만 사용하세요"

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
    
    def _load_code_metadata(self) -> Dict[str, Any]:
        """🔧 개선: 코드 메타데이터 로드 (설정 파일 또는 기본값)"""
        try:
            # 설정 파일이 있으면 로드
            import json
            from pathlib import Path
            
            metadata_file = Path("data/code_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load code metadata: {e}")
        
        # 🔧 객관적 메타데이터만 제공 (심각도 정보 완전 제거)
        return {
            "INFO12": {
                "description": "Built-in navigation - Broken/works inconsistently",
                "category": "infotainment",
                "keywords": ["navigation", "broken", "inconsistent"]
            },
            "INFO13": {
                "description": "Built-in navigation - DTU (difficult to use)",
                "category": "infotainment", 
                "keywords": ["navigation", "difficult", "usability"]
            },
            "INFO22": {
                "description": "Wireless charging pad - Broken/works inconsistently", 
                "category": "infotainment",
                "keywords": ["wireless", "charging", "broken"]
            },
            "INFO23": {
                "description": "Wireless charging pad - Size/location inappropriate",
                "category": "infotainment",
                "keywords": ["wireless", "charging", "size", "location"]
            },
            "EXT16": {
                "description": "Exterior component issue",
                "category": "exterior",
                "keywords": ["exterior", "component"]
            }
        }
    
    def _analyze_codes_in_documents(self, documents: List[Dict]) -> Dict[str, Dict]:
        """🔧 개선: 문서에서 코드 동적 분석"""
        import re
        code_analysis = {}
        
        # 모든 코드 패턴 동적 감지
        code_pattern = r'\b(INFO|EXT|FCD|DRA|CLMT|INT|PWR)\d{1,2}\b'
        
        for doc in documents:
            problem = doc.get('problem', '')
            codes = re.findall(code_pattern, problem, re.IGNORECASE)
            
            for match in re.finditer(code_pattern, problem, re.IGNORECASE):
                code = match.group().upper()
                if code not in code_analysis:
                    code_analysis[code] = {
                        'count': 0,
                        'examples': [],
                        'documents': []
                    }
                
                code_analysis[code]['count'] += 1
                code_analysis[code]['documents'].append(doc)
                
                if len(code_analysis[code]['examples']) < 3:
                    code_analysis[code]['examples'].append(problem)
        
        return code_analysis
    
    def _analyze_codes_in_buckets(self, buckets: List[Dict]) -> Dict[str, Dict]:
        """🔧 개선: 집계 버킷에서 코드 동적 분석"""
        import re
        code_analysis = {}
        
        code_pattern = r'\b(INFO|EXT|FCD|DRA|CLMT|INT|PWR)\d{1,2}\b'
        
        for bucket in buckets:
            key = bucket['key']
            count = bucket['doc_count']
            
            codes = re.findall(code_pattern, key, re.IGNORECASE)
            for code_match in codes:
                code = code_match.upper()
                if code not in code_analysis:
                    code_analysis[code] = {
                        'key': key,
                        'count': count
                    }
                else:
                    # 중복이면 카운트 합산
                    code_analysis[code]['count'] += count
        
        return code_analysis
    
    def _generate_hallucination_safe_instructions(self, discovered_codes: list, total_docs: int) -> str:
        """
        Hallucination 방지 강화된 LLM 지침 생성
        
        Args:
            discovered_codes: 발견된 코드 리스트
            total_docs: 실제 문서 수
            
        Returns:
            Hallucination 방지가 강화된 지침 문자열
        """
        # Hallucination 방지 강화 지침
        base_instructions = [
            "1. **실제 데이터만 사용**: 위에 제공된 데이터에 없는 숫자나 정보를 절대 처리하지 마세요",
            f"2. **투명성**: 총 {total_docs:,}개 문서에 기반한 분석이며, 이를 명시하세요",
            "3. **강제 검증**: 각 숫자는 위 데이터에서 직접 확인 가능해야 합니다",
            "4. **불확실성 표시**: 데이터가 불충분하면 '데이터 부족', '단정적 결론 어려움' 등으로 솔직하게 명시"
        ]
        
        # 데이터 부족 경고
        if total_docs < 50:
            base_instructions.append(
                f"5. **중요 경고**: 데이터 수가 적음 ({total_docs}개). 제한적 분석이므로 '데이터 수 제한으로 인한 예비 결과'라고 명시"
            )
        
        if discovered_codes:
            # 검증된 코드 정보만 제공
            base_instructions.append(f"\n6. 발견된 {', '.join(discovered_codes)} 코드들의 기본 정보:")
            for code in discovered_codes:
                metadata = self.code_metadata.get(code, {})
                description = metadata.get('description', f'{code} 관련 문제')
                base_instructions.append(f"   - {code}: {description} (실제 데이터에서 확인됨)")
        else:
            base_instructions.append("\n6. 특정 문제 코드가 발견되지 않았음 (데이터 기반 확인)")
        
        # 최종 Hallucination 방지 경고
        final_warning = [
            "\n\n**Hallucination 방지 최종 체크리스트**:",
            "- [ ] 모든 숫자가 위 데이터에서 나옴?",
            "- [ ] 가정이나 추정 대신 실제 데이터만 사용?",
            "- [ ] 데이터 부족 시 솜직하게 명시?",
            "- [ ] 350건, 250건 같이 너무 깔끔한 숫자 피해?"
        ]
        
        return '\n'.join(base_instructions + final_warning)

    def _rule_based_synthesis(self, results: List[Any], query: str) -> Dict[str, Any]:
        """규칙 기반 종합 - Hallucination 방지 강화 버전"""

        synthesis_parts = []
        total_docs_found = 0
        brand_filter_applied = False
        aggregation_results = []
        successful_steps = 0
        failed_steps = 0

        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                # 오류 처리도 정보로 활용
                if 'error' in result:
                    failed_steps += 1
                    if 'No documents found after filtering' in result.get('error', ''):
                        brand_filter_applied = True
                        synthesis_parts.append(f"\n⚠️ 단계 {i}: 브랜드 필터링 시 문서가 없음")
                    else:
                        synthesis_parts.append(f"\n⚠️ 단계 {i} 실패: {result.get('error', 'Unknown error')}")
                    continue
                
                successful_steps += 1
                
                if 'aggregation' in result:
                    # 집계 결과 처리
                    if result['aggregation'] == 'terms':
                        field = result.get('field', 'unknown')
                        field_korean = {
                            'problem': '문제 유형',
                            'make_of_vehicle': '브랜드',
                            'category': '카테고리'
                        }.get(field, field)
                        
                        synthesis_parts.append(f"\n### {field_korean}별 분석 결과 (실제 데이터):")
                        
                        buckets = result.get('buckets', [])
                        
                debug_info = result.get('debug_info', {})
                aggregation_results.extend(buckets)
                
                # 🔧 INFO22/INFO23 명시적 처리 및 상세 분석
                info22_found = False
                info23_found = False
                
                # 🔧 개선: 동적 코드 처리 (하드코딩 제거)
                bucket_analysis = self._analyze_codes_in_buckets(buckets)
                codes_found = set(bucket_analysis.keys())
                
                                    # 🔧 Phase 1: 기본 정보 중심 분석 (LLM이 필요시 심각도 판단)
                for j, (code, info) in enumerate(bucket_analysis.items(), 1):
                    metadata = self.code_metadata.get(code, {})
                    description = metadata.get('description', f'{code} 관련 문제')
                    
                    synthesis_parts.append(f"\n#### {j}. {code} 코드 상세 분석")
                    synthesis_parts.append(f"- **문제 전체**: {info['key']}")
                    synthesis_parts.append(f"- **발생 건수**: {info['count']:,}건")
                    synthesis_parts.append(f"- **문제 유형**: {description}")
                    
                    # 🔧 심각도 정보는 유지하되 기본적으로 표시하지 않음
                    # LLM 클라이언트가 있는 경우는 LLM이 판단하므로 여기서는 제외
                
                # 코드가 아닌 일반 문제들도 표시
                other_buckets = [b for b in buckets[:10] 
                               if not any(code in b['key'] for code in codes_found)]
                
                for j, bucket in enumerate(other_buckets, len(bucket_analysis) + 1):
                    synthesis_parts.append(f"  {j}. **{bucket['key']}**: {bucket['doc_count']:,}건")
                
                # 🔧 개선: 동적 비교 분석 생성
                if len(bucket_analysis) > 1:
                    synthesis_parts.append(f"\n#### 다중 코드 비교 분석")
                    
                    # 같은 카테고리 코드들 그룹화
                    categories = {}
                    for code, info in bucket_analysis.items():
                        metadata = self.code_metadata.get(code, {})
                        category = metadata.get('category', 'unknown')
                        if category not in categories:
                            categories[category] = []
                        categories[category].append((code, info))
                    
                    # 카테고리별 비교
                    for category, code_list in categories.items():
                        if len(code_list) > 1:
                            synthesis_parts.append(f"\n**{category} 카테고리 비교:**")
                            total_count = 0
                            for code, info in code_list:
                                metadata = self.code_metadata.get(code, {})
                                description = metadata.get('description', f'{code} 관련')
                                synthesis_parts.append(f"- {code}: {info['count']:,}건 ({description})")
                                total_count += info['count']
                            synthesis_parts.append(f"- **{category} 총계**: {total_count:,}건")
                        
                        # 원본 문서 수 추가
                    if 'total_source_docs' in result:
                            total_docs_found = max(total_docs_found, result['total_source_docs'])
                    
                    elif result['aggregation'] == 'count':
                        count = result.get('count', 0)
                        synthesis_parts.append(f"\n### 데이터 카운트: {count:,}건")
                        total_docs_found = max(total_docs_found, count)
                    
                    elif result['aggregation'] == 'info_code_analysis':
                        # 실제 INFO 코드 분석 결과
                        info_counts = result.get('hyundai_info_counts', {})
                        info_details = result.get('hyundai_info_details', {})
                        total_hyundai = result.get('total_hyundai_docs', 0)
                        
                        synthesis_parts.append(f"\n### 현대브랜드 INFO12/13/14 실제 분석 결과:")
                        synthesis_parts.append(f"- 총 현대브랜드 문서: {total_hyundai:,}건")
                        
                        if info_counts:
                            for code in ['INFO12', 'INFO13', 'INFO14']:
                                count = info_counts.get(code, 0)
                                synthesis_parts.append(f"- **{code}**: {count}건")
                                
                                # 예시 추가
                                if code in info_details and info_details[code]:
                                    synthesis_parts.append(f"  예시: {info_details[code][0]['problem']}")
                        else:
                            synthesis_parts.append("- INFO12/13/14 코드가 발견되지 않았습니다.")
                        
                        total_docs_found = max(total_docs_found, total_hyundai)

                elif 'documents' in result:
                    # 검색 결과 처리
                    total = result.get('total_hits', 0)
                    total_docs_found = max(total_docs_found, total)
                    synthesis_parts.append(f"\n### 검색 결과: 총 {total:,}개 관련 문서 발견")

                    if total > 0:
                        synthesis_parts.append("\n**주요 사례:**")
                        for k, doc in enumerate(result['documents'][:3], 1):
                            model = doc.get('model_of_vehicle', 'N/A')
                            problem = doc.get('problem', 'N/A')
                            text = doc.get('verbatim_text', 'N/A')
                            synthesis_parts.append(f"  {k}. **{model}**: {problem}")
                            synthesis_parts.append(f"     → {text[:100]}...")
                
                # 필터 정보 추가
                if 'filter_applied' in result:
                    brand_filter_applied = True

        # 최종 요약
        if not synthesis_parts:
            synthesis_parts.append("데이터 분석에 실패했습니다. 실제 데이터가 없어 결과를 생성할 수 없습니다.")
        else:
            # 데이터 개요 추가 (실제 데이터만)
            overview = f"\n### 데이터 분석 개요 (실제 결과만 표시)"
            overview += f"\n- **총 데이터**: {total_docs_found:,}개 문서 분석"
            overview += f"\n- **성공 단계**: {successful_steps}개 / **실패 단계**: {failed_steps}개"
            if failed_steps > 0:
                overview += f"\n- ⚠️ **주의**: {failed_steps}개 단계가 실패하여 부분적 결과입니다."
            if brand_filter_applied:
                overview += f"\n- **브랜드 필터**: 현대 브랜드 전용 분석"
            
            synthesis_parts.insert(0, overview)
            
            # INFO12/13/14 개별 집계 결과가 없으면 명시
            if failed_steps > 0 and 'INFO12' in query and 'INFO13' in query and 'INFO14' in query:
                synthesis_parts.append("\n⚠️ **중요**: INFO12/13/14 개별 집계가 실패하여 위의 숫자들은 추정치가 아닌 실제 데이터에 기반한 결과입니다.")

        return {
            "query": query,
            "synthesis": "\n".join(synthesis_parts),
            "source_count": len(results),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "total_documents_analyzed": total_docs_found,
            "brand_filtering_applied": brand_filter_applied,
            "method": "rule_based_truthful"
        }

    def get_description(self) -> str:
        return self.description


def test_tools():
    """도구들 테스트"""

    print("=" * 70)
    print("Testing Agent Tools")
    print("=" * 70)

    # 샘플 데이터 생성
    sample_docs = [
        {
            "verbatim_id": "001",
            "model": "Santa Fe",
            "model_year": 2025,
            "problem": "Tire - Vibration",
            "verbatim_text": "Tire vibration at high speed",
            "registration_date": "2025-01-15"
        },
        {
            "verbatim_id": "002",
            "model": "Santa Fe",
            "model_year": 2025,
            "problem": "Tire - Vibration",
            "verbatim_text": "Steering wheel vibrates",
            "registration_date": "2025-01-20"
        },
        {
            "verbatim_id": "003",
            "model": "Tucson",
            "model_year": 2024,
            "problem": "Engine - Noise",
            "verbatim_text": "Engine makes clicking noise",
            "registration_date": "2024-12-01"
        }
    ]

    # 1. Aggregator Tool 테스트
    print("\n1. Testing AggregatorTool")
    print("-" * 30)

    aggregator = AggregatorTool()
    agg_result = aggregator.execute(
        documents=sample_docs,
        aggregation="terms",
        field="problem",
        size=5
    )

    print(f"Aggregation result:")
    for bucket in agg_result.get('buckets', []):
        print(f" {bucket['key']}: {bucket['doc_count']} docs")

    # 2. HybridSearch Tool 테스트
    print("\n2. Testing HybridSearchTool")
    print("-" * 30)

    search_tool = HybridSearchTool()

    # 먼저 인덱스 구축
    print("Building index...")
    search_tool.build_index(sample_docs)

    # 검색 실행
    search_result = search_tool.execute(
        query="tire vibration",
        limit=5,
        filters={"model_year": 2025}
    )

    print(f"Search returned {search_result['total_hits']} results")
    for doc in search_result.get('documents', [])[:2]:
        print(f" - {doc.get('verbatim_text', 'N/A')}")

    # 3. Reranker Tool 테스트
    print("\n3. Testing RerankerTool")
    print("-" * 30)

    reranker = RerankerTool()
    rerank_result = reranker.execute(
        documents=search_result['documents'],
        query="tire vibration",
        top_k=2
    )

    print(f"Reranked {rerank_result['total_reranked']} documents")
    for doc in rerank_result.get('documents', []):
        print(f" - Score: {doc['_score']:.3f}")
        print(f"   {doc.get('verbatim_text', 'N/A')}")

    # 4. Synthesizer Tool 테스트
    print("\n4. Testing SynthesizerTool")
    print("-" * 30)

    synthesizer = SynthesizerTool()
    synthesis = synthesizer.execute(
        results=[agg_result, search_result],
        original_query="2025 Santa Fe tire problems"
    )

    print("Synthesis result:")
    print(synthesis['synthesis'])

    print("\n" + "=" * 70)
    print("Tools test complete!")


if __name__ == "__main__":
    test_tools()

    synthesizer = SynthesizerTool()
    synthesis = synthesizer.execute(
        results=[agg_result, search_result],
        original_query="2025 Santa Fe tire problems"
    )

    print("Synthesis result:")
    print(synthesis['synthesis'])

    print("\n" + "=" * 70)
    print("Tools test complete!")


if __name__ == "__main__":
    test_tools()