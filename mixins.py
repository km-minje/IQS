"""
Agent Tool Mixins - 공통 기능을 위한 Mixin 클래스들
자기 평가, 에러 처리, 로깅 등 중복 기능을 통합
"""
import json
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from src.utils.logger import log


class SelfAssessmentMixin:
    """
    자기 평가 기능을 위한 Mixin 클래스
    
    모든 도구에서 중복되던 assess_query_suitability()와 assess_step_suitability() 로직을 통합
    설정 기반 평가 시스템으로 하드코딩 제거
    """
    
    # 🔧 새로운 기능: 클래스 레벨 평가 결과 캐싱
    _assessment_cache = {}
    
    def __init__(self, *args, **kwargs):
        """
        Mixin 초기화 - 다중 상속 호환성 보장
        """
        super().__init__(*args, **kwargs)
        
        # 설정 로드 (필요시 지연 로딩)
        self._assessment_config = None
        self._global_assessment_config = None
    
    def _load_assessment_config(self) -> Dict[str, Any]:
        """
        도구별 평가 설정 로드
        각 도구에서 override하거나 tool name 기반으로 자동 로드
        
        Returns:
            해당 도구의 평가 설정
        """
        if self._assessment_config is not None:
            return self._assessment_config
        
        try:
            # 전체 설정 파일 로드
            if self._global_assessment_config is None:
                config_path = Path("config/tool_assessment.json")
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self._global_assessment_config = json.load(f)
                        # 🔧 새로운 기능: 설정 파일 검증
                        self._validate_config(self._global_assessment_config)
                else:
                    log.warning(f"Assessment config file not found: {config_path}")
                    self._global_assessment_config = {}
            
            # 도구명 자동 감지 (tool name 기반)
            tool_name = getattr(self, 'name', None)
            if tool_name and tool_name in self._global_assessment_config:
                self._assessment_config = self._global_assessment_config[tool_name]
                log.debug(f"Loaded assessment config for tool: {tool_name}")
            else:
                # 기본 설정 사용
                self._assessment_config = self._get_default_assessment_config()
                log.warning(f"Using default assessment config for tool: {tool_name}")
                
        except Exception as e:
            log.error(f"Failed to load assessment config: {e}")
            self._assessment_config = self._get_default_assessment_config()
        
        return self._assessment_config
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        🔧 새로운 기능: 설정 파일 유효성 검사
        
        Args:
            config: 검증할 설정
        """
        required_tools = ['aggregator', 'hybrid_search', 'reranker', 'glossary_lookup', 'synthesizer']
        required_fields = ['query_keywords', 'base_score', 'threshold']
        
        validation_errors = []
        
        for tool_name in required_tools:
            if tool_name not in config:
                validation_errors.append(f"Missing tool config: {tool_name}")
                continue
            
            tool_config = config[tool_name]
            for field in required_fields:
                if field not in tool_config:
                    validation_errors.append(f"Missing field '{field}' in {tool_name}")
            
            # 점수 범위 검증
            base_score = tool_config.get('base_score', 0)
            threshold = tool_config.get('threshold', 0)
            
            if not 0 <= base_score <= 1:
                validation_errors.append(f"Invalid base_score in {tool_name}: {base_score}")
            if not 0 <= threshold <= 1:
                validation_errors.append(f"Invalid threshold in {tool_name}: {threshold}")
        
        if validation_errors:
            log.warning(f"Config validation issues: {validation_errors}")
        else:
            log.debug("Config validation passed")
    
    def _get_default_assessment_config(self) -> Dict[str, Any]:
        """
        기본 평가 설정 (설정 파일이 없을 때)
        
        Returns:
            기본 평가 설정
        """
        return {
            "query_keywords": [],
            "base_score": 0.3,
            "keyword_score": 0.1,
            "threshold": 0.4,
            "suggested_params": {},
            "intent_weights": {}
        }
    
    def assess_query_suitability(self, query: str) -> Dict[str, Any]:
        """
        🔧 통합된 쿼리 적합성 평가 - 모든 도구의 중복 로직 대체
        
        설정 기반으로 키워드 감지, 점수 계산, 임계값 비교를 수행
        
        Args:
            query: 평가할 쿼리
            
        Returns:
            적합성 평가 결과
        """
        # 🔧 새로운 기능: 캐싱 시스템
        tool_name = getattr(self, 'name', 'unknown')
        cache_key = f"{tool_name}:{hash(query)}"
        
        if cache_key in self._assessment_cache:
            log.debug(f"Assessment cache hit for {tool_name}: {query}")
            return self._assessment_cache[cache_key]
        
        config = self._load_assessment_config()
        query_lower = query.lower()
        
        # 기본 점수 설정
        relevance_score = config.get("base_score", 0.3)
        reasons = []
        
        # 1. 기본 키워드 검사
        query_keywords = config.get("query_keywords", [])
        keyword_score = config.get("keyword_score", 0.1)
        
        for keyword in query_keywords:
            if keyword in query_lower:
                relevance_score += keyword_score
                reasons.append(f"'{keyword}' 키워드 발견")
        
        # 2. 추가 키워드 검사 (도구별 특별 처리)
        additional_analysis = self._assess_additional_patterns(query_lower, config, reasons)
        relevance_score += additional_analysis
        
        # 3. 신뢰도 계산 및 적합성 판단
        confidence = min(relevance_score, 1.0)
        threshold = config.get("threshold", 0.4)
        suitable = confidence > threshold
        
        # 4. 제안 파라미터 동적 생성
        suggested_params = self._generate_suggested_params(confidence, config)
        
        result = {
            'suitable': suitable,
            'confidence': confidence,
            'reason': '; '.join(reasons) if reasons else f'기본 평가 (도구: {getattr(self, "name", "unknown")})',
            'suggested_params': suggested_params,
            'assessment_method': 'config_based_mixin_v1.1',
            'cache_hit': False
        }
        
        # 🔧 새로운 기능: 결과 캐싱 (성능 최적화 적용)
        self._assessment_cache[cache_key] = result
        
        # 캐시 크기 제한 (100개를 초과하면 오래된 것부터 제거)
        if len(self._assessment_cache) > 100:
            # 가장 오래된 항목 제거 (FIFO) - 효율적 구현
            cache_items = list(self._assessment_cache.items())
            # 오래된 절반 제거
            for i in range(len(cache_items) // 2):
                key_to_remove = cache_items[i][0]
                del self._assessment_cache[key_to_remove]
            
            log.debug(f"Cache cleanup: removed {len(cache_items) // 2} old entries")
        
        log.debug(f"Assessment for '{query}': suitable={suitable}, confidence={confidence:.2f}, cached={result.get('cache_hit', False)}")
        return result
    
    def _assess_additional_patterns(self, query_lower: str, config: Dict[str, Any], reasons: List[str]) -> float:
        """
        추가 패턴 분석 - 도구별 특별한 평가 로직
        
        Args:
            query_lower: 소문자 쿼리
            config: 평가 설정
            reasons: 평가 이유 목록 (변경됨)
            
        Returns:
            추가 점수
        """
        additional_score = 0.0
        
        # 분석 키워드 (aggregator 등에서 사용)
        if "analysis_keywords" in config:
            analysis_score = config.get("analysis_score", 0.1)
            for keyword in config["analysis_keywords"]:
                if keyword in query_lower:
                    additional_score += analysis_score
                    reasons.append(f"'{keyword}' 분석 의도 감지")
        
        # 비교 키워드 (reranker 등에서 사용)
        if "comparison_keywords" in config:
            comparison_score = config.get("comparison_score", 0.1)
            for keyword in config["comparison_keywords"]:
                if keyword in query_lower:
                    additional_score += comparison_score
                    reasons.append(f"'{keyword}' 비교/필터링 요청")
        
        # 특별 패턴 처리 (search 등에서 사용)
        if "special_patterns" in config:
            patterns = config["special_patterns"]
            
            # 코드 패턴 감지
            if "code_pattern" in patterns:
                code_pattern = patterns["code_pattern"]
                if re.search(code_pattern, query_lower, re.IGNORECASE):
                    code_bonus = config.get("code_bonus", 0.2)
                    additional_score += code_bonus
                    reasons.append("코드 패턴 감지")
            
            # 🔧 새로운 기능: 한글 패턴 감지 (glossary_lookup 용)
            if "korean_pattern" in patterns:
                korean_pattern = patterns["korean_pattern"]
                if re.search(korean_pattern, query_lower):
                    korean_bonus = config.get("korean_bonus", 0.3)
                    additional_score += korean_bonus
                    reasons.append("한글 텍스트 감지")
            
            # 🔧 새로운 기능: 한글+코드 복합 패턴
            if "code_with_korean" in patterns:
                code_korean_pattern = patterns["code_with_korean"]
                if re.search(code_korean_pattern, query_lower):
                    additional_score += 0.4  # 높은 점수
                    reasons.append("한글+코드 복합 패턴 감지")
            
            # 대용량 데이터 요청 감지
            if "large_data_keywords" in patterns:
                large_data_bonus = config.get("large_data_bonus", 0.15)
                for keyword in patterns["large_data_keywords"]:
                    if keyword in query_lower:
                        additional_score += large_data_bonus
                        reasons.append("대용량 데이터 요청")
                        break
        
        return additional_score
    
    def _generate_suggested_params(self, confidence: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        신뢰도 기반 제안 파라미터 동적 생성
        
        Args:
            confidence: 계산된 신뢰도
            config: 도구 설정
            
        Returns:
            제안 파라미터
        """
        suggested_params = config.get("suggested_params", {}).copy()
        
        # 신뢰도에 따른 동적 파라미터 조정
        if confidence > 0.8:
            # 매우 높은 신뢰도 - 최대 성능
            if "limit_massive" in suggested_params:
                suggested_params["limit"] = suggested_params["limit_massive"]
            elif "limit_high" in suggested_params:
                suggested_params["limit"] = suggested_params["limit_high"]
            if "size_high" in suggested_params:
                suggested_params["size"] = suggested_params["size_high"]
            if "top_k_high" in suggested_params:
                suggested_params["top_k"] = suggested_params["top_k_high"]
        elif confidence > 0.6:
            # 높은 신뢰도 - 더 세밀한 분석
            if "size_high" in suggested_params:
                suggested_params["size"] = suggested_params["size_high"]
            if "limit_high" in suggested_params:
                suggested_params["limit"] = suggested_params["limit_high"]
            if "top_k_high" in suggested_params:
                suggested_params["top_k"] = suggested_params["top_k_high"]
        else:
            # 기본 신뢰도 - 표준 분석
            if "size_low" in suggested_params:
                suggested_params["size"] = suggested_params["size_low"]
            if "limit_low" in suggested_params:
                suggested_params["limit"] = suggested_params["limit_low"]
            if "top_k_low" in suggested_params:
                suggested_params["top_k"] = suggested_params["top_k_low"]
        
        # 🔧 새로운 기능: 도구별 특별 파라미터 처리
        tool_name = getattr(self, 'name', 'unknown')
        if tool_name == 'glossary_lookup':
            # 한글 감지시 동의어 포함
            suggested_params['include_synonyms'] = confidence > 0.7
            suggested_params['include_codes'] = True
        elif tool_name == 'synthesizer':
            # 종합 도구는 LLM 사용 여부 결정
            suggested_params['llm_synthesis'] = confidence > 0.8
            suggested_params['include_metadata'] = True
        
        # 임시 키 제거
        for key in ["size_low", "size_high", "limit_low", "limit_high", "limit_massive", "top_k_low", "top_k_high"]:
            suggested_params.pop(key, None)
        
        return suggested_params
    
    def assess_step_suitability(self, step: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        🔧 통합된 단계 적합성 평가 - 모든 도구의 중복 로직 대체
        
        Args:
            step: 실행 단계 정보
            context: 컨텍스트 정보
            
        Returns:
            적합성 점수 (0.0 ~ 1.0)
        """
        config = self._load_assessment_config()
        step_desc = step.get('description', '').lower()
        intent = context.get('intent', 'unknown')
        
        # 기본 점수
        suitability_score = config.get("base_score", 0.3)
        
        # 의도 기반 점수 조정
        intent_weights = config.get("intent_weights", {})
        if intent in intent_weights:
            suitability_score += intent_weights[intent]
        
        # 단계 설명 기반 키워드 매칭
        query_keywords = config.get("query_keywords", [])
        for keyword in query_keywords:
            if keyword in step_desc:
                suitability_score += 0.1
                break
        
        # 추가 키워드 매칭
        if "analysis_keywords" in config:
            for keyword in config["analysis_keywords"]:
                if keyword in step_desc:
                    suitability_score += 0.1
                    break
        
        if "comparison_keywords" in config:
            for keyword in config["comparison_keywords"]:
                if keyword in step_desc:
                    suitability_score += 0.1
                    break
        
        final_score = min(suitability_score, 1.0)
        log.debug(f"Step assessment: intent={intent}, score={final_score:.2f}")
        
        return final_score
    
    def get_assessment_info(self) -> Dict[str, Any]:
        """
        현재 도구의 평가 설정 정보 반환 (디버깅용)
        
        Returns:
            평가 설정 정보
        """
        config = self._load_assessment_config()
        return {
            "tool_name": getattr(self, 'name', 'unknown'),
            "assessment_config": config,
            "assessment_method": "SelfAssessmentMixin_v1.1",
            "config_loaded": self._assessment_config is not None,
            "cache_size": len(self._assessment_cache),
            "version": "1.1.0",
            "features": [
                "config_based_assessment", 
                "result_caching", 
                "pattern_detection", 
                "dynamic_parameters",
                "validation_system"
            ]
        }
    
    def clear_cache(self):
        """
        🔧 새로운 기능: 평가 결과 캐시 클리어
        """
        cleared_count = len(self._assessment_cache)
        self._assessment_cache.clear()
        log.info(f"Cleared assessment cache: {cleared_count} entries")
        return cleared_count
    
    @classmethod
    def get_global_cache_stats(cls) -> Dict[str, Any]:
        """
        🔧 새로운 기능: 전체 캐시 통계
        
        Returns:
            전체 캐시 통계 정보
        """
        return {
            "cache_size": len(cls._assessment_cache),
            "cache_limit": 100,
            "cache_efficiency": "FIFO eviction policy"
        }


class ErrorHandlingMixin:
    """
    에러 처리 기능을 위한 Mixin 클래스
    모든 도구에서 공통으로 사용할 에러 처리 패턴
    """
    
    def safe_execute(self, execute_func, *args, **kwargs):
        """
        안전한 실행 래퍼 - 공통 에러 처리 패턴
        
        Args:
            execute_func: 실행할 함수
            *args, **kwargs: 함수 인자들
            
        Returns:
            실행 결과 또는 에러 정보
        """
        tool_name = getattr(self, 'name', 'unknown_tool')
        
        try:
            log.info(f"Executing {tool_name}...")
            result = execute_func(*args, **kwargs)
            log.info(f"{tool_name} completed successfully")
            return result
            
        except Exception as e:
            log.error(f"{tool_name} execution failed: {e}")
            return {
                "error": str(e),
                "tool": tool_name,
                "error_type": type(e).__name__,
                "execution_method": "safe_execute_mixin"
            }


class LoggingMixin:
    """
    로깅 기능을 위한 Mixin 클래스
    일관된 로깅 형식 제공
    """
    
    def log_execution_start(self, operation: str, **details):
        """실행 시작 로깅"""
        tool_name = getattr(self, 'name', 'unknown_tool')
        log.info(f"{tool_name}: Starting {operation} - {details}")
    
    def log_execution_end(self, operation: str, success: bool, **details):
        """실행 완료 로깅"""
        tool_name = getattr(self, 'name', 'unknown_tool')
        status = "SUCCESS" if success else "FAILED"
        log.info(f"{tool_name}: {operation} {status} - {details}")