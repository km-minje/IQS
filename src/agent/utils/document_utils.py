"""
Document extraction and manipulation utilities
문서 추출 및 조작을 위한 통합 유틸리티

이 모듈은 Agent 도구들 간의 문서 전달 로직을 통합
"""
from typing import List, Dict, Any, Optional, Union, Tuple
from src.utils.logger import log


class DocumentExtractor:
    """
    이전 결과나 다양한 소스에서 문서를 안전하게 추출하는 통합 클래스
    AggregatorTool과 RerankerTool에서 중복되던 로직을 통합
    """
    
    @staticmethod
    def extract_from_previous_result(
        previous_result: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], 
        documents: Optional[List[Dict[str, Any]]] = None,
        fallback_fields: Optional[List[str]] = None,
        extract_query: bool = True,
        log_details: bool = True
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        이전 결과에서 문서와 쿼리를 안전하게 추출
        
        Args:
            previous_result: 이전 단계의 결과 (dict 또는 list)
            documents: 직접 제공된 문서 리스트 (우선순위 높음)
            fallback_fields: 문서를 찾을 추가 필드명 리스트 (기본: ['source_documents'])
            extract_query: 쿼리도 함께 추출할지 여부
            log_details: 상세 로깅 여부
            
        Returns:
            (추출된 문서 리스트, 추출된 쿼리 또는 None)
        """
        if fallback_fields is None:
            fallback_fields = ['source_documents']
        
        extracted_documents = []
        extracted_query = None
        
        # 1. 직접 제공된 문서가 있으면 우선 사용
        if documents:
            extracted_documents = documents
            if log_details:
                log.info(f"Using directly provided documents: {len(extracted_documents)} docs")
        
        # 2. 이전 결과가 없으면 바로 반환
        elif previous_result is None:
            if log_details:
                log.warning("No previous result and no documents provided")
            return [], None
        
        # 3. 이전 결과가 리스트면 직접 사용
        elif isinstance(previous_result, list):
            extracted_documents = previous_result
            if log_details:
                log.info(f"Using previous result as document list: {len(extracted_documents)} docs")
        
        # 4. 이전 결과가 딕셔너리면 여러 필드에서 시도
        elif isinstance(previous_result, dict):
            # 4-1. 기본 'documents' 필드 확인
            extracted_documents = previous_result.get('documents', [])
            
            if extracted_documents:
                if log_details:
                    log.info(f"Using 'documents' from previous result: {len(extracted_documents)} docs")
            else:
                # 4-2. fallback 필드들에서 순차 시도
                for field in fallback_fields:
                    if field in previous_result and previous_result[field]:
                        extracted_documents = previous_result[field]
                        if log_details:
                            log.info(f"Using '{field}' from previous result: {len(extracted_documents)} docs")
                        break
                
                # 4-3. 여전히 문서가 없으면 경고
                if not extracted_documents and log_details:
                    available_keys = list(previous_result.keys())
                    log.warning(f"No documents found in previous result. Available keys: {available_keys}")
            
            # 4-4. 쿼리 추출 (요청된 경우)
            if extract_query:
                extracted_query = previous_result.get('query') or previous_result.get('original_query')
        
        else:
            if log_details:
                log.error(f"Invalid previous_result type: {type(previous_result)}")
            return [], None
        
        # 5. 결과 검증
        if not isinstance(extracted_documents, list):
            if log_details:
                log.warning(f"Extracted documents is not a list: {type(extracted_documents)}")
            extracted_documents = []
        
        # 6. 최종 로깅
        if log_details:
            log.info(f"DocumentExtractor: Successfully extracted {len(extracted_documents)} documents"
                    + (f" and query: '{extracted_query}'" if extracted_query else ""))
        
        return extracted_documents, extracted_query
    
    @staticmethod
    def extract_documents_only(
        previous_result: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], 
        documents: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        문서만 추출하는 간편한 메서드 (쿼리는 추출하지 않음)
        
        Args:
            previous_result: 이전 결과
            documents: 직접 제공된 문서
            **kwargs: extract_from_previous_result에 전달할 추가 인자
            
        Returns:
            추출된 문서 리스트
        """
        extracted_docs, _ = DocumentExtractor.extract_from_previous_result(
            previous_result=previous_result,
            documents=documents,
            extract_query=False,
            **kwargs
        )
        return extracted_docs
    
    @staticmethod
    def safe_document_access(
        documents: List[Dict[str, Any]], 
        index: int, 
        field: str = None,
        default: Any = None
    ) -> Any:
        """
        문서 리스트에서 안전하게 데이터에 접근
        
        Args:
            documents: 문서 리스트
            index: 접근할 인덱스
            field: 접근할 필드명 (None이면 전체 문서 반환)
            default: 기본값
            
        Returns:
            추출된 값 또는 기본값
        """
        try:
            if not documents or index >= len(documents):
                return default
            
            doc = documents[index]
            if field is None:
                return doc
            
            return doc.get(field, default)
        except (IndexError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def validate_documents(
        documents: List[Dict[str, Any]], 
        required_fields: Optional[List[str]] = None,
        min_count: int = 0
    ) -> Tuple[bool, str]:
        """
        문서 리스트의 유효성 검증
        
        Args:
            documents: 검증할 문서 리스트
            required_fields: 필수 필드 목록
            min_count: 최소 문서 개수
            
        Returns:
            (유효성 여부, 오류 메시지)
        """
        if not isinstance(documents, list):
            return False, f"Documents must be a list, got {type(documents)}"
        
        if len(documents) < min_count:
            return False, f"Need at least {min_count} documents, got {len(documents)}"
        
        if required_fields:
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    return False, f"Document {i} is not a dict: {type(doc)}"
                
                missing_fields = [field for field in required_fields if field not in doc]
                if missing_fields:
                    return False, f"Document {i} missing fields: {missing_fields}"
        
        return True, "Documents are valid"
    
    @staticmethod
    def count_documents_by_field(
        documents: List[Dict[str, Any]], 
        field: str,
        case_sensitive: bool = False
    ) -> Dict[str, int]:
        """
        특정 필드 값별로 문서 개수 카운트
        
        Args:
            documents: 문서 리스트
            field: 카운트할 필드
            case_sensitive: 대소문자 구분 여부
            
        Returns:
            {필드값: 개수} 딕셔너리
        """
        counts = {}
        
        for doc in documents:
            if not isinstance(doc, dict) or field not in doc:
                continue
            
            value = doc[field]
            if not case_sensitive and isinstance(value, str):
                value = value.lower()
            
            counts[value] = counts.get(value, 0) + 1
        
        return counts


# 하위 호환성을 위한 함수 형태 래퍼들
def extract_documents_from_result(previous_result, documents=None, **kwargs):
    """하위 호환성을 위한 함수 래퍼"""
    return DocumentExtractor.extract_documents_only(previous_result, documents, **kwargs)


def safe_get_document_field(documents, index, field, default=None):
    """하위 호환성을 위한 함수 래퍼"""  
    return DocumentExtractor.safe_document_access(documents, index, field, default)