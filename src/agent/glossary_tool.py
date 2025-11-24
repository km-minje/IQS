"""
Glossary Lookup Tool - 한-영 용어 변환 도구
LLM Agent가 한글 쿼리를 영어로 변환할 때 사용
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import log


class GlossaryLookupTool:
    """
    용어 사전 조회 도구
    한글 쿼리를 영어로 변환하고 동의어 확장
    """
    
    def __init__(self, glossary_path: Optional[str] = None):
        """
        Initialize Glossary Lookup Tool
        
        Args:
            glossary_path: 용어 사전 파일 경로
        """
        self.name = "glossary_lookup"
        self.description = "한-영 용어 변환 및 동의어 확장"
        
        # 기본 사전 파일 경로
        if glossary_path is None:
            glossary_path = "data/glossary/kor_eng_automotive_terms.json"
        
        self.glossary_path = Path(glossary_path)
        self.glossary_data = self._load_glossary()
        
        log.info(f"Initialized GlossaryLookupTool with {len(self.glossary_data.get('translations', {}))} terms")
    
    def _load_glossary(self) -> Dict:
        """용어 사전 로드"""
        try:
            if self.glossary_path.exists():
                with open(self.glossary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    log.success(f"Loaded glossary from {self.glossary_path}")
                    return data
            else:
                log.warning(f"Glossary file not found: {self.glossary_path}")
                return self._get_default_glossary()
        except Exception as e:
            log.error(f"Failed to load glossary: {e}")
            return self._get_default_glossary()
    
    def _get_default_glossary(self) -> Dict:
        """기본 용어 사전 (파일이 없을 때)"""
        return {
            "translations": {
                "인포테인먼트": "infotainment",
                "엔진": "engine", 
                "타이어": "tire",
                "브레이크": "brake",
                "소음": "noise",
                "문제": "problem"
            },
            "automotive_codes": {
                "INFO": "infotainment",
                "PWR": "powertrain",
                "EXT": "exterior"
            },
            "synonyms": {
                "infotainment": ["entertainment", "multimedia", "touchscreen"]
            }
        }
    
    def translate_korean_to_english(self, text: str) -> str:
        """
        한글 텍스트를 영어로 번역
        
        Args:
            text: 번역할 텍스트
            
        Returns:
            번역된 텍스트
        """
        if not self.glossary_data or 'translations' not in self.glossary_data:
            return text
        
        translations = self.glossary_data['translations']
        translated_text = text
        
        # 한글 단어들을 찾아서 번역
        korean_words = re.findall(r'[가-힣]+', text)
        
        for korean_word in korean_words:
            if korean_word in translations:
                english_word = translations[korean_word]
                translated_text = translated_text.replace(korean_word, english_word)
                log.debug(f"Translated: {korean_word} -> {english_word}")
        
        return translated_text
    
    def expand_automotive_codes(self, text: str) -> str:
        """
        자동차 코드를 확장
        
        Args:
            text: 입력 텍스트
            
        Returns:
            코드가 확장된 텍스트
        """
        if not self.glossary_data or 'automotive_codes' not in self.glossary_data:
            return text
        
        codes = self.glossary_data['automotive_codes']
        expanded_text = text
        
        # 코드 패턴 찾기 (예: INFO, FCD35)
        code_pattern = r'\b([A-Z]{2,5})\d*\b'
        matches = re.findall(code_pattern, text)
        
        for code in matches:
            if code in codes:
                expansion = codes[code]
                # 원본 코드 유지하면서 의미 추가
                expanded_text += f" {expansion}"
                log.debug(f"Expanded code: {code} -> {expansion}")
        
        return expanded_text
    
    def get_synonyms(self, word: str) -> List[str]:
        """
        단어의 동의어 목록 반환
        
        Args:
            word: 검색할 단어
            
        Returns:
            동의어 목록
        """
        if not self.glossary_data or 'synonyms' not in self.glossary_data:
            return []
        
        synonyms = self.glossary_data['synonyms']
        word_lower = word.lower()
        
        if word_lower in synonyms:
            return synonyms[word_lower]
        
        return []
    
    def enhance_query(self, query: str) -> Tuple[str, List[str]]:
        """
        쿼리 전체적 향상
        
        Args:
            query: 원본 쿼리
            
        Returns:
            (향상된 쿼리, 추가 검색어 목록)
        """
        log.info(f"Enhancing query: '{query}'")
        
        # 1. 한글 → 영어 번역
        enhanced_query = self.translate_korean_to_english(query)
        
        # 2. 자동차 코드 확장
        enhanced_query = self.expand_automotive_codes(enhanced_query)
        
        # 3. 동의어 수집
        additional_terms = []
        words = re.findall(r'\b\w+\b', enhanced_query.lower())
        
        for word in words:
            synonyms = self.get_synonyms(word)
            additional_terms.extend(synonyms)
        
        # 중복 제거
        additional_terms = list(set(additional_terms))
        
        log.info(f"Enhanced: '{query}' -> '{enhanced_query}'")
        if additional_terms:
            log.info(f"Additional terms: {additional_terms}")
        
        return enhanced_query, additional_terms
    
    def execute(self, 
                query: str,
                include_synonyms: bool = True,
                include_codes: bool = True) -> Dict[str, any]:
        """
        도구 실행 (Agent에서 호출)
        
        Args:
            query: 변환할 쿼리
            include_synonyms: 동의어 포함 여부
            include_codes: 코드 확장 포함 여부
            
        Returns:
            변환 결과
        """
        try:
            log.info(f"Executing glossary lookup for: '{query}'")
            
            # 쿼리 향상
            enhanced_query, synonyms = self.enhance_query(query)
            
            result = {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "translation_applied": enhanced_query != query,
                "additional_terms": synonyms if include_synonyms else [],
                "suggested_searches": []
            }
            
            # 추천 검색어 생성
            suggested_searches = [enhanced_query]
            if include_synonyms and synonyms:
                # 주요 동의어와 조합
                for synonym in synonyms[:3]:  # 상위 3개만
                    suggested_searches.append(f"{enhanced_query} {synonym}")
            
            result["suggested_searches"] = suggested_searches
            
            log.success(f"Glossary lookup completed: {len(suggested_searches)} search variations")
            return result
            
        except Exception as e:
            log.error(f"Glossary lookup failed: {e}")
            return {
                "original_query": query,
                "enhanced_query": query,
                "translation_applied": False,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        """도구 설명 반환"""
        return self.description


def test_glossary_tool():
    """Glossary Tool 테스트"""
    print("=" * 60)
    print("Testing Glossary Lookup Tool")
    print("=" * 60)
    
    # 도구 초기화
    tool = GlossaryLookupTool()
    
    # 테스트 쿼리들
    test_queries = [
        "인포테인먼트 문제",
        "엔진 소음 문제",
        "타이어 진동",
        "FCD35 코드",
        "INFO 관련 문제",
        "브레이크 소음",
        "터치스크린 오작동"
    ]
    
    for query in test_queries:
        print(f"\n쿼리: '{query}'")
        result = tool.execute(query)
        
        print(f"  원본: {result['original_query']}")
        print(f"  향상: {result['enhanced_query']}")
        print(f"  번역 적용: {result['translation_applied']}")
        
        if result.get('additional_terms'):
            print(f"  추가 용어: {result['additional_terms']}")
        
        if result.get('suggested_searches'):
            print(f"  추천 검색:")
            for i, search in enumerate(result['suggested_searches'][:3], 1):
                print(f"    {i}. {search}")
    
    print("\n" + "=" * 60)
    print("Glossary tool test complete!")


if __name__ == "__main__":
    test_glossary_tool()