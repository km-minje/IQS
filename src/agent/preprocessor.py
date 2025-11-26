"""
Korean Text Preprocessor using Kiwi
Based on llm-customized/kiwi_custom/tokenizer.py
"""
import os
import pandas as pd
from typing import List, Dict, Any, Tuple

try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None

from src.utils.logger import log
from config.settings import settings

class KiwiPreprocessor:
    """
    Kiwi 형태소 분석기를 사용한 전처리 클래스 (안전한 초기화 버전)
    """

    def __init__(self, dictionary_dir: str = "data/glossary"):
        self.dictionary_dir = dictionary_dir

        # 1. 변수 선언을 먼저 해서 AttributeError 방지
        self.kiwi = None
        self.entity_dict = {}
        self.initialized = False

        # 2. Kiwi 초기화 시도
        if Kiwi is None:
            log.error("kiwipiepy library not found. Preprocessing disabled.")
            return

                try:
            log.info("Initializing Kiwi...")
            # 먼저 기본 모델로 시도
            try:
                self.kiwi = Kiwi(num_workers=0)
            except Exception as basic_e:
                log.warning(f"Basic Kiwi initialization failed: {basic_e}")
                # 대안: 더 간단한 설정으로 시도
                try:
                    self.kiwi = Kiwi(num_workers=0, model_type=None)
                except Exception as alt_e:
                    log.error(f"Alternative Kiwi initialization also failed: {alt_e}")
                    raise alt_e

            # 사전 로드 실행
            self._initialize_dictionaries()
            self.initialized = True
            log.success("Kiwi Preprocessor initialized successfully")

        except Exception as e:
            # 3. 초기화 실패 시에도 self.kiwi는 None으로 유지되어 AttributeError 방지
            log.error(f"Failed to initialize Kiwi: {e}")
            log.warning("System will proceed without Korean morphological analysis.")
            self.kiwi = None

    def _initialize_dictionaries(self):
        """사전 파일 로드 및 Kiwi 사용자 단어 등록"""
        if not self.kiwi:
            return

        try:
            auto_path = os.path.join(self.dictionary_dir, "autopedia.xlsx")
            dict_path = os.path.join(self.dictionary_dir, "cft_dictionary.xlsx")

            # 1. Autopedia 데이터 추가
            if os.path.exists(auto_path):
                auto_df = pd.read_excel(auto_path, engine='openpyxl', sheet_name=1)
                auto_keywords = list(auto_df["용어명(*)[한국어]"]) + list(auto_df["용어명(*)[영어]"])
                for keyword in auto_keywords:
                    keyword = str(keyword).strip()
                    if keyword and not keyword.isdigit():
                        self.kiwi.add_user_word(keyword, 'NNP', 0)

            # 2. CFT 사전 데이터 추가
            if os.path.exists(dict_path):
                cft_df = pd.read_excel(dict_path, engine='openpyxl', sheet_name=1)
                cft_keywords = list(cft_df["키워드"])
                for keyword in cft_keywords:
                    keyword = str(keyword).strip()
                    if keyword and not keyword.isdigit():
                        self.kiwi.add_user_word(keyword, 'NNP', 0)

                # 3. Entity 매핑 데이터 로드
                entity_df = pd.read_excel(dict_path, engine='openpyxl', sheet_name=2)
                self.entity_dict = {
                    str(entity_df['키워드'][idx]): str(entity_df['이름'][idx])
                    for idx in range(len(entity_df["키워드"]))
                }

        except Exception as e:
            log.warning(f"Dictionary loading failed (partial functionality): {e}")

    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드(명사구) 추출"""
        # 안전장치: Kiwi가 없으면 빈 리스트 반환 (에러 발생 안 함)
        if not self.kiwi or not text:
            return []

        try:
            tokens = self.kiwi.tokenize(text, normalize_coda=False)
            keywords = []

            for i in range(len(tokens)):
                token = tokens[i]
                if token.tag in ["NNG", "NNP", "NNB", "NR", "NP", "SL", "SN"]:
                    keywords.append(token.form)
                elif i < len(tokens) - 1:
                    next_token = tokens[i+1]
                    if token.tag in ["XR", "NNG"] and next_token.tag == "XSN":
                        keywords.append(token.form + next_token.form)

            return list(set(keywords))
        except Exception as e:
            log.error(f"Keyword extraction failed: {e}")
            return []

        def extract_entities(self, text: str) -> Dict[str, Any]:
        """사용자 쿼리에서 구조화된 엔티티 추출"""
        entities = {
            "model": None,
            "year": None,
            "codes": [],
            "parts": [],
            "symptoms": []
        }
        
        # Kiwi가 없어도 기본적인 엔티티 추출은 정규표현식으로 수행
        import re
        
        # 연도 추출 (2024, 2025년 등)
        year_patterns = re.findall(r'(20[0-9]{2})년?', text)
        if year_patterns:
            entities["year"] = int(year_patterns[0])
        
        # 코드 추출 (INFO12, FCD35 등)
        code_patterns = re.findall(r'([A-Z]{2,5}[0-9]{1,3})', text.upper())
        entities["codes"] = code_patterns
        
        # 차종 추출 (간단한 매핑)
        car_models = ['싼타페', '투싼', '아반떼', '소나타', '그랜저', '팰리세이드']
        for model in car_models:
            if model in text:
                entities["model"] = model
                break
        
        # Kiwi가 있으면 추가 키워드 추출
        if self.kiwi:
            try:
                keywords = self.extract_keywords(text)
                # 기존 엔티티 매핑 로직 적용
                for word in keywords:
                    if word in self.entity_dict:
                        entities["codes"].append(self.entity_dict[word])
                    else:
                        entities["parts"].append(word)
            except Exception as e:
                log.warning(f"Kiwi-based entity extraction failed: {e}")
        
        return entities

    # 싱글톤 인스턴스

try:
    preprocessor = KiwiPreprocessor()
except Exception as e:
    log.error(f"Critical preprocessor failure: {e}")
    preprocessor = None