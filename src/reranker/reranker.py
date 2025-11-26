"""
Deep Learning based Reranker using BGE-M3
Replaces basic scoring with Cross-Encoder model
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# 먼저 로거 임포트
from src.utils.logger import log
from src.search.semantic_search import SearchResult
from config.settings import settings

# 그 다음 FlagEmbedding 시도
try:
    from FlagEmbedding import FlagReranker
    FLAGEMBEDDING_AVAILABLE = True
except Exception as e:
    log.warning(f"FlagEmbedding not available: {e}")
    FlagReranker = None
    FLAGEMBEDDING_AVAILABLE = False

@dataclass
class RerankScore:
    """Reranking score components"""
    deep_score: float      # FlagReranker 모델 점수
    recency_score: float   # 최신성 점수
    metadata_score: float  # 메타데이터 일치 점수
    total_score: float     # 최종 합산 점수
    explanation: str

class Reranker:
    """
    Rerank search results using Cross-Encoder model (FlagReranker)
    Combines Deep Learning score with Business Logic (Metadata/Recency)
    """
    def __init__(self):
        """Initialize Reranker model with fallback support"""
        self.model_path = getattr(settings, 'RERANKER_MODEL_PATH', 'BAAI/bge-reranker-v2-m3')
        self.use_fp16 = getattr(settings, 'RERANKER_USE_FP16', False)
        self.model = None
        self.use_fallback = False
        
        # FlagEmbedding 사용 가능한 경우 시도
        if FLAGEMBEDDING_AVAILABLE and FlagReranker:
            try:
                log.info(f"Loading FlagReranker model: {self.model_path} (fp16={self.use_fp16})...")
                self.model = FlagReranker(
                    self.model_path,
                    use_fp16=self.use_fp16
                )
                log.success("FlagReranker model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load FlagReranker model: {e}")
                log.warning("Falling back to basic text similarity reranker")
                self.use_fallback = True
        else:
            log.warning("FlagEmbedding not available, using fallback reranker")
            self.use_fallback = True
        
        # 폴백 리랭커 초기화
        if self.use_fallback:
            from src.reranker.fallback_reranker import FallbackReranker
            self.fallback_reranker = FallbackReranker()

        # 가중치 설정 (딥러닝 모델 점수를 가장 중요하게 반영)
        self.weights = {
            'deep_score': 0.7,   # 문맥 유사도 (핵심)
            'metadata': 0.2,     # 차종/부품 일치 여부
            'recency': 0.1       # 최신 데이터 우대
        }

    def rerank(self,
            results: List[SearchResult],
            query: str,
            query_plan: Optional[Dict[str, Any]] = None,
            top_k: int = 20) -> List[Tuple[SearchResult, RerankScore]]:
        """
        Execute reranking pipeline with fallback support
        """
        if not results:
            return []
        
        # 폴백 모드인 경우 폴백 리랭커 사용
        if self.use_fallback:
            return self.fallback_reranker.rerank(results, query, query_plan, top_k)

        # FlagReranker 사용
        return self._rerank_with_flag_embedding(results, query, query_plan, top_k)
    
    def _rerank_with_flag_embedding(self,
            results: List[SearchResult],
            query: str,
            query_plan: Optional[Dict[str, Any]] = None,
            top_k: int = 20) -> List[Tuple[SearchResult, RerankScore]]:
        """
        FlagEmbedding을 사용한 리랭킹
        """

        # 1. 딥러닝 모델 입력을 위한 Pair 생성 (Query, Document)
        pairs = []
        valid_indices = []

        for i, result in enumerate(results):
            # 매칭된 텍스트가 없으면 본문 전체 사용
            text = result.matched_text if result.matched_text else str(result.content.get('problem', ''))
            if text.strip():
                pairs.append([query, text])
                valid_indices.append(i)

        if not pairs:
            return []

        # 2. FlagReranker 점수 계산 (Batch processing)
        try:
            log.debug(f"Computing scores for {len(pairs)} pairs...")
            # normalize=True로 0~1 사이 점수 반환 유도
            deep_scores = self.model.compute_score(pairs, normalize=True)
            
            log.debug(f"Deep scores type: {type(deep_scores)}, value: {deep_scores}")

            # 결과가 하나일 때 float로 반환되는 경우 처리
            if isinstance(deep_scores, float):
                deep_scores = [deep_scores]
            elif isinstance(deep_scores, (list, tuple, np.ndarray)):
                # numpy array나 리스트인 경우 float로 변환
                deep_scores = [float(score) for score in deep_scores]
            else:
                log.warning(f"Unexpected deep_scores type: {type(deep_scores)}")
                deep_scores = [0.5] * len(pairs)  # 기본값으로 처리

        except Exception as e:
            log.error(f"FlagReranker computation failed: {e}")
            log.warning("Falling back to basic text similarity")
            # 폴백 모드로 전환
            self.use_fallback = True
            if not hasattr(self, 'fallback_reranker'):
                from src.reranker.fallback_reranker import FallbackReranker
                self.fallback_reranker = FallbackReranker()
            return self.fallback_reranker.rerank(results, query, query_plan, top_k)

        # 3. 비즈니스 로직 점수와 결합 (Ensemble)
        final_results = []

        for idx, original_idx in enumerate(valid_indices):
            result = results[original_idx]
            d_score = deep_scores[idx]
            
            log.debug(f"Processing result {idx}: d_score={d_score}, type={type(d_score)}")
            
            # 점수가 문자열인 경우 float로 변환
            try:
                if isinstance(d_score, str):
                    d_score = float(d_score)
                elif not isinstance(d_score, (int, float)):
                    log.warning(f"Converting unexpected score type: {type(d_score)}")
                    d_score = float(d_score)
            except (ValueError, TypeError) as e:
                log.error(f"Score conversion failed: {e}, using default 0.5")
                d_score = 0.5

            # 메타데이터/최신성 점수 계산 (기존 로직 유지)
            metadata_score = self._calculate_metadata_score(result.content, query_plan)
            recency_score = self._calculate_recency_score(result.content)

            # 최종 가중치 합산
            total_score = (
                self.weights['deep_score'] * float(d_score) +
                self.weights['metadata'] * float(metadata_score) +
                self.weights['recency'] * float(recency_score)
            )

            explanation = self._generate_explanation(d_score, metadata_score)

            score_obj = RerankScore(
                deep_score=d_score,
                recency_score=recency_score,
                metadata_score=metadata_score,
                total_score=total_score,
                explanation=explanation
            )

            final_results.append((result, score_obj))

        # 4. 점수순 정렬
        final_results.sort(key=lambda x: x[1].total_score, reverse=True)

        return final_results[:top_k]

    def _calculate_metadata_score(self, content: Dict[str, Any], query_plan: Optional[Dict[str, Any]]) -> float:
        """
        메타데이터(차종, 연식 등) 일치 여부 확인
        """
        if not query_plan or 'entities' not in query_plan:
            return 0.5

        entities = query_plan.get('entities', {})
        if not entities:
            return 0.5

        matches = 0
        total_checks = 0

        # 차종 (Model) 확인
        if entities.get('model'):
            total_checks += 1
            if str(content.get('model', '')).lower() == str(entities['model']).lower():
                matches += 1

        # 연식 (Year) 확인
        if entities.get('year'):
            total_checks += 1
            if str(content.get('model_year')) == str(entities['year']):
                matches += 1

        # 부품/증상 키워드 포함 여부 (Metadata field에서)
        if entities.get('parts'):
            total_checks += 1
            doc_text = str(content).lower()
            for part in entities['parts']:
                if part.lower() in doc_text:
                    matches += 1
                    break

        if total_checks == 0:
            return 0.5

        return matches / total_checks

    def _calculate_recency_score(self, content: Dict[str, Any]) -> float:
        """
        최신 데이터 우대 로직 (2024, 2025 등 최신 연식)
        """
        try:
            model_year = int(content.get('model_year', 0))
            current_year = datetime.now().year

            if model_year >= current_year:
                return 1.0
            elif model_year == current_year - 1:
                return 0.8
            elif model_year == current_year - 2:
                return 0.6
            else:
                return 0.4
        except:
            return 0.5

    def _generate_explanation(self, deep_score: float, metadata_score: float) -> str:
        parts = []
        if deep_score > 0.7:
            parts.append("High context match")
        elif deep_score < 0.3:
            parts.append("Low context match")

        if metadata_score > 0.8:
            parts.append("Exact metadata match")

        return ", ".join(parts) if parts else "Standard match"
