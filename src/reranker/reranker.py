"""
Reranking module for improving search result relevance
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import re

from src.search.semantic_search import SearchResult
from src.utils.logger import log


@dataclass
class RerankScore:
    """Reranking score components"""
    semantic_score: float
    relevance_score: float
    recency_score: float
    metadata_score: float
    total_score: float
    explanation: str


class Reranker:
    """
    Rerank search results using multiple signals
    Combines semantic similarity with metadata and relevance features
    """
    
    def __init__(self, 
                 weight_semantic: float = 0.4,
                 weight_relevance: float = 0.3,
                 weight_recency: float = 0.1,
                 weight_metadata: float = 0.2):
        """
        Initialize reranker with score weights
        
        Args:
            weight_semantic: Weight for semantic similarity score
            weight_relevance: Weight for keyword relevance score
            weight_recency: Weight for recency score
            weight_metadata: Weight for metadata matching score
        """
        self.weights = {
            'semantic': weight_semantic,
            'relevance': weight_relevance,
            'recency': weight_recency,
            'metadata': weight_metadata
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        log.info(f"Initialized Reranker with weights: {self.weights}")
    
    def rerank(self, 
               results: List[SearchResult],
               query: str,
               query_plan: Optional[Dict[str, Any]] = None,
               top_k: int = 20) -> List[Tuple[SearchResult, RerankScore]]:
        """
        Rerank search results
        
        Args:
            results: Initial search results
            query: Original query
            query_plan: Query plan with entities and filters
            top_k: Number of top results to return
        
        Returns:
            List of (result, score) tuples sorted by relevance
        """
        if not results:
            return []
        
        # Calculate scores for each result
        scored_results = []
        
        for result in results:
            score = self._calculate_score(result, query, query_plan)
            scored_results.append((result, score))
        
        # Sort by total score (descending)
        scored_results.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Return top k results
        return scored_results[:top_k]
    
    def _calculate_score(self, 
                        result: SearchResult,
                        query: str,
                        query_plan: Optional[Dict[str, Any]] = None) -> RerankScore:
        """
        Calculate comprehensive reranking score
        
        Args:
            result: Search result
            query: Original query
            query_plan: Query plan with extracted entities
        
        Returns:
            RerankScore with component scores
        """
        # 1. Semantic similarity score (from initial search)
        semantic_score = result.score
        
        # 2. Keyword relevance score
        relevance_score = self._calculate_relevance_score(
            result.matched_text, 
            query
        )
        
        # 3. Recency score
        recency_score = self._calculate_recency_score(
            result.content
        )
        
        # 4. Metadata matching score
        metadata_score = self._calculate_metadata_score(
            result.content,
            query_plan
        )
        
        # Calculate weighted total
        total_score = (
            self.weights['semantic'] * semantic_score +
            self.weights['relevance'] * relevance_score +
            self.weights['recency'] * recency_score +
            self.weights['metadata'] * metadata_score
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            semantic_score, relevance_score, recency_score, metadata_score
        )
        
        return RerankScore(
            semantic_score=semantic_score,
            relevance_score=relevance_score,
            recency_score=recency_score,
            metadata_score=metadata_score,
            total_score=total_score,
            explanation=explanation
        )
    
    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """
        Calculate keyword relevance score
        
        Args:
            text: Document text
            query: Search query
        
        Returns:
            Relevance score [0, 1]
        """
        if not text or not query:
            return 0.0
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Extract query terms
        query_terms = set(query_lower.split())
        
        # Count exact matches
        exact_matches = sum(1 for term in query_terms if term in text_lower)
        
        # Count partial matches
        partial_matches = sum(
            1 for term in query_terms 
            if any(term in word for word in text_lower.split())
        )
        
        # Calculate score
        if len(query_terms) == 0:
            return 0.0
        
        exact_score = exact_matches / len(query_terms)
        partial_score = partial_matches / len(query_terms)
        
        # Weighted combination (exact matches worth more)
        score = 0.7 * exact_score + 0.3 * partial_score
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate recency score based on date
        
        Args:
            content: Document content
        
        Returns:
            Recency score [0, 1]
        """
        # Get registration date or model year
        date_str = content.get('registration_date', '')
        model_year = content.get('model_year', 0)
        
        if date_str:
            try:
                # Parse date
                if isinstance(date_str, str):
                    # Try different date formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
                        try:
                            doc_date = datetime.strptime(date_str, fmt)
                            break
                        except:
                            continue
                    else:
                        # If no format works, use model year
                        if model_year:
                            doc_date = datetime(model_year, 1, 1)
                        else:
                            return 0.5  # Default middle score
                else:
                    return 0.5
                
                # Calculate days since document
                days_old = (datetime.now() - doc_date).days
                
                # Score based on age (newer = higher score)
                # Documents < 30 days: score 1.0
                # Documents > 365 days: score 0.2
                if days_old < 30:
                    return 1.0
                elif days_old < 90:
                    return 0.8
                elif days_old < 180:
                    return 0.6
                elif days_old < 365:
                    return 0.4
                else:
                    return 0.2
                    
            except Exception as e:
                log.debug(f"Failed to parse date: {e}")
                return 0.5
        
        elif model_year:
            # Score based on model year
            current_year = datetime.now().year
            years_old = current_year - model_year
            
            if years_old <= 0:
                return 1.0
            elif years_old == 1:
                return 0.8
            elif years_old == 2:
                return 0.6
            elif years_old == 3:
                return 0.4
            else:
                return 0.2
        
        return 0.5  # Default middle score
    
    def _calculate_metadata_score(self, 
                                 content: Dict[str, Any],
                                 query_plan: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate metadata matching score
        
        Args:
            content: Document content
            query_plan: Query plan with extracted entities
        
        Returns:
            Metadata score [0, 1]
        """
        if not query_plan or 'entities' not in query_plan:
            return 0.5  # Default middle score
        
        entities = query_plan.get('entities', {})
        if not entities:
            return 0.5
        
        matches = 0
        total_checks = 0
        
        # Check model match
        if 'model' in entities:
            total_checks += 1
            if content.get('model', '').lower() == entities['model'].lower():
                matches += 1
        
        # Check year match
        if 'year' in entities:
            total_checks += 1
            if content.get('model_year') == entities['year']:
                matches += 1
        
        # Check part match
        if 'part' in entities:
            total_checks += 1
            doc_text = (content.get('problem', '') + ' ' + 
                       content.get('verbatim_text', '')).lower()
            if entities['part'].lower() in doc_text:
                matches += 1
        
        # Calculate score
        if total_checks == 0:
            return 0.5
        
        return matches / total_checks
    
    def _generate_explanation(self, 
                             semantic: float,
                             relevance: float,
                             recency: float,
                             metadata: float) -> str:
        """
        Generate explanation for reranking score
        
        Args:
            semantic: Semantic score
            relevance: Relevance score
            recency: Recency score
            metadata: Metadata score
        
        Returns:
            Human-readable explanation
        """
        explanation = []
        
        if semantic > 0.8:
            explanation.append("High semantic similarity")
        elif semantic > 0.5:
            explanation.append("Moderate semantic similarity")
        
        if relevance > 0.8:
            explanation.append("Strong keyword match")
        elif relevance > 0.5:
            explanation.append("Partial keyword match")
        
        if recency > 0.8:
            explanation.append("Very recent")
        elif recency < 0.3:
            explanation.append("Older document")
        
        if metadata > 0.8:
            explanation.append("Exact metadata match")
        elif metadata > 0.5:
            explanation.append("Partial metadata match")
        
        return "; ".join(explanation) if explanation else "Standard ranking"


def test_reranker():
    """Test reranker functionality"""
    
    print("=" * 70)
    print("Testing Reranker")
    print("=" * 70)
    
    # Create sample search results
    results = [
        SearchResult(
            doc_id="001",
            score=0.85,
            content={
                "model": "Santa Fe",
                "model_year": 2024,
                "registration_date": "2024-01-15",
                "problem": "Tire slips on wet roads"
            },
            matched_text="Tire slips on wet roads"
        ),
        SearchResult(
            doc_id="002",
            score=0.75,
            content={
                "model": "Tucson",
                "model_year": 2023,
                "registration_date": "2023-06-20",
                "problem": "Wheel alignment issues"
            },
            matched_text="Wheel alignment issues"
        ),
        SearchResult(
            doc_id="003",
            score=0.80,
            content={
                "model": "Santa Fe",
                "model_year": 2025,
                "registration_date": "2025-01-01",
                "problem": "Tire pressure warning"
            },
            matched_text="Tire pressure warning"
        )
    ]
    
    # Create query plan
    query = "tire problems Santa Fe 2024"
    query_plan = {
        'entities': {
            'model': 'Santa Fe',
            'year': 2024,
            'part': 'tire'
        }
    }
    
    # Initialize reranker
    reranker = Reranker()
    
    # Rerank results
    print(f"\nQuery: {query}")
    print("\nOriginal ranking:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.score:.3f} - {result.matched_text}")
    
    reranked = reranker.rerank(results, query, query_plan)
    
    print("\nReranked results:")
    for i, (result, score) in enumerate(reranked, 1):
        print(f"  {i}. Total: {score.total_score:.3f}")
        print(f"     Text: {result.matched_text}")
        print(f"     Components: S={score.semantic_score:.2f}, "
              f"R={score.relevance_score:.2f}, "
              f"T={score.recency_score:.2f}, "
              f"M={score.metadata_score:.2f}")
        print(f"     Explanation: {score.explanation}")
    
    print("\n" + "=" * 70)
    print("Reranker test complete!")


if __name__ == "__main__":
    test_reranker()