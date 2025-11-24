"""
Semantic search implementation using embeddings
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import faiss
import pickle

from src.search.embeddings.embedding_model import EmbeddingModel
from src.utils.logger import log
from config.settings import settings


@dataclass
class SearchResult:
    """Search result container"""
    doc_id: str
    score: float
    content: Dict[str, Any]
    matched_text: str


class SemanticSearcher:
    """
    Semantic search using vector similarity
    Uses Faiss for efficient similarity search when Elasticsearch is not available
    """
    
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 use_elasticsearch: bool = False):
        """
        Initialize semantic searcher
        
        Args:
            embedding_model: Embedding model instance
            use_elasticsearch: Whether to use Elasticsearch (if available)
        """
        self.embedding_model = embedding_model or EmbeddingModel(model_type="ollama")
        self.use_elasticsearch = use_elasticsearch
        self.dimension = self.embedding_model.get_dimension()
        
        # For local search (when ES is not available)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        
        # Elasticsearch client (when available)
        self.es_client = None
        if use_elasticsearch:
            self._initialize_elasticsearch()
        
        # Index file paths
        self.index_dir = settings.EMBEDDINGS_DIR
        self.documents_file = self.index_dir / "documents.json"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.faiss_index_file = self.index_dir / "faiss.index"
        
        log.info(f"Initialized SemanticSearcher (dim={self.dimension}, ES={use_elasticsearch})")
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch client if available"""
        try:
            from src.search.elasticsearch.es_client import ElasticSearchClient
            self.es_client = ElasticSearchClient()
            log.success("Elasticsearch client initialized successfully")
        except Exception as e:
            log.warning(f"Failed to initialize Elasticsearch: {e}")
            log.info("Falling back to local search mode")
            self.use_elasticsearch = False
            self.es_client = None
    
    def build_index_from_documents(self, documents: List[Dict[str, Any]], 
                                  text_field: str = "verbatim_text",
                                  batch_size: int = 32):
        """
        Build search index from documents
        
        Args:
            documents: List of documents
            text_field: Field containing text to embed
            batch_size: Batch size for encoding
        """
        log.info(f"Building index for {len(documents)} documents...")
        
        self.documents = documents
        
        # Extract texts for embedding
        texts = []
        valid_docs = []
        
        for doc in documents:
            text = doc.get(text_field, "")
            if text:
                texts.append(text)
                valid_docs.append(doc)
        
        if not texts:
            log.warning("No valid texts found for indexing")
            return
        
        # Generate embeddings
        log.info(f"Generating embeddings for {len(texts)} texts...")
        self.embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
        self.documents = valid_docs
        
        # Build Faiss index
        self._build_faiss_index()
        
        # Save to disk
        self._save_index()
        
        log.success(f"Index built with {len(self.documents)} documents")
    
    def _build_faiss_index(self):
        """Build Faiss index from embeddings"""
        if self.embeddings is None or len(self.embeddings) == 0:
            log.warning("No embeddings to build index")
            return
        
        # Create Faiss index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        log.info(f"Faiss index built with {self.index.ntotal} vectors")
    
    def _save_index(self):
        """Save index to disk"""
        try:
            # Save documents
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            # Save Faiss index
            if self.index is not None:
                faiss.write_index(self.index, str(self.faiss_index_file))
            
            log.success(f"Index saved to {self.index_dir}")
            
        except Exception as e:
            log.error(f"Failed to save index: {e}")
    
    def load_index(self):
        """Load index from disk"""
        try:
            # Load documents
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                log.info(f"Loaded {len(self.documents)} documents")
            
            # Load embeddings
            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
                log.info(f"Loaded embeddings with shape {self.embeddings.shape}")
            
            # Load Faiss index
            if self.faiss_index_file.exists():
                self.index = faiss.read_index(str(self.faiss_index_file))
                log.info(f"Loaded Faiss index with {self.index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to load index: {e}")
            return False
    
    def search(self, 
              query: str,
              k: int = 10,
              filters: Optional[Dict[str, Any]] = None,
              search_type: str = "hybrid") -> List[SearchResult]:
        """
        Perform semantic search with multiple search types
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Optional filters (e.g., {"model_year": 2024})
            search_type: "text", "vector", or "hybrid"
        
        Returns:
            List of search results
        """
        # Try Elasticsearch first if available
        if self.use_elasticsearch and self.es_client:
            try:
                return self.search_with_elasticsearch(query, k, filters, search_type)
            except Exception as e:
                log.warning(f"Elasticsearch search failed: {e}")
                log.info("Falling back to local search")
        
        # Fallback to local search (vector-based)
        if self.index is None or self.index.ntotal == 0:
            log.warning("No index available. Please build or load index first.")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in Faiss
        distances, indices = self.index.search(query_embedding, min(k * 3, self.index.ntotal))
        
        # Convert to search results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Faiss returns -1 for missing results
                continue
            
            doc = self.documents[idx]
            
            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    if doc.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Calculate similarity score (convert L2 distance to similarity)
            # Similarity = 1 / (1 + distance)
            score = 1.0 / (1.0 + float(dist))
            
            result = SearchResult(
                doc_id=doc.get('verbatim_id', ''),
                score=score,
                content=doc,
                matched_text=doc.get('verbatim_text', '')
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_with_elasticsearch(self, 
                                 query: str,
                                 k: int = 10,
                                 filters: Optional[Dict[str, Any]] = None,
                                 search_type: str = "hybrid") -> List[SearchResult]:
        """
        Search using Elasticsearch with multiple search types
        
        Args:
            query: Search query
            k: Number of results
            filters: Optional filters
            search_type: "text", "vector", or "hybrid"
        
        Returns:
            List of search results
        """
        if not self.es_client:
            raise Exception("Elasticsearch client not available")
        
        try:
            if search_type == "vector":
                return self._vector_search_es(query, k, filters)
            elif search_type == "text":
                return self._text_search_es(query, k, filters)
            else:  # hybrid
                return self._hybrid_search_es(query, k, filters)
                
        except Exception as e:
            log.error(f"Elasticsearch search failed: {e}")
            log.info("Falling back to local search")
            return self.search(query, k, filters)
    
    def _text_search_es(self, query: str, k: int, filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Text-based search using multi_match"""
        es_query = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["problem", "verbatim_text"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "_source": True
        }
        
        # Add filters
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})
            
            es_query["query"] = {
                "bool": {
                    "must": [es_query["query"]],
                    "filter": filter_clauses
                }
            }
        
        response = self.es_client.search(es_query)
        
        results = []
        for hit in response['hits']['hits']:
            result = SearchResult(
                doc_id=hit['_id'],
                score=hit['_score'],
                content=hit['_source'],
                matched_text=hit['_source'].get('verbatim_text', '')
            )
            results.append(result)
        
        log.info(f"Text search returned {len(results)} results")
        return results
    
    def _vector_search_es(self, query: str, k: int, filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Vector-based search using BGE-M3 embeddings"""
        # Generate query vector
        log.info(f"Generating vector for query: '{query}'")
        query_vector = self.embedding_model.encode(query)
        
        # Build KNN query with proper num_candidates
        # num_candidates must be >= k, typically k * 2 to k * 10 for better quality
        num_candidates = max(k * 2, min(k * 10, 10000))  # At least k*2, max 10000
        
        es_query = {
            "size": k,
            "knn": {
                "field": "verbatim_vector",
                "query_vector": query_vector.tolist(),
                "k": k,
                "num_candidates": num_candidates
            },
            "_source": True
        }
        
        log.info(f"KNN search: k={k}, num_candidates={num_candidates}")
        
        # Add filters to KNN query
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})
            
            es_query["knn"]["filter"] = filter_clauses
        
        response = self.es_client.search(es_query)
        
        results = []
        for hit in response['hits']['hits']:
            result = SearchResult(
                doc_id=hit['_id'],
                score=hit['_score'],  # KNN similarity score
                content=hit['_source'],
                matched_text=hit['_source'].get('verbatim_text', '')
            )
            results.append(result)
        
        log.success(f"Vector search returned {len(results)} results")
        return results
    
    def _hybrid_search_es(self, query: str, k: int, filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Hybrid search combining text and vector search"""
        # Get results from both methods
        text_results = self._text_search_es(query, k//2 + 2, filters)
        vector_results = self._vector_search_es(query, k//2 + 2, filters)
        
        # Combine and deduplicate results
        combined_docs = {}
        
        # Add text results with weight 0.3
        for result in text_results:
            doc_id = result.doc_id
            combined_docs[doc_id] = {
                'result': result,
                'text_score': result.score * 0.3,
                'vector_score': 0.0
            }
        
        # Add vector results with weight 0.7
        for result in vector_results:
            doc_id = result.doc_id
            if doc_id in combined_docs:
                combined_docs[doc_id]['vector_score'] = result.score * 0.7
            else:
                combined_docs[doc_id] = {
                    'result': result,
                    'text_score': 0.0,
                    'vector_score': result.score * 0.7
                }
        
        # Calculate combined scores and sort
        final_results = []
        for doc_data in combined_docs.values():
            combined_score = doc_data['text_score'] + doc_data['vector_score']
            
            # Update the result with combined score
            result = doc_data['result']
            result.score = combined_score
            final_results.append(result)
        
        # Sort by combined score and return top k
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        log.success(f"Hybrid search returned {len(final_results[:k])} results")
        return final_results[:k]
    
    def find_similar(self, 
                    text: str,
                    k: int = 5,
                    exclude_self: bool = True) -> List[SearchResult]:
        """
        Find similar documents to given text
        
        Args:
            text: Text to find similar documents for
            k: Number of similar documents
            exclude_self: Whether to exclude exact matches
        
        Returns:
            List of similar documents
        """
        results = self.search(text, k + 1 if exclude_self else k)
        
        if exclude_self and results:
            # Remove exact match (usually first result)
            results = [r for r in results if r.matched_text != text][:k]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            "total_documents": len(self.documents) if self.documents else 0,
            "index_dimension": self.dimension,
            "index_type": "Elasticsearch" if self.use_elasticsearch else "Faiss",
        }
        
        if self.index:
            stats["indexed_vectors"] = self.index.ntotal
        
        if self.embeddings is not None:
            stats["embedding_shape"] = self.embeddings.shape
        
        return stats


def test_semantic_search():
    """Test semantic search functionality"""
    
    print("=" * 70)
    print("Testing Semantic Search")
    print("=" * 70)
    
    # Sample documents
    sample_docs = [
        {
            "verbatim_id": "001",
            "model": "Santa Fe",
            "model_year": 2024,
            "verbatim_text": "Tire slips on wet road conditions"
        },
        {
            "verbatim_id": "002", 
            "model": "Tucson",
            "model_year": 2024,
            "verbatim_text": "Exterior trim damaged during delivery"
        },
        {
            "verbatim_id": "003",
            "model": "Santa Fe",
            "model_year": 2023,
            "verbatim_text": "Navigation system shows incorrect location"
        }
    ]
    
    # Initialize searcher
    searcher = SemanticSearcher()
    
    # Build index
    print("\n1. Building index...")
    searcher.build_index_from_documents(sample_docs)
    print(f"   ✅ Index built")
    
    # Test search
    print("\n2. Testing search...")
    queries = [
        "tire problems",
        "damaged parts",
        "GPS issues"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = searcher.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result.score:.3f}")
            print(f"      Text: {result.matched_text}")
            print(f"      Model: {result.content.get('model')} ({result.content.get('model_year')})")
    
    # Test with filters
    print("\n3. Testing filtered search...")
    results = searcher.search("problems", k=5, filters={"model_year": 2024})
    print(f"   Found {len(results)} results for year 2024")
    
    # Get statistics
    print("\n4. Index statistics:")
    stats = searcher.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Semantic search test complete!")


if __name__ == "__main__":
    test_semantic_search()