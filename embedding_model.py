"""
Embedding model wrapper for text vectorization
Supports OpenAI and Sentence-BERT models
"""
import os
from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import pickle
import hashlib
from pathlib import Path

from src.utils.logger import log
from config.settings import settings


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI embedding model wrapper"""
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize OpenAI embedding model
        
        Args:
            model_name: Model name (default from settings)
            api_key: API key (default from settings)
        """
        self.model_name = model_name or settings.OPENAI_EMBEDDING_MODEL
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            log.success(f"Initialized OpenAI embedding model: {self.model_name}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode texts to embeddings using OpenAI API
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for API calls (None이면 settings에서 가져옴)
        
        Returns:
            Embedding array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 기본값을 settings에서 가져오기
        batch_size = batch_size if batch_size is not None else getattr(settings, 'OPENAI_EMBEDDING_BATCH_SIZE', 32)
        
        # Process in batches to avoid API limits
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with OpenAI"):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                log.error(f"OpenAI API error: {str(e)}")
                # Return zero vectors for failed batches
                zero_embeddings = [[0.0] * self.get_dimension()] * len(batch)
                all_embeddings.extend(zero_embeddings)
        
        return np.array(all_embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI model"""
        # settings에서 모델별 차원 설정을 가져오거나 기본값 사용
        dimensions = getattr(settings, 'OPENAI_EMBEDDING_DIMENSIONS', {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        })
        default_dimension = getattr(settings, 'OPENAI_EMBEDDING_DEFAULT_DIMENSION', 1536)
        return dimensions.get(self.model_name, default_dimension)


class SentenceBERTEmbedding(BaseEmbeddingModel):
    """Sentence-BERT embedding model wrapper"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Sentence-BERT model
        
        Args:
            model_name: HuggingFace model name (None이면 settings에서 가져옴)
        """
        # 기본 모델명을 settings에서 가져오기
        self.model_name = model_name or getattr(settings, 'SBERT_DEFAULT_MODEL', 'all-MiniLM-L6-v2')
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            log.success(f"Initialized Sentence-BERT model: {model_name}")
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode texts to embeddings using Sentence-BERT
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (None이면 settings에서 가져옴)
        
        Returns:
            Embedding array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 기본값을 settings에서 가져오기
        batch_size = batch_size if batch_size is not None else getattr(settings, 'SBERT_BATCH_SIZE', 32)
        show_progress = getattr(settings, 'SBERT_SHOW_PROGRESS', True)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class OllamaEmbedding(BaseEmbeddingModel):
    """Ollama embedding model wrapper for BGE-M3 and other models"""
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Ollama embedding model
        
        Args:
            model_name: Ollama model name (기본값: bge-m3)
            base_url: Ollama server URL (기본값: http://localhost:11434)
        """
        self.model_name = model_name or getattr(settings, 'OLLAMA_EMBEDDING_MODEL', 'bge-m3')
        self.base_url = base_url or getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Ollama 서버 연결 테스트
        self._test_connection()
        
        log.success(f"Initialized Ollama embedding model: {self.model_name} at {self.base_url}")
    
    def _test_connection(self):
        """Ollama 서버 연결 테스트"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                # 모델 이름 매칭 (태그 포함/불포함 모두 지원)
                model_found = False
                for available_model in available_models:
                    if (self.model_name == available_model or 
                        self.model_name in available_model or 
                        available_model.startswith(self.model_name + ":")):  
                        model_found = True
                        break
                
                if not model_found:
                    log.warning(f"Model {self.model_name} not found in Ollama. Available: {available_models}")
                    log.info(f"You may need to run: ollama pull {self.model_name}")
                else:
                    log.success(f"Ollama model {self.model_name} is available (found as {available_model})")
            else:
                raise ConnectionError(f"Ollama server responded with {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}: {e}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode texts to embeddings using Ollama API
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for API calls (기본값: settings에서 가져옴)
        
        Returns:
            Embedding array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        import requests
        
        all_embeddings = []
        batch_size = batch_size if batch_size is not None else getattr(settings, 'OLLAMA_BATCH_SIZE', 8)
        
        # 작은 배치로 처리 (Ollama는 단일 요청 처리가 일반적)
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding with Ollama {self.model_name}"):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                try:
                    # 빈 텍스트 처리: 기본값으로 대체
                    if not text or not text.strip():
                        log.warning(f"Empty text detected, using default placeholder")
                        text = "[EMPTY_TEXT]"
                    
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.model_name,
                            "prompt": text
                        },
                        timeout=getattr(settings, 'OLLAMA_TIMEOUT', 30)
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        embedding = result.get('embedding')
                        
                        # 빈 임베딩 처리
                        if embedding and len(embedding) == self.get_dimension():
                            all_embeddings.append(embedding)
                        elif not embedding or len(embedding) == 0:
                            log.warning(f"Empty embedding returned for text: '{text[:50]}...', using zero vector")
                            zero_embedding = [0.0] * self.get_dimension()
                            all_embeddings.append(zero_embedding)
                        else:
                            log.warning(f"Unexpected embedding dimension: {len(embedding)}, expected: {self.get_dimension()}")
                            # 차원 맞춤 또는 제로 벡터
                            if len(embedding) > self.get_dimension():
                                all_embeddings.append(embedding[:self.get_dimension()])
                            else:
                                padded = embedding + [0.0] * (self.get_dimension() - len(embedding))
                                all_embeddings.append(padded)
                    else:
                        log.error(f"Ollama API error: {response.status_code} - {response.text}")
                        zero_embedding = [0.0] * self.get_dimension()
                        all_embeddings.append(zero_embedding)
                        
                except Exception as e:
                    log.error(f"Ollama encoding error for text '{text[:50] if text else 'None'}...': {e}")
                    zero_embedding = [0.0] * self.get_dimension()
                    all_embeddings.append(zero_embedding)
        
        return np.array(all_embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension for Ollama model"""
        # BGE-M3와 일반적인 Ollama 모델의 차원 매핑
        model_dimensions = getattr(settings, 'OLLAMA_MODEL_DIMENSIONS', None)
        if model_dimensions is None:
            model_dimensions = {
                'bge-m3': 1024,
                'bge-large': 1024,
                'nomic-embed-text': 768,
                'mxbai-embed-large': 1024,
                'all-minilm': 384
            }
        
        default_dimension = getattr(settings, 'OLLAMA_DEFAULT_DIMENSION', 1024)
        
        # 모델명에서 태그 제거 후 매칭
        clean_model_name = self.model_name.split(':')[0] if ':' in self.model_name else self.model_name
        return model_dimensions.get(clean_model_name, default_dimension)


class EmbeddingCache:
    """Cache for storing and retrieving embeddings"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or settings.EMBEDDINGS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                log.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            log.error(f"Failed to save cache: {e}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.cache.get(key)
    
    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding for text"""
        key = hashlib.md5(text.encode()).hexdigest()
        self.cache[key] = embedding
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Get cached embeddings for batch of texts
        
        Returns:
            Dictionary mapping text to cached embedding (None if not cached)
        """
        result = {}
        for text in texts:
            result[text] = self.get(text)
        return result
    
    def set_batch(self, texts: List[str], embeddings: np.ndarray):
        """Cache embeddings for batch of texts"""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
        self._save_cache()


class EmbeddingModel:
    """
    Main embedding model class with caching support
    """
    
    def __init__(self, 
                 model_type: Optional[str] = None,  # "openai", "sbert", "mock"
                 model_name: Optional[str] = None,
                 use_cache: Optional[bool] = None):
        """
        Initialize embedding model with cache
        
        Args:
            model_type: Type of model to use
            model_name: Specific model name
            use_cache: Whether to use caching
        """
        # 기본값을 settings에서 가져오기
        model_type = model_type if model_type is not None else settings.EMBEDDING_MODEL_TYPE
        use_cache = use_cache if use_cache is not None else settings.EMBEDDING_USE_CACHE
        
        # Initialize base model
        if model_type == "openai":
            self.model = OpenAIEmbedding(model_name)
        elif model_type == "sbert":
            self.model = SentenceBERTEmbedding(model_name)
        elif model_type == "ollama":
            self.model = OllamaEmbedding(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: 'openai', 'sbert', 'ollama'")
        
        # Initialize cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = EmbeddingCache()
        else:
            self.cache = None
        
        self.model_type = model_type
        log.info(f"Initialized {model_type} embedding model (cache={'on' if use_cache else 'off'})")
    
    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode texts with caching support
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (None이면 기본값 사용)
        
        Returns:
            Embedding array
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Check cache
        if self.cache:
            # 더 안전한 캐싱 로직
            all_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # 각 텍스트별로 캐시 확인
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    all_embeddings.append(cached_embedding)
                else:
                    all_embeddings.append(None)  # 플레이스홀더
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # 캐시되지 않은 텍스트들 처리
            if uncached_texts:
                try:
                    # 기본 batch_size 설정
                    actual_batch_size = batch_size if batch_size is not None else settings.EMBEDDING_DEFAULT_BATCH_SIZE
                    new_embeddings = self.model.encode(uncached_texts, actual_batch_size)
                    
                    # 캐시에 저장
                    self.cache.set_batch(uncached_texts, new_embeddings)
                    
                    # 플레이스홀더를 실제 임베딩으로 교체
                    for i, (idx, embedding) in enumerate(zip(uncached_indices, new_embeddings)):
                        all_embeddings[idx] = embedding
                        
                except Exception as e:
                    log.error(f"Failed to encode uncached texts: {e}")
                    # 캐시만 사용하거나 폴백
                    all_embeddings = [emb for emb in all_embeddings if emb is not None]
                    if not all_embeddings:
                        # 캐시도 없으면 직접 인코딩 시도
                        actual_batch_size = batch_size if batch_size is not None else settings.EMBEDDING_DEFAULT_BATCH_SIZE
                        return self.model.encode(texts, actual_batch_size)
            
            embeddings = np.array(all_embeddings)
        else:
            # No cache, encode directly
            actual_batch_size = batch_size if batch_size is not None else settings.EMBEDDING_DEFAULT_BATCH_SIZE
            embeddings = self.model.encode(texts, actual_batch_size)
        
        if single_text:
            return embeddings[0]
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_dimension()


def test_embedding_models():
    """Test different embedding models"""
    
    test_texts = [
        "The tire slips when driving on wet roads",
        "Exterior molding damaged at delivery",
        "Navigation system is inaccurate"
    ]
    
    print("=" * 70)
    print("Testing Embedding Models")
    print("=" * 70)
    
    # Test Ollama model (BGE-M3)
    print("\n1. Testing Ollama BGE-M3 Model...")
    try:
        ollama_model = EmbeddingModel(model_type="ollama")
        ollama_embeddings = ollama_model.encode(test_texts[0])  # Test single text first
        print(f"   ✅ Ollama embeddings shape: {ollama_embeddings.shape}")
        print(f"   Sample embedding (first 5 dims): {ollama_embeddings[:5]}")
    except Exception as e:
        print(f"   ❌ Ollama test failed: {e}")
    
    # Test OpenAI if API key is available
    if settings.OPENAI_API_KEY:
        print("\n2. Testing OpenAI Model...")
        try:
            openai_model = EmbeddingModel(model_type="openai")
            openai_embeddings = openai_model.encode(test_texts[0])  # Test single text
            print(f"   ✅ OpenAI embedding shape: {openai_embeddings.shape}")
        except Exception as e:
            print(f"   ❌ OpenAI test failed: {e}")
    else:
        print("\n2. OpenAI Model - Skipped (no API key)")
    
    # Test cache with Ollama
    print("\n3. Testing Cache with Ollama...")
    try:
        cached_model = EmbeddingModel(model_type="ollama", use_cache=True)
        
        # First call - will compute
        _ = cached_model.encode(test_texts[0])
        
        # Second call - should use cache
        _ = cached_model.encode(test_texts[0])
        print("   ✅ Cache working with Ollama")
    except Exception as e:
        print(f"   ❌ Cache test failed: {e}")
    
    print("\n" + "=" * 70)
    print("Embedding models ready!")


if __name__ == "__main__":
    test_embedding_models()