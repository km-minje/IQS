"""
Global configuration settings for IQS Verbatim Search System
"""
from pathlib import Path
from typing import Optional
import tempfile
import os
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # 한글 경로 문제 해결을 위해 시스템 임시 디렉토리 사용
    EMBEDDINGS_DIR: Path = Path(tempfile.gettempdir()) / "iqs_embeddings"
    
    # Excel file settings (모든 경로는 .env에서 설정)
    EXCEL_FILE_PATH: Optional[str] = Field(
        default=None,
        description="Path to the main Excel file (required in .env)"
    )
    EXCEL_SHEET_NAME: str = Field(
        default=None,
        description="Excel sheet name to read (required in .env)"
    )
    
    # Elasticsearch settings
    ES_HOST: str = Field(
        default="localhost",
        description="Elasticsearch host"
    )
    ES_PORT: int = Field(
        default=9200,
        description="Elasticsearch port"
    )
    ES_INDEX_NAME: str = Field(
        default="iqs_verbatim",
        description="Main index name"
    )
    ES_VECTOR_DIMS: int = Field(
        default=1536,
        description="Embedding vector dimensions"
    )
    
    # OpenAI settings (선택사항 - 현대차 내부 환경에서는 사용 불가)
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key (not used in Hyundai internal environment)"
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default=None,
        description="OpenAI embedding model (not used in Hyundai internal environment)"
    )
    OPENAI_CHAT_MODEL: str = Field(
        default=None,
        description="OpenAI chat model (not used in Hyundai internal environment)"
    )
    
    # H-Chat settings (모든 설정은 .env에서 관리, 하드코딩 금지)
    HCHAT_API_KEY: Optional[str] = Field(
        default=None,
        description="H-Chat API key (required in .env)"
    )
    HCHAT_BASE_URL: str = Field(
        default=None,
        description="H-Chat API base URL (required in .env)"
    )
    HCHAT_MODEL: str = Field(
        default=None,
        description="H-Chat model name (required in .env)"
    )
    HCHAT_KEY_NAME: Optional[str] = Field(
        default=None,
        description="H-Chat API key name for management"
    )
    
    # Processing settings
    BATCH_SIZE: int = Field(
        default=100,
        description="Batch size for processing"
    )
    MAX_WORKERS: int = Field(
        default=4,
        description="Max parallel workers"
    )
    
    # Search settings
    SEARCH_TOP_K: int = Field(
        default=100,
        description="Number of candidates to retrieve"
    )
    RERANK_TOP_K: int = Field(
        default=20,
        description="Number of results after reranking"
    )
    
    # Agent settings (모든 설정은 .env에서 관리)
    AGENT_LLM_TYPE: str = Field(
        default="h-chat",
        description="Default LLM type for agent (h-chat, openai)"
    )
    AGENT_MAX_STEPS: int = Field(
        default=None,
        description="Maximum execution steps for agent (set in .env)"
    )
    AGENT_TIMEOUT: int = Field(
        default=None,
        description="Agent execution timeout in seconds (set in .env)"
    )
    
    # OpenAI LLM Settings (모든 설정은 .env에서 관리)
    OPENAI_TEMPERATURE: Optional[float] = Field(
        default=None,
        description="OpenAI temperature setting (set in .env)"
    )
    OPENAI_MAX_TOKENS: Optional[int] = Field(
        default=None,
        description="OpenAI max tokens setting (set in .env)"
    )
    OPENAI_SYSTEM_MESSAGE: Optional[str] = Field(
        default=None,
        description="OpenAI system message (set in .env)"
    )
    OPENAI_JSON_SUFFIX: Optional[str] = Field(
        default=None,
        description="OpenAI JSON request suffix (set in .env)"
    )
    OPENAI_JSON_TEMPERATURE: Optional[float] = Field(
        default=None,
        description="OpenAI JSON request temperature (set in .env)"
    )
    
    # H-Chat LLM Settings (모든 설정은 .env에서 관리)
    HCHAT_TEMPERATURE: Optional[float] = Field(
        default=None,
        description="H-Chat temperature setting (set in .env)"
    )
    HCHAT_TIMEOUT: Optional[int] = Field(
        default=None,
        description="H-Chat API timeout in seconds (set in .env)"
    )
    HCHAT_SYSTEM_MESSAGE: Optional[str] = Field(
        default=None,
        description="H-Chat system message (set in .env)"
    )
    HCHAT_JSON_SUFFIX: Optional[str] = Field(
        default=None,
        description="H-Chat JSON request suffix (set in .env)"
    )
    HCHAT_JSON_TEMPERATURE: Optional[float] = Field(
        default=None,
        description="H-Chat JSON request temperature (set in .env)"
    )
    
    # H-Chat Test Settings (모든 설정은 .env에서 관리)
    HCHAT_TEST_BASIC_PROMPT: Optional[str] = Field(
        default=None,
        description="H-Chat basic test prompt (set in .env)"
    )
    HCHAT_TEST_JSON_PROMPT: Optional[str] = Field(
        default=None,
        description="H-Chat JSON test prompt (set in .env)"
    )
    HCHAT_TEST_ANALYSIS_PROMPT: Optional[str] = Field(
        default=None,
        description="H-Chat analysis test prompt (set in .env)"
    )
    HCHAT_TEST_TEMPERATURE: Optional[float] = Field(
        default=None,
        description="H-Chat test temperature (set in .env)"
    )
    
    # Embedding Model Settings (모든 설정은 .env에서 관리)
    EMBEDDING_MODEL_TYPE: str = Field(
        default="ollama",
        description="Default embedding model type (ollama, sbert, openai)"
    )
    EMBEDDING_USE_CACHE: bool = Field(
        default=True,
        description="Whether to use embedding cache"
    )
    EMBEDDING_DEFAULT_BATCH_SIZE: int = Field(
        default=32,
        description="Default batch size for embeddings"
    )
    
    # OpenAI Embedding Settings (모든 설정은 .env에서 관리)
    OPENAI_EMBEDDING_BATCH_SIZE: Optional[int] = Field(
        default=None,
        description="OpenAI embedding batch size (set in .env)"
    )
    OPENAI_EMBEDDING_DIMENSIONS: Optional[dict] = Field(
        default=None,
        description="OpenAI model dimensions mapping (set in .env)"
    )
    OPENAI_EMBEDDING_DEFAULT_DIMENSION: Optional[int] = Field(
        default=None,
        description="OpenAI default embedding dimension (set in .env)"
    )
    
    # Sentence-BERT Settings
    SBERT_DEFAULT_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Default Sentence-BERT model"
    )
    SBERT_BATCH_SIZE: int = Field(
        default=32,
        description="Sentence-BERT batch size"
    )
    SBERT_SHOW_PROGRESS: bool = Field(
        default=True,
        description="Show progress bar for Sentence-BERT"
    )
    
    # Ollama Settings (모든 설정은 .env에서 관리)
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL (set in .env)"
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="bge-m3",
        description="Ollama embedding model name (set in .env)"
    )
    OLLAMA_BATCH_SIZE: int = Field(
        default=8,
        description="Ollama batch size (smaller due to API limitations)"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=30,
        description="Ollama API timeout in seconds"
    )
    OLLAMA_MODEL_DIMENSIONS: Optional[dict] = Field(
        default={
            'bge-m3': 1024,
            'bge-large': 1024,
            'nomic-embed-text': 768,
            'mxbai-embed-large': 1024,
            'all-minilm': 384
        },
        description="Ollama model dimensions mapping"
    )
    OLLAMA_DEFAULT_DIMENSION: int = Field(
        default=1024,
        description="Default dimension for unknown Ollama models"
    )
    
    # Elasticsearch Vector Settings
    ES_USE_ELASTICSEARCH: bool = Field(
        default=True,
        description="Use Elasticsearch for search when available"
    )
    
    # Logging (모든 설정은 .env에서 관리)
    LOG_LEVEL: str = Field(
        default=None,
        description="Logging level (set in .env)"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Log file path (set in .env)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Mock 관련 설정들 무시
    
    @validator('EMBEDDINGS_DIR')
    def ensure_safe_embeddings_path(cls, v):
        """한글 경로 문제를 방지하기 위한 안전한 경로 설정"""
        safe_path = Path(tempfile.gettempdir()) / "iqs_embeddings"
        safe_path.mkdir(parents=True, exist_ok=True)
        return safe_path
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAW_DATA_DIR.mkdir(exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()