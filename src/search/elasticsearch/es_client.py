"""
Elasticsearch client and connection manager
"""
from typing import Optional, Dict, Any, List
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, RequestError
import time
from src.utils.logger import log
from config.settings import settings


class ElasticSearchClient:
    """Elasticsearch client wrapper with connection management"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize Elasticsearch client
        
        Args:
            host: ES host (default from settings)
            port: ES port (default from settings)
        """
        self.host = host or settings.ES_HOST
        self.port = port or settings.ES_PORT
        self.index_name = settings.ES_INDEX_NAME
        
        self.client: Optional[Elasticsearch] = None
        self._connect()
    
    def _connect(self, retry_count: int = 3, retry_delay: int = 5):
        """
        Establish connection to Elasticsearch with retry logic
        
        Args:
            retry_count: Number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        for attempt in range(retry_count):
            try:
                self.client = Elasticsearch(
                    [{'host': self.host, 'port': self.port, 'scheme': 'http'}],
                    verify_certs=False,
                    ssl_show_warn=False,
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                
                # Test connection
                if self.client.ping():
                    info = self.client.info()
                    log.success(f"Connected to Elasticsearch {info['version']['number']}")
                    return
                else:
                    raise ConnectionError("Failed to ping Elasticsearch")
                    
            except Exception as e:
                log.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    log.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    log.error("Failed to connect to Elasticsearch")
                    raise ConnectionError(f"Could not connect to Elasticsearch at {self.host}:{self.port}")
    
    def create_index(self, index_name: Optional[str] = None, force_recreate: bool = False) -> bool:
        """
        Create index with mapping for IQS verbatim data
        
        Args:
            index_name: Name of the index (default from settings)
            force_recreate: If True, delete existing index and recreate
        
        Returns:
            True if successful
        """
        index = index_name or self.index_name
        
        try:
            # Check if index exists
            if self.client.indices.exists(index=index):
                if force_recreate:
                    log.warning(f"Deleting existing index: {index}")
                    self.client.indices.delete(index=index)
                else:
                    log.info(f"Index {index} already exists")
                    return True
            
            # Define index mapping
            mapping = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "verbatim_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        # Document ID
                        "verbatim_id": {"type": "keyword"},
                        
                        # Core fields
                        "vin": {"type": "keyword"},
                        "model_year": {"type": "integer"},
                        "registration_date": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                        "ownership": {"type": "keyword"},
                        
                        # Vehicle information
                        "make": {"type": "keyword"},
                        "model": {"type": "keyword"},
                        "trim": {"type": "keyword"},
                        
                        # Problem information
                        "part": {"type": "keyword"},
                        "problem": {
                            "type": "text",
                            "analyzer": "verbatim_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        
                        # Main text for embedding (will add vector later)
                        "verbatim_text": {
                            "type": "text",
                            "analyzer": "verbatim_analyzer"
                        },
                        
                        # Vector field for semantic search (1024 dimensions for BGE-M3)
                        "verbatim_vector": {
                            "type": "dense_vector",
                            "dims": 1024,  # BGE-M3 차원
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # Metadata
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        },
                        
                        # Timestamps
                        "indexed_at": {"type": "date"}
                    }
                }
            }
            
            # Create index
            response = self.client.indices.create(index=index, body=mapping)
            
            if response.get('acknowledged'):
                log.success(f"Successfully created index: {index}")
                return True
            else:
                log.error(f"Failed to create index: {response}")
                return False
                
        except RequestError as e:
            error_type = getattr(e, 'error', str(e))
            if 'resource_already_exists_exception' in str(error_type):
                log.info(f"Index {index} already exists")
                return True
            else:
                log.error(f"Error creating index: {str(e)}")
                raise
        except Exception as e:
            log.error(f"Unexpected error creating index: {str(e)}")
            raise
    
    def index_document(self, document: Dict[str, Any], index_name: Optional[str] = None) -> bool:
        """
        Index a single document
        
        Args:
            document: Document to index
            index_name: Target index name
        
        Returns:
            True if successful
        """
        index = index_name or self.index_name
        
        try:
            response = self.client.index(
                index=index,
                id=document.get('verbatim_id'),
                body=document
            )
            return response['result'] in ['created', 'updated']
        except Exception as e:
            log.error(f"Error indexing document: {str(e)}")
            return False
    
    def bulk_index(self, documents: List[Dict[str, Any]], 
                   index_name: Optional[str] = None,
                   batch_size: int = 100) -> Dict[str, int]:
        """
        Bulk index multiple documents
        
        Args:
            documents: List of documents to index
            index_name: Target index name
            batch_size: Number of documents per batch
        
        Returns:
            Dictionary with success and failure counts
        """
        index = index_name or self.index_name
        
        # Prepare documents for bulk indexing
        actions = []
        for doc in documents:
            # Add current timestamp
            doc['indexed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            action = {
                "_index": index,
                "_id": doc.get('verbatim_id'),
                "_source": doc
            }
            actions.append(action)
        
        # Perform bulk indexing
        try:
            log.info(f"Starting bulk indexing of {len(documents)} documents...")
            
            success, failed = helpers.bulk(
                self.client,
                actions,
                chunk_size=batch_size,
                raise_on_error=False,
                raise_on_exception=False
            )
            
            log.success(f"Bulk indexing complete: {success} succeeded, {len(failed)} failed")
            
            if failed:
                for item in failed[:5]:  # Show first 5 failures
                    log.error(f"Failed item: {item}")
            
            return {"success": success, "failed": len(failed)}
            
        except Exception as e:
            log.error(f"Bulk indexing error: {str(e)}")
            raise
    
    def search(self, query: Dict[str, Any], index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute search query
        
        Args:
            query: Elasticsearch query
            index_name: Index to search
        
        Returns:
            Search results
        """
        index = index_name or self.index_name
        
        try:
            response = self.client.search(index=index, body=query)
            return response
        except Exception as e:
            log.error(f"Search error: {str(e)}")
            raise
    
    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """
        Delete an index
        
        Args:
            index_name: Index to delete
        
        Returns:
            True if successful
        """
        index = index_name or self.index_name
        
        try:
            if self.client.indices.exists(index=index):
                response = self.client.indices.delete(index=index)
                log.success(f"Deleted index: {index}")
                return response.get('acknowledged', False)
            else:
                log.warning(f"Index {index} does not exist")
                return True
        except Exception as e:
            log.error(f"Error deleting index: {str(e)}")
            return False
    
    def get_index_stats(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get index statistics
        
        Args:
            index_name: Index name
        
        Returns:
            Index statistics
        """
        index = index_name or self.index_name
        
        try:
            # Get index stats (ES 8.x compatible)
            stats = self.client.indices.stats(index=index)
            
            # Get document count
            count = self.client.count(index=index)
            
            # ES 8.x 응답 구조에 맞게 수정
            index_stats = stats.get('indices', {}).get(index, {})
            total_stats = index_stats.get('total', {})
            store_stats = total_stats.get('store', {})
            
            return {
                "index": index,
                "document_count": count.get('count', 0),
                "size_in_bytes": store_stats.get('size_in_bytes', 0),
                "size_human": store_stats.get('size', 'unknown')
            }
        except Exception as e:
            log.error(f"Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    def close(self):
        """Close the Elasticsearch connection"""
        if self.client:
            self.client.close()
            log.info("Elasticsearch connection closed")


def test_connection():
    """Test Elasticsearch connection"""
    try:
        client = ElasticSearchClient()
        
        # Test connection
        info = client.client.info()
        log.success(f"Elasticsearch version: {info['version']['number']}")
        
        # Get cluster health
        health = client.client.cluster.health()
        log.info(f"Cluster status: {health['status']}")
        
        return client
        
    except Exception as e:
        log.error(f"Connection test failed: {str(e)}")
        return None


if __name__ == "__main__":
    test_connection()