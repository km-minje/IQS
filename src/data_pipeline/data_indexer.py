"""
Index IQS verbatim data into Elasticsearch
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_pipeline.excel_loader import ExcelDataLoader
from src.search.elasticsearch.es_client import ElasticSearchClient
from src.utils.logger import log
from config.settings import settings


class DataIndexer:
    """Index verbatim data into Elasticsearch"""
    
    def __init__(self, excel_path: Optional[str] = None):
        """
        Initialize data indexer
        
        Args:
            excel_path: Path to Excel file
        """
        self.excel_loader = ExcelDataLoader(excel_path)
        self.es_client = ElasticSearchClient()
        self.documents: List[Dict[str, Any]] = []
    
    def prepare_data(self) -> List[Dict[str, Any]]:
        """
        Load and prepare data for indexing
        
        Returns:
            List of prepared documents
        """
        log.info("Preparing data for indexing...")
        
        # Load Excel data
        self.excel_loader.load_excel()
        
        # Validate schema
        if not self.excel_loader.validate_schema():
            raise ValueError("Schema validation failed")
        
        # Clean data
        self.excel_loader.clean_data()
        
        # Process for indexing
        self.documents = self.excel_loader.process_for_indexing()
        
        log.success(f"Prepared {len(self.documents)} documents for indexing")
        return self.documents
    
    def add_dummy_vectors(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add dummy vectors to documents for testing
        (Later will be replaced with actual embeddings)
        
        Args:
            documents: List of documents
        
        Returns:
            Documents with dummy vectors added
        """
        log.info("Adding dummy vectors for testing...")
        
        for doc in tqdm(documents, desc="Adding vectors"):
            # Create a dummy vector (will be replaced with real embeddings later)
            # Using random vectors for now to test the pipeline
            dummy_vector = np.random.randn(settings.ES_VECTOR_DIMS).tolist()
            doc['verbatim_vector'] = dummy_vector
        
        return documents
    
    def index_to_elasticsearch(self, force_recreate: bool = False) -> Dict[str, int]:
        """
        Index all documents to Elasticsearch
        
        Args:
            force_recreate: If True, recreate the index
        
        Returns:
            Indexing results
        """
        if not self.documents:
            raise ValueError("No documents to index. Call prepare_data() first")
        
        # Create or verify index
        log.info("Creating/verifying Elasticsearch index...")
        self.es_client.create_index(force_recreate=force_recreate)
        
        # Add dummy vectors for testing (will be replaced with real embeddings)
        documents_with_vectors = self.add_dummy_vectors(self.documents)
        
        # Bulk index documents
        log.info(f"Indexing {len(documents_with_vectors)} documents to Elasticsearch...")
        results = self.es_client.bulk_index(
            documents_with_vectors,
            batch_size=settings.BATCH_SIZE
        )
        
        # Get index statistics
        stats = self.es_client.get_index_stats()
        log.info(f"Index statistics: {stats}")
        
        return results
    
    def verify_indexing(self, sample_size: int = 5):
        """
        Verify indexed documents by sampling
        
        Args:
            sample_size: Number of documents to sample
        """
        log.info(f"Verifying indexed documents (sampling {sample_size})...")
        
        # Simple match_all query to get some documents
        query = {
            "size": sample_size,
            "query": {
                "match_all": {}
            }
        }
        
        results = self.es_client.search(query)
        
        log.info(f"Total documents in index: {results['hits']['total']['value']}")
        log.info(f"Sample documents:")
        
        for hit in results['hits']['hits']:
            doc = hit['_source']
            log.info(f"  - ID: {doc['verbatim_id']}")
            log.info(f"    Model: {doc['model']} ({doc['model_year']})")
            log.info(f"    Problem: {doc['problem'][:100]}...")
            log.info("")
    
    def run_full_pipeline(self, force_recreate: bool = False):
        """
        Run the full indexing pipeline
        
        Args:
            force_recreate: If True, recreate the index
        
        Returns:
            Indexing results
        """
        log.info("=" * 70)
        log.info("Starting full indexing pipeline")
        log.info("=" * 70)
        
        try:
            # 1. Prepare data
            self.prepare_data()
            
            # 2. Index to Elasticsearch
            results = self.index_to_elasticsearch(force_recreate=force_recreate)
            
            # 3. Verify indexing
            self.verify_indexing()
            
            log.success("=" * 70)
            log.success("Indexing pipeline completed successfully!")
            log.success(f"Indexed: {results['success']} documents")
            if results['failed'] > 0:
                log.warning(f"Failed: {results['failed']} documents")
            log.success("=" * 70)
            
            return results
            
        except Exception as e:
            log.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            # Close connections
            self.es_client.close()


def main():
    """Main function to run the indexer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Index IQS verbatim data to Elasticsearch')
    parser.add_argument('--recreate', action='store_true', 
                       help='Recreate the index (delete existing)')
    parser.add_argument('--excel', type=str, default=None,
                       help='Path to Excel file (overrides .env setting)')
    
    args = parser.parse_args()
    
    try:
        # Initialize indexer
        indexer = DataIndexer(excel_path=args.excel)
        
        # Run pipeline
        results = indexer.run_full_pipeline(force_recreate=args.recreate)
        
        return results
        
    except Exception as e:
        log.error(f"Indexing failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()
