"""
Excel data loader for IQS Verbatim data
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import hashlib
from src.utils.logger import log
from config.settings import settings


class ExcelDataLoader:
    """Load and validate Excel data containing verbatim feedback"""
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize Excel loader
        
        Args:
            file_path: Path to Excel file. If None, uses settings.EXCEL_FILE_PATH
        """
        self.file_path = file_path or settings.EXCEL_FILE_PATH
        if not self.file_path:
            raise ValueError("Excel file path must be provided either as argument or in settings")
        
        self.file_path = Path(self.file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.file_path}")
        
        self.df: Optional[pd.DataFrame] = None
        self.processed_data: List[Dict[str, Any]] = []
        
        log.info(f"Initialized ExcelDataLoader with file: {self.file_path}")
    
    def load_excel(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load Excel file into DataFrame
        
        Args:
            sheet_name: Sheet name to load. If None, uses settings.EXCEL_SHEET_NAME
        
        Returns:
            Loaded DataFrame
        """
        sheet = sheet_name or settings.EXCEL_SHEET_NAME
        
        try:
            log.info(f"Loading Excel file: {self.file_path}, sheet: {sheet}")
            self.df = pd.read_excel(
                self.file_path,
                sheet_name=sheet,
                engine='openpyxl'
            )
            log.success(f"Successfully loaded {len(self.df)} rows from Excel")
            
            # Display basic info
            self._display_data_info()
            
            return self.df
            
        except Exception as e:
            log.error(f"Failed to load Excel file: {str(e)}")
            raise
    
    def _display_data_info(self):
        """Display basic information about loaded data"""
        if self.df is None:
            return
        
        log.info("=" * 50)
        log.info("Data Overview:")
        log.info(f"  Total rows: {len(self.df)}")
        log.info(f"  Total columns: {len(self.df.columns)}")
        log.info(f"  Columns: {', '.join(self.df.columns.tolist())}")
        log.info(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        log.info("=" * 50)
    
    def validate_schema(self, required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate that required columns exist in the DataFrame
        
        Args:
            required_columns: List of required column names
        
        Returns:
            True if validation passes
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_excel() first")
        
        # Default required columns for IQS verbatim data
        if required_columns is None:
            required_columns = [
                'Model Year',
                'Make of Vehicle', 
                'Model of Vehicle',
                'Problem',
                'Verbatim Text'
            ]
        
        missing_columns = set(required_columns) - set(self.df.columns)
        
        if missing_columns:
            log.error(f"Missing required columns: {missing_columns}")
            return False
        
        log.success("Schema validation passed")
        return True
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_excel() first")
        
        log.info("Starting data cleaning...")
        initial_rows = len(self.df)
        
        # Create a copy to avoid modifying original
        cleaned_df = self.df.copy()
        
        # 1. Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        
        # 2. Strip whitespace from string columns
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in string_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            # Replace 'nan' string with empty string
            cleaned_df[col] = cleaned_df[col].replace('nan', '')
        
        # 3. Standardize year format
        if 'Model Year' in cleaned_df.columns:
            cleaned_df['Model Year'] = pd.to_numeric(
                cleaned_df['Model Year'], 
                errors='coerce'
            ).fillna(0).astype(int)
        
        # 4. Create unique ID for each record
        cleaned_df['verbatim_id'] = cleaned_df.apply(
            lambda row: self._generate_id(row),
            axis=1
        )
        
        # 5. Add processing timestamp
        cleaned_df['processed_at'] = datetime.now().isoformat()
        
        final_rows = len(cleaned_df)
        log.info(f"Data cleaning complete. Rows: {initial_rows} → {final_rows}")
        
        self.df = cleaned_df
        return cleaned_df
    
    def _generate_id(self, row: pd.Series) -> str:
        """
        Generate unique ID for a verbatim record
        
        Args:
            row: DataFrame row
        
        Returns:
            Unique ID string
        """
        # Combine key fields to create unique identifier
        id_components = [
            str(row.get('VIN', '')),
            str(row.get('Model Year', '')),
            str(row.get('Problem', '')),
            str(row.name)  # row index
        ]
        id_string = '_'.join(id_components)
        
        # Create hash for consistent ID
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def process_for_indexing(self) -> List[Dict[str, Any]]:
        """
        Process data into format ready for Elasticsearch indexing
        
        Returns:
            List of documents ready for indexing
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_excel() first")
        
        log.info("Processing data for indexing...")
        
        self.processed_data = []
        
        for idx, row in self.df.iterrows():
            # Create document with ALL columns for comprehensive search
            doc = {
                'verbatim_id': row.get('verbatim_id', ''),
                'row_index': idx,
                'processed_at': row.get('processed_at', ''),
                
                # Main searchable text fields
                'verbatim_text': str(row.get('Verbatim Text', '')),
                'problem': str(row.get('Problem', '')),
                'problems_indicated': str(row.get('Problems Indicated', '')),
            }
            
            # Add ALL original columns as individual fields
            for col in self.df.columns:
                if col not in ['verbatim_id', 'processed_at', 'row_index']:  # Avoid duplicates
                    # Clean column name for field name (replace spaces and special chars)
                    field_name = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
                    field_name = ''.join(c for c in field_name if c.isalnum() or c == '_')
                    
                    # Convert value to appropriate type
                    value = row.get(col, '')
                    if pd.isna(value) or value == 'nan':
                        doc[field_name] = ''
                    elif isinstance(value, (int, float)):
                        doc[field_name] = value
                    else:
                        doc[field_name] = str(value)
                        
            # Create searchable text by combining key text fields
            searchable_parts = []
            text_fields = ['Verbatim Text', 'Problem', 'Problems Indicated', 'Model of Vehicle', 'Category', 'Sub Category']
            for field in text_fields:
                if field in row and pd.notna(row[field]) and str(row[field]) != 'nan':
                    searchable_parts.append(str(row[field]))
            
            doc['_searchable_text'] = ' '.join(searchable_parts)
            
            # Include documents that have meaningful content
            has_content = False
            
            # Check if document has any meaningful text content
            if doc['verbatim_text'] and doc['verbatim_text'] not in ['', 'nan']:
                has_content = True
            elif doc['problem'] and doc['problem'] not in ['', 'nan']:
                has_content = True
            elif doc.get('_searchable_text', '').strip():
                has_content = True
                
            if has_content:
                self.processed_data.append(doc)
        
        log.success(f"Processed {len(self.processed_data)} documents for indexing")
        return self.processed_data
    
    def save_processed_data(self, output_path: Optional[Path] = None):
        """
        Save processed data to JSON for backup/debugging
        
        Args:
            output_path: Path to save processed data
        """
        if not self.processed_data:
            log.warning("No processed data to save")
            return
        
        if output_path is None:
            output_path = settings.PROCESSED_DATA_DIR / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        log.success(f"Saved processed data to: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data
        
        Returns:
            Dictionary with data statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_excel() first")
        
        stats = {
            'total_records': len(self.df),
            'unique_models': self.df['Model of Vehicle'].nunique() if 'Model of Vehicle' in self.df.columns else 0,
            'unique_years': self.df['Model Year'].nunique() if 'Model Year' in self.df.columns else 0,
            'unique_problems': self.df['Problem'].nunique() if 'Problem' in self.df.columns else 0,
            'unique_makes': self.df['Make of Vehicle'].nunique() if 'Make of Vehicle' in self.df.columns else 0,
            'unique_categories': self.df['Category'].nunique() if 'Category' in self.df.columns else 0,
        }
        
        # Year distribution
        if 'Model Year' in self.df.columns:
            year_dist = self.df['Model Year'].value_counts().head(10).to_dict()
            stats['top_years'] = year_dist
        
        # Model distribution
        if 'Model of Vehicle' in self.df.columns:
            model_dist = self.df['Model of Vehicle'].value_counts().head(10).to_dict()
            stats['top_models'] = model_dist
        
        # Problem distribution
        if 'Problem' in self.df.columns:
            problem_dist = self.df['Problem'].value_counts().head(10).to_dict()
            stats['top_problems'] = problem_dist
        
        return stats


def main():
    """Test the Excel loader with sample data"""
    
    # Initialize loader (make sure to set EXCEL_FILE_PATH in .env or pass directly)
    loader = ExcelDataLoader()
    
    # Load data
    df = loader.load_excel()
    
    # Validate schema
    loader.validate_schema()
    
    # Clean data
    loader.clean_data()
    
    # Process for indexing
    documents = loader.process_for_indexing()
    
    # Get statistics
    stats = loader.get_statistics()
    
    log.info("Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            log.info(f"  {key}:")
            for k, v in value.items():
                log.info(f"    {k}: {v}")
        else:
            log.info(f"  {key}: {value}")
    
    # Save processed data
    loader.save_processed_data()
    
    return loader


if __name__ == "__main__":
    main()