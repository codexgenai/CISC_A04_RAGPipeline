import json
import logging
from pathlib import Path
from typing import Dict, Any, List

class ConfigManager:
    """Manages configuration settings for the RAG pipeline."""
    
    def __init__(self, config_file: str):
        """Initialize the config manager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.config = self._load_config()
        
        # Validate required fields
        required_fields = [
            "raw_input_directory",
            "cleaned_text_directory",
            "embeddings_directory",
            "vectordb_directory",
            "collection_name",
            "embedding_model_name",
            "llm_model_name",
            "llm_api_url",
            "retriever_min_score_threshold",
            "log_level",
            "model_type",
            "prompt_template"
        ]
        
        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {missing_fields}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Dict containing configuration settings
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_file}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file: {self.config_file}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def get_directory_names(self) -> List[str]:
        """Get list of directory configuration keys.
        
        Returns:
            List of directory configuration keys
        """
        return [
            "raw_input_directory",
            "cleaned_text_directory",
            "embeddings_directory",
            "vectordb_directory"
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration settings.
        
        Returns:
            Dict containing all configuration settings
        """
        return self.config.copy()
