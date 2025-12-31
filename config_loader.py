#!/usr/bin/env python3
"""
Configuration Loader for NanoMamba-Edge
Loads and validates configuration from config.ini file
"""

import configparser
import os
import json
import ast
from typing import Dict, Any, Optional
import logging

class ConfigLoader:
    """Loads and manages configuration for NanoMamba-Edge project"""
    
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Set up logging based on configuration"""
        log_dir = self.get("LOGGING", "LOG_DIR", "./logs")
        log_level = self.get("LOGGING", "LOG_LEVEL", "INFO")
        
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "nanomamba.log")),
                logging.StreamHandler()
            ]
        )
        
    def get(self, section: str, key: str, default: Optional[Any] = None) -> Any:
        """Get configuration value with type conversion"""
        try:
            value = self.config.get(section, key)
            
            # Try to convert to appropriate type
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            elif value.isdigit():
                return int(value)
            elif value.replace('.', '', 1).isdigit() and '.' in value:
                return float(value)
            elif value.startswith('[') and value.endswith(']'):
                # Parse list
                return ast.literal_eval(value)
            elif value.startswith('{') and value.endswith('}'):
                # Parse dict
                return ast.literal_eval(value)
            else:
                return value
                
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is not None:
                return default
            raise ValueError(f"Missing configuration: [{section}] {key}")
    
    def get_hf_token(self) -> str:
        """Get Hugging Face token"""
        return self.get("HUGGINGFACE", "HF_TOKEN")
    
    def get_aws_config(self) -> Dict[str, str]:
        """Get AWS configuration"""
        return {
            "region": self.get("AWS", "AWS_REGION"),
            "role": self.get("AWS", "AWS_SAGEMAKER_ROLE"),
            "instance_type": self.get("AWS", "SAGEMAKER_INSTANCE_TYPE"),
            "volume_size": self.get("AWS", "SAGEMAKER_VOLUME_SIZE")
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration"""
        return {
            "model_name": self.get("MODEL", "MODEL_NAME"),
            "hidden_dim": self.get("MODEL", "HIDDEN_DIM"),
            "num_layers": self.get("MODEL", "NUM_LAYERS"),
            "state_dim": self.get("MODEL", "STATE_DIM"),
            "num_heads_q": self.get("MODEL", "NUM_HEADS_Q"),
            "num_heads_kv": self.get("MODEL", "NUM_HEADS_KV"),
            "vocab_size": self.get("MODEL", "VOCAB_SIZE"),
            "max_seq_len": self.get("MODEL", "MAX_SEQ_LEN"),
            "attention_layers": self.get("MODEL", "ATTENTION_LAYERS"),
            "total_params": self.get("MODEL", "TOTAL_PARAMS")
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            "micro_batch_size": self.get("TRAINING", "MICRO_BATCH_SIZE"),
            "gradient_accumulation_steps": self.get("TRAINING", "GRADIENT_ACCUMULATION_STEPS"),
            "effective_batch_size": self.get("TRAINING", "EFFECTIVE_BATCH_SIZE"),
            "learning_rate": self.get("TRAINING", "LEARNING_RATE"),
            "max_steps": self.get("TRAINING", "MAX_STEPS"),
            "warmup_steps": self.get("TRAINING", "WARMUP_STEPS"),
            "save_steps": self.get("TRAINING", "SAVE_STEPS"),
            "eval_steps": self.get("TRAINING", "EVAL_STEPS")
        }
    
    def get_dry_run_config(self) -> Dict[str, Any]:
        """Get dry run configuration"""
        return {
            "enabled": self.get("DRY_RUN", "DRY_RUN_ENABLED"),
            "steps": self.get("DRY_RUN", "DRY_RUN_STEPS"),
            "batch_size": self.get("DRY_RUN", "DRY_RUN_BATCH_SIZE"),
            "seq_len": self.get("DRY_RUN", "DRY_RUN_SEQ_LEN"),
            "save_path": self.get("DRY_RUN", "DRY_RUN_SAVE_PATH")
        }
    
    def validate_config(self) -> bool:
        """Validate that all required configuration is present"""
        required_sections = [
            "HUGGINGFACE", "AWS", "MODEL", "QUANTIZATION", 
            "TRAINING", "DATA", "DISTILLATION", "DRY_RUN", 
            "VISUALIZATION", "LOGGING"
        ]
        
        for section in required_sections:
            if section not in self.config:
                logging.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate critical values
        hf_token = self.get_hf_token()
        if hf_token == "your_huggingface_token_here":
            logging.warning("HF_TOKEN is using default value. Please set your actual Hugging Face token.")
        
        return True
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        output_path = output_path or self.config_path
        with open(output_path, 'w') as f:
            self.config.write(f)
        logging.info(f"Configuration saved to {output_path}")

if __name__ == "__main__":
    # Test the configuration loader
    config = ConfigLoader()
    
    if config.validate_config():
        print("✅ Configuration loaded successfully!")
        print(f"Model: {config.get_model_config()['model_name']}")
        print(f"Total params: {config.get_model_config()['total_params']}")
        print(f"Training steps: {config.get_training_config()['max_steps']}")
        print(f"Dry run enabled: {config.get_dry_run_config()['enabled']}")
    else:
        print("❌ Configuration validation failed!")