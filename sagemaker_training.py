#!/usr/bin/env python3
"""
SageMaker Training Pipeline for NanoMamba-Edge
Implements the complete training workflow on AWS SageMaker
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.huggingface import HuggingFace

# Import local modules
from config_loader import ConfigLoader

class NanoMambaDataset(Dataset):
    """Custom dataset for NanoMamba-Edge training"""
    
    def __init__(self, tokenized_data, max_length: int = 2048):
        self.tokenized_data = tokenized_data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        
        # Truncate if necessary
        if len(item["input_ids"]) > self.max_length:
            item["input_ids"] = item["input_ids"][:self.max_length]
            item["attention_mask"] = item["attention_mask"][:self.max_length]
            item["labels"] = item["labels"][:self.max_length]
        
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

class NanoMambaTrainer:
    """Main training class for NanoMamba-Edge on SageMaker"""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.aws_config = config.get_aws_config()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize SageMaker session
        self._init_sagemaker()
        
        # Set up directories
        self._setup_directories()
    
    def _init_sagemaker(self) -> None:
        """Initialize SageMaker session"""
        try:
            self.sagemaker_session = sagemaker.Session()
            self.role = self.aws_config["role"]
            self.region = self.aws_config["region"]
            self.instance_type = self.aws_config["instance_type"]
            
            self.logger.info(f"SageMaker session initialized in region: {self.region}")
            self.logger.info(f"Using instance type: {self.instance_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SageMaker session: {e}")
            raise
    
    def _setup_directories(self) -> None:
        """Set up necessary directories"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
    def prepare_data(self) -> None:
        """Prepare training data from multiple sources"""
        self.logger.info("Preparing training data...")
        
        # Load and mix datasets according to configuration
        dataset_composition = self.config.get("DATA", "DATASET_COMPOSITION")
        
        datasets_to_mix = []
        
        # Load each dataset component
        for dataset_name, tokens in dataset_composition.items():
            try:
                if dataset_name == "fineweb_edu":
                    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train")
                elif dataset_name == "stack_v2_code":
                    dataset = load_dataset("bigcode/stack-v2", split="train")
                elif dataset_name == "openwebmath":
                    dataset = load_dataset("openwebmath/openwebmath", split="train")
                elif dataset_name == "ultrachat_200k":
                    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train")
                elif dataset_name == "mc4_multilingual":
                    dataset = load_dataset("mc4", "en", split="train")
                else:
                    self.logger.warning(f"Unknown dataset: {dataset_name}, skipping")
                    continue
                
                # Sample proportionally based on token count
                # For simplicity, we'll use a fixed sample size for each
                sample_size = min(10000, len(dataset))  # Sample 10k examples from each
                sampled_dataset = dataset.select(range(sample_size))
                datasets_to_mix.append(sampled_dataset)
                
                self.logger.info(f"Loaded {dataset_name}: {len(sampled_dataset)} samples")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        # Mix datasets
        if datasets_to_mix:
            mixed_dataset = concatenate_datasets(datasets_to_mix)
            mixed_dataset = mixed_dataset.shuffle(seed=42)
            
            # Save to disk
            mixed_dataset.save_to_disk("data/mixed_dataset")
            self.logger.info(f"Mixed dataset created with {len(mixed_dataset)} samples")
            
            return mixed_dataset
        else:
            raise ValueError("No datasets were successfully loaded")
    
    def tokenize_data(self, dataset) -> NanoMambaDataset:
        """Tokenize the dataset"""
        self.logger.info("Tokenizing data...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("HUGGINGFACE", "HF_MODEL_ID"),
            token=self.config.get_hf_token()
        )
        
        def tokenize_function(examples):
            # Tokenize with padding/truncation
            tokenized = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.model_config["max_seq_len"],
                return_tensors="pt"
            )
            
            # Set labels to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=4  # Use multiple processes
        )
        
        # Convert to PyTorch dataset
        return NanoMambaDataset(tokenized_dataset)
    
    def create_training_script(self) -> str:
        """Create the training script for SageMaker"""
        training_script = """
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
import argparse
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='NanoMamba-Edge-v3')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=24)
    parser.add_argument('--state-dim', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    
    # Load dataset
    train_dataset = torch.load(os.path.join(args.train_dir, 'train_dataset.pt'))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model (simplified for example)
    # In a real implementation, this would be the NanoMamba architecture
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for step in range(args.max_steps):
        batch = next(iter(train_dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if step % 100 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if step % args.save_steps == 0 or step == args.max_steps - 1:
            checkpoint_path = os.path.join(args.model_dir, f'checkpoint_{step}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step,
                'loss': loss.item()
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    logger.info(f"Training completed. Model saved to {args.model_dir}")

if __name__ == '__main__':
    train()
"""
        
        return training_script
    
    def launch_sagemaker_job(self, dry_run: bool = False) -> None:
        """Launch SageMaker training job"""
        self.logger.info("Launching SageMaker training job...")
        
        # Create training script
        training_script = self.create_training_script()
        training_script_path = "train.py"
        
        with open(training_script_path, 'w') as f:
            f.write(training_script)
        
        # Prepare data for SageMaker
        # In a real implementation, we would upload data to S3
        # For this example, we'll use a local dataset
        
        try:
            # Initialize Hugging Face estimator
            huggingface_estimator = HuggingFace(
                entry_point=training_script_path,
                source_dir='.',
                instance_type=self.instance_type,
                instance_count=1,
                role=self.role,
                transformers_version='4.26.0',
                pytorch_version='1.13.1',
                py_version='py39',
                hyperparameters={
                    'model-name': self.model_config['model_name'],
                    'hidden-dim': self.model_config['hidden_dim'],
                    'num-layers': self.model_config['num_layers'],
                    'state-dim': self.model_config['state_dim'],
                    'batch-size': self.training_config['micro_batch_size'],
                    'learning-rate': self.training_config['learning_rate'],
                    'max-steps': self.training_config['max_steps'],
                    'warmup-steps': self.training_config['warmup_steps'],
                    'save-steps': self.training_config['save_steps']
                }
            )
            
            if dry_run:
                self.logger.info("🏃 DRY RUN MODE: Not actually launching SageMaker job")
                self.logger.info(f"Would launch with config: {huggingface_estimator.hyperparameters}")
                return
            
            # Start training job
            huggingface_estimator.fit({'train': 's3://your-bucket/data/'})
            
            self.logger.info("SageMaker training job launched successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to launch SageMaker job: {e}")
            raise
    
    def run_dry_run(self) -> None:
        """Run a dry run test to validate the training setup"""
        self.logger.info("🚀 Starting dry run test...")
        
        dry_run_config = self.config.get_dry_run_config()
        
        try:
            # Prepare a small dataset for dry run
            self.logger.info("Preparing dry run dataset...")
            
            # Create a simple synthetic dataset
            synthetic_data = []
            for i in range(dry_run_config['batch_size'] * 10):  # 10 batches
                # Create synthetic text data
                text = f"This is a test sentence number {i} for NanoMamba-Edge dry run. " * 50
                synthetic_data.append({"text": text})
            
            # Convert to dataset
            from datasets import Dataset
            dry_run_dataset = Dataset.from_list(synthetic_data)
            
            # Tokenize
            tokenized_dataset = self.tokenize_data(dry_run_dataset)
            
            # Create data loader
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=dry_run_config['batch_size'],
                shuffle=True
            )
            
            self.logger.info(f"Dry run dataset created: {len(tokenized_dataset)} samples")
            
            # Test a few batches
            self.logger.info("Testing data loading...")
            batch_count = 0
            for batch in dataloader:
                self.logger.info(f"Batch {batch_count + 1}: input_ids shape {batch['input_ids'].shape}")
                batch_count += 1
                if batch_count >= 3:  # Test 3 batches
                    break
            
            self.logger.info("✅ Data loading test passed!")
            
            # Test SageMaker job launch (dry run mode)
            self.launch_sagemaker_job(dry_run=True)
            
            self.logger.info("✅ Dry run completed successfully!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dry run failed: {e}")
            return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NanoMamba-Edge SageMaker Training")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry run mode")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader(args.config)
    
    if not config.validate_config():
        sys.exit(1)
    
    # Initialize trainer
    trainer = NanoMambaTrainer(config)
    
    if args.dry_run:
        # Run dry run test
        success = trainer.run_dry_run()
        sys.exit(0 if success else 1)
    else:
        # Run full training pipeline
        try:
            # Prepare data
            dataset = trainer.prepare_data()
            
            # Tokenize data
            tokenized_dataset = trainer.tokenize_data(dataset)
            
            # Launch SageMaker job
            trainer.launch_sagemaker_job()
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            sys.exit(1)