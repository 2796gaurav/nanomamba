#!/usr/bin/env python3
"""
Dry Run Test for NanoMamba-Edge
Comprehensive testing of all components without actual training or AWS costs
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, List
import json
import time
from datetime import datetime

# Import local modules
from config_loader import ConfigLoader
from sagemaker_training import NanoMambaTrainer
from metrics_visualization import MetricsVisualizer

class DryRunTester:
    """Comprehensive dry run testing for NanoMamba-Edge"""
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trainer = NanoMambaTrainer(config)
        self.visualizer = MetricsVisualizer(config)
        
        # Dry run results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "components_tested": [],
            "tests_passed": 0,
            "tests_failed": 0,
            "execution_time_seconds": 0,
            "dry_run_config": config.get_dry_run_config()
        }
    
    def test_configuration(self) -> bool:
        """Test configuration loading and validation"""
        test_name = "Configuration Test"
        self.logger.info(f"🧪 Running {test_name}...")
        
        try:
            # Test config validation
            if not self.config.validate_config():
                raise ValueError("Configuration validation failed")
            
            # Test all config sections
            hf_token = self.config.get_hf_token()
            aws_config = self.config.get_aws_config()
            model_config = self.config.get_model_config()
            training_config = self.config.get_training_config()
            dry_run_config = self.config.get_dry_run_config()
            
            # Verify critical values
            assert hf_token is not None, "HF_TOKEN is None"
            assert aws_config["region"] is not None, "AWS region is None"
            assert model_config["hidden_dim"] == 512, "Model config incorrect"
            assert training_config["micro_batch_size"] == 16, "Training config incorrect"
            
            self.logger.info(f"✅ {test_name} passed")
            self.results["components_tested"].append(test_name)
            self.results["tests_passed"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} failed: {e}")
            self.results["components_tested"].append(test_name)
            self.results["tests_failed"] += 1
            return False
    
    def test_data_preparation(self) -> bool:
        """Test data preparation pipeline"""
        test_name = "Data Preparation Test"
        self.logger.info(f"🧪 Running {test_name}...")
        
        try:
            # Test with synthetic data
            from datasets import Dataset
            
            # Create synthetic dataset
            synthetic_data = []
            for i in range(100):  # Small dataset for testing
                text = f"Test sentence {i} for NanoMamba-Edge dry run testing. " * 20
                synthetic_data.append({"text": text})
            
            dataset = Dataset.from_list(synthetic_data)
            
            # Test tokenization
            tokenized_dataset = self.trainer.tokenize_data(dataset)
            
            # Verify tokenized dataset
            assert len(tokenized_dataset) == 100, "Tokenized dataset size incorrect"
            sample = tokenized_dataset[0]
            assert sample["input_ids"].shape[0] <= 2048, "Sequence length too long"
            
            self.logger.info(f"✅ {test_name} passed - {len(tokenized_dataset)} samples tokenized")
            self.results["components_tested"].append(test_name)
            self.results["tests_passed"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} failed: {e}")
            self.results["components_tested"].append(test_name)
            self.results["tests_failed"] += 1
            return False
    
    def test_sagemaker_integration(self) -> bool:
        """Test SageMaker integration (dry run mode)"""
        test_name = "SageMaker Integration Test"
        self.logger.info(f"🧪 Running {test_name}...")
        
        try:
            # Test SageMaker session initialization
            assert self.trainer.sagemaker_session is not None, "SageMaker session not initialized"
            
            # Test dry run launch
            self.trainer.launch_sagemaker_job(dry_run=True)
            
            self.logger.info(f"✅ {test_name} passed")
            self.results["components_tested"].append(test_name)
            self.results["tests_passed"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} failed: {e}")
            self.results["components_tested"].append(test_name)
            self.results["tests_failed"] += 1
            return False
    
    def test_metrics_visualization(self) -> bool:
        """Test metrics visualization pipeline"""
        test_name = "Metrics Visualization Test"
        self.logger.info(f"🧪 Running {test_name}...")
        
        try:
            # Generate sample metrics
            metrics = self.visualizer.generate_sample_metrics()
            
            # Verify metrics structure
            assert "steps" in metrics, "Metrics missing steps"
            assert "loss" in metrics, "Metrics missing loss"
            assert len(metrics["steps"]) > 0, "No metric data"
            
            # Generate visualizations
            self.visualizer.generate_training_plots(metrics)
            
            # Check if plots were created
            plot_path = os.path.join(self.visualizer.visualization_dir, f"training_metrics.{self.visualizer.plot_format}")
            assert os.path.exists(plot_path), f"Training plot not created at {plot_path}"
            
            self.logger.info(f"✅ {test_name} passed")
            self.results["components_tested"].append(test_name)
            self.results["tests_passed"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} failed: {e}")
            self.results["components_tested"].append(test_name)
            self.results["tests_failed"] += 1
            return False
    
    def test_benchmark_generation(self) -> bool:
        """Test benchmark generation"""
        test_name = "Benchmark Generation Test"
        self.logger.info(f"🧪 Running {test_name}...")
        
        try:
            # Generate sample benchmarks
            benchmarks = self.visualizer.generate_sample_benchmarks()
            
            # Verify benchmarks structure
            assert "models" in benchmarks, "Benchmarks missing models"
            assert len(benchmarks["models"]) > 0, "No benchmark data"
            
            # Generate benchmark visualizations
            self.visualizer.generate_benchmark_comparison(benchmarks)
            
            # Check if benchmark plots were created
            mmlu_plot = os.path.join(self.visualizer.visualization_dir, f"benchmark_mmlu.{self.visualizer.plot_format}")
            assert os.path.exists(mmlu_plot), f"MMLU benchmark plot not created"
            
            self.logger.info(f"✅ {test_name} passed")
            self.results["components_tested"].append(test_name)
            self.results["tests_passed"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} failed: {e}")
            self.results["components_tested"].append(test_name)
            self.results["tests_failed"] += 1
            return False
    
    def test_end_to_end_workflow(self) -> bool:
        """Test complete end-to-end workflow"""
        test_name = "End-to-End Workflow Test"
        self.logger.info(f"🧪 Running {test_name}...")
        
        try:
            # Run all individual tests
            config_ok = self.test_configuration()
            data_ok = self.test_data_preparation()
            sagemaker_ok = self.test_sagemaker_integration()
            metrics_ok = self.test_metrics_visualization()
            benchmark_ok = self.test_benchmark_generation()
            
            # Check all tests passed
            all_passed = all([config_ok, data_ok, sagemaker_ok, metrics_ok, benchmark_ok])
            
            if all_passed:
                self.logger.info(f"✅ {test_name} passed - All components working together")
                self.results["components_tested"].append(test_name)
                self.results["tests_passed"] += 1
                return True
            else:
                raise ValueError("Some component tests failed")
            
        except Exception as e:
            self.logger.error(f"❌ {test_name} failed: {e}")
            self.results["components_tested"].append(test_name)
            self.results["tests_failed"] += 1
            return False
    
    def save_results(self) -> None:
        """Save dry run test results"""
        try:
            # Calculate execution time
            self.results["execution_time_seconds"] = int(time.time() - self.start_time)
            
            # Create results directory
            results_dir = self.config.get("DRY_RUN", "DRY_RUN_SAVE_PATH")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(results_dir, f"dry_run_results_{timestamp}.json")
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Also save a summary
            summary_file = os.path.join(results_dir, f"dry_run_summary_{timestamp}.txt")
            with open(summary_file, 'w') as f:
                f.write("NanoMamba-Edge Dry Run Test Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {self.results['timestamp']}\n")
                f.write(f"Execution Time: {self.results['execution_time_seconds']} seconds\n")
                f.write(f"Tests Passed: {self.results['tests_passed']}\n")
                f.write(f"Tests Failed: {self.results['tests_failed']}\n")
                f.write(f"Success Rate: {self.results['tests_passed'] / max(1, self.results['tests_passed'] + self.results['tests_failed']) * 100:.1f}%\n")
                f.write("\nComponents Tested:\n")
                for component in self.results['components_tested']:
                    f.write(f"  ✓ {component}\n")
            
            self.logger.info(f"📊 Dry run results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dry run results: {e}")
    
    def run_all_tests(self) -> bool:
        """Run all dry run tests"""
        self.logger.info("🚀 Starting comprehensive dry run testing...")
        self.start_time = time.time()
        
        try:
            # Run individual component tests
            self.test_configuration()
            self.test_data_preparation()
            self.test_sagemaker_integration()
            self.test_metrics_visualization()
            self.test_benchmark_generation()
            
            # Run end-to-end test
            self.test_end_to_end_workflow()
            
            # Save results
            self.save_results()
            
            # Summary
            total_tests = self.results["tests_passed"] + self.results["tests_failed"]
            success_rate = self.results["tests_passed"] / max(1, total_tests) * 100
            
            self.logger.info("=" * 60)
            self.logger.info("🏁 DRY RUN TEST SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 Total Tests: {total_tests}")
            self.logger.info(f"✅ Passed: {self.results['tests_passed']}")
            self.logger.info(f"❌ Failed: {self.results['tests_failed']}")
            self.logger.info(f"📈 Success Rate: {success_rate:.1f}%")
            self.logger.info(f"⏱️  Execution Time: {self.results['execution_time_seconds']} seconds")
            
            if self.results["tests_failed"] == 0:
                self.logger.info("🎉 ALL TESTS PASSED! System is ready for production use.")
                return True
            else:
                self.logger.warning("⚠️  Some tests failed. Please review the logs.")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Dry run testing failed: {e}")
            self.save_results()
            return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NanoMamba-Edge Dry Run Tester")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader(args.config)
    
    if not config.validate_config():
        sys.exit(1)
    
    # Initialize and run dry run tester
    tester = DryRunTester(config)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)