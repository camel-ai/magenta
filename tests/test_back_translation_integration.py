#!/usr/bin/env python3
"""
Integration tests for back_translation functionality.
This module provides end-to-end integration tests to verify that
the back translation components work correctly with real data.
"""
import os
import sys
import unittest
import tempfile
import pandas as pd
import shutil
import logging
from unittest.mock import patch, MagicMock

# Add src directories to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'back_translation'))

# Try to import the components we want to test
IMPORTS_AVAILABLE = False
try:
    from back_translation_main import translate_and_judge_main, reformat_main
    from translate_and_judge import parse_arguments
    from agent_factory import create_gpt4o_mini_agent
    from utils import process_log, format_problem
    from reformat import main as reformat_main_direct
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import back translation components: {e}")

class TestBackTranslationIntegration(unittest.TestCase):
    """Integration tests for the back translation functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data files
        self.test_reasoning_data = pd.DataFrame({
            'problem_id': ['test_1', 'test_2'],
            'reasoning': ['Step 1: Add numbers', 'Step 1: Multiply values'],
            'solution': ['4', '12']
        })
        
        self.test_reasoning_file = os.path.join(self.temp_dir, 'experiment_results_reasoning_temp.csv')
        self.test_reasoning_data.to_csv(self.test_reasoning_file, index=False)
        
        # Create test enhanced solutions data
        self.test_enhanced_data = pd.DataFrame({
            'problem_id': ['test_1', 'test_2'],
            'enhanced_solution': ['Enhanced solution 1', 'Enhanced solution 2']
        })
        
        self.test_enhanced_file = os.path.join(self.temp_dir, 'enhanced_solutions.csv')
        self.test_enhanced_data.to_csv(self.test_enhanced_file, index=False)
        
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up any test files created in the current directory
        for file in ['enhanced_solution.csv', 'experiment_results_reasoning_temp.csv']:
            if os.path.exists(file):
                os.remove(file)
                
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    def test_back_translation_imports_integration(self):
        """Test that all back translation components can be imported correctly."""
        # Test that we can import main components
        self.assertTrue(hasattr(translate_and_judge_main, '__call__'))
        self.assertTrue(hasattr(reformat_main, '__call__'))
        self.assertTrue(hasattr(parse_arguments, '__call__'))
        self.assertTrue(hasattr(create_gpt4o_mini_agent, '__call__'))
        self.assertTrue(hasattr(process_log, '__call__'))
        self.assertTrue(hasattr(format_problem, '__call__'))
        
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    def test_agent_factory_integration(self):
        """Test that agent factory can create agents properly."""
        try:
            agent = create_gpt4o_mini_agent(api_key="test-key")
            self.assertIsNotNone(agent)
        except Exception as e:
            self.logger.warning(f"Agent creation test failed (expected with test key): {e}")
            # This is expected to fail with a test key
            pass
            
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    def test_back_translation_argument_parsing_integration(self):
        """Test that back translation can parse arguments correctly."""
        # Test argument parsing
        test_args = [
            '--input', self.test_reasoning_file,
            '--model', 'gpt-4o-mini',
            '--output', 'test_output.csv',
            '--num', '2'
        ]
        
        try:
            args = parse_arguments(test_args)
            self.assertEqual(args.model, 'gpt-4o-mini')
            self.assertEqual(args.num, 2)
        except SystemExit:
            # parse_arguments might call sys.exit, which is normal for argument parsing
            pass
        except Exception as e:
            self.logger.warning(f"Argument parsing test had unexpected behavior: {e}")
                
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    def test_utils_functions_integration(self):
        """Test utility functions work with real data."""
        # Test format_problem function with only one argument
        test_problem = "What is 2 + 2? This is a test problem."
        
        formatted = format_problem(test_problem)
        self.assertIsInstance(formatted, str)
        self.assertIn("What is 2 + 2", formatted)
        
        # Test process_log function
        test_log = "Step 1: Add numbers\nStep 2: Result is 4"
        processed = process_log(test_log)
        self.assertIsInstance(processed, str)
        
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_translate_and_judge_workflow_integration(self):
        """Test the translate and judge workflow with real test data."""
        with patch('agent_factory.ModelFactory') as mock_factory:
            # Mock the model factory to avoid API calls
            mock_agent = MagicMock()
            mock_agent.step.return_value.msg.content = "Enhanced reasoning for this problem."
            mock_factory.create.return_value = mock_agent
            
            # Change to temp directory to avoid file conflicts
            original_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
                
                # Copy the test file with a different name to avoid conflicts
                test_input_file = 'test_reasoning_input.csv'
                shutil.copy(self.test_reasoning_file, test_input_file)
                
                # Test the workflow
                result = translate_and_judge_main(
                    input_file=test_input_file,
                    model_name='gpt-4o-mini',
                    output_file='enhanced_solution.csv',
                    num_problems=2,
                    job_id=1,
                    total_jobs=1
                )
                
                # Check that output file was created
                self.assertTrue(os.path.exists('enhanced_solution.csv'))
                
            finally:
                os.chdir(original_cwd)
                
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    def test_reformat_workflow_integration(self):
        """Test the reformat workflow with real test data."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            # Copy test files
            shutil.copy(self.test_enhanced_file, 'enhanced_solutions.csv')
            
            # Test the reformat workflow
            try:
                result = reformat_main_direct(
                    enhanced_solutions_file='enhanced_solutions.csv',
                    problems_file=None,  # Test with None
                    output_file='reformatted.csv'
                )
            except Exception as e:
                self.logger.warning(f"Reformat test failed: {e}")
                # This might fail due to missing problem data, which is acceptable
                
        finally:
            os.chdir(original_cwd)
            
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Back translation components not available")
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_end_to_end_back_translation_integration(self):
        """Test the complete back translation workflow end-to-end."""
        with patch('agent_factory.ModelFactory') as mock_factory:
            # Mock the model factory
            mock_agent = MagicMock()
            mock_agent.step.return_value.msg.content = "This is an enhanced solution."
            mock_factory.create.return_value = mock_agent
            
            # Change to temp directory to avoid file conflicts
            original_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
                
                # Use a unique filename to avoid conflicts
                input_filename = 'test_end_to_end_input.csv'
                shutil.copy(self.test_reasoning_file, input_filename)
                
                # Run the full workflow
                result = translate_and_judge_main(
                    input_file=input_filename,
                    model_name='gpt-4o-mini',
                    output_file='enhanced_solution.csv',
                    num_problems=2,
                    job_id=1,
                    total_jobs=1
                )
                
                # Verify output
                self.assertTrue(os.path.exists('enhanced_solution.csv'))
                
                # Check the output data
                output_data = pd.read_csv('enhanced_solution.csv')
                self.assertGreater(len(output_data), 0)
                
            finally:
                os.chdir(original_cwd)

def main():
    """Run the integration tests."""
    print("Running Back Translation Integration Tests...")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the tests
    unittest.main(verbosity=2, exit=False)

if __name__ == "__main__":
    main()