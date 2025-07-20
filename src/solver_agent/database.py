import pandas as pd
from datetime import datetime    
import os
import csv      
import logging

class DatabaseManager:
    def __init__(self, base_path='experiment_results'):
        """Initialize the CSV-based database manager.
        
        Args:
            base_path (str): Base path for CSV files (without extension)
        """
        self.base_path = base_path
        self.logger = logging.getLogger("database_manager")
        self.metadata_path = f"{base_path}_metadata.csv"
        self.problem_path = f"{base_path}_problem.csv"
        self.reasoning_path = f"{base_path}_reasoning.csv"
        self._create_tables()

    def clean_all_tables(self):
        """Clean all CSV files by recreating them."""
        self._create_tables()

    def _create_tables(self):
        """Create empty CSV files with headers if they don't exist."""
        # Metadata CSV
        if not os.path.exists(self.metadata_path):
            pd.DataFrame(columns=[
                'metadata_id', 'git_commit_version', 'date_run', 'model', 'level',
                'num_problems', 'multi_step', 'sympy_toolkit', 'code_toolkit', 'geometry_toolkit', 'dataset'
            ]).to_csv(self.metadata_path, index=False)

        # Problem CSV
        if not os.path.exists(self.problem_path):
            pd.DataFrame(columns=[
                'problem_id', 'problem_text', 'level', 'category', 'standard_solution'
            ]).to_csv(self.problem_path, index=False)

        # Reasoning Path CSV
        if not os.path.exists(self.reasoning_path):
            pd.DataFrame(columns=[
                'reasoning_path_id', 'metadata_id', 'problem_id', 'date_run',
                'solution_log', 'used_sympy', 'used_code_toolkit', 'used_geometry_toolkit', 'is_correct'
            ]).to_csv(self.reasoning_path, index=False)

    def _get_next_id(self, csv_path, id_column):
        """Get the next available ID for a given CSV file."""
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return 1
            return df[id_column].max() + 1
        except:
            return 1

    def insert_metadata(self, args, commit_hash: str):
        """Insert metadata into the CSV file."""
        df = pd.read_csv(self.metadata_path)
        metadata_id = self._get_next_id(self.metadata_path, 'metadata_id')
        
        new_row = pd.DataFrame([{
            'metadata_id': metadata_id,
            'git_commit_version': commit_hash,
            'date_run': datetime.now().isoformat(),
            'model': args.model,
            'level': args.level,
            'num_problems': args.num,
            'multi_step': 1 if args.multi_step else 0,
            'sympy_toolkit': 1 if args.sympy_toolkit else 0,
            'code_toolkit': 1 if args.code_toolkit else 0,
            'geometry_toolkit': 1 if args.geometry_toolkit else 0,
            'dataset': args.dataset
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.metadata_path, index=False)
        return metadata_id

    def format_problem(self, problem_text):
        """Format problem text for consistent extraction."""
        # Split text into lines and remove extra whitespace in each line
        lines = problem_text.splitlines()
        lines = [' '.join(line.split()) for line in lines]
        # Join lines with literal '\\n' to preserve explicit newline characters in CSV
        problem_text = '\\n'.join(lines)
        
        # Ensure consistent LaTeX formatting
        problem_text = problem_text.replace('\\[', '[').replace('\\]', ']')
        problem_text = problem_text.replace('\\(', '(').replace('\\)', ')')
        
        # Add proper punctuation if missing
        if not problem_text.endswith(('.', '!', '?')):
            problem_text += '.'
        
        # Escape double quotes
        problem_text = problem_text.replace('"', '""')
        return problem_text

    def insert_problem(self, problem_id: str, problem_text: str, level: str,
                      category: str, standard_solution: str):
        """Insert problem into the CSV file if it doesn't exist."""
        # df = pd.read_csv(self.problem_path)
        # Use more robust CSV parsing settings
        df = pd.read_csv(
            self.problem_path, 
            quoting=csv.QUOTE_ALL,
            encoding='utf-8',
            # on_bad_lines='skip'  # Skip bad lines instead of raising an error
        )
        if problem_id not in df['problem_id'].values:
            formatted_text = self.format_problem(problem_text)
            standard_solution = self.format_problem(standard_solution)
            new_row = pd.DataFrame([{
                'problem_id': problem_id,
                'problem_text': formatted_text,
                'level': level,
                'category': category,
                'standard_solution': standard_solution
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.problem_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _format_tool_call(self, tool_call, result=None):
        """Format a tool call into XML-style tags with its result."""
        if isinstance(tool_call, dict):
            if 'function' in tool_call:
                func = tool_call['function']
                name = func.get('name', '')
                args = func.get('arguments', '')
                tool_xml = f"<tool>\n  <tool_name>{name}</tool_name>\n  <args>{args}</args>"
                if result:
                    if name == "execute_code":
                        # For execute_code, only include "Execution Results:" and the actual results
                        result_lines = str(result).split('\n')
                        execution_results = []
                        found_results = False
                        for line in result_lines:
                            if found_results or line.strip().startswith("> Execution Results:"):
                                found_results = True
                                execution_results.append(line)
                        result = '\n'.join(execution_results)
                    result = str(result)
                    tool_xml += f"\n  <result>{result}</result>"
                else:
                    tool_xml += "\n  <result> No response </result>"
                tool_xml += "\n</tool>\n"
                return tool_xml

    def _format_solution_log(self, solution_log):
        """Convert solution log memory object into a formatted text with tool calls."""
        formatted_log = []
        current_tool_call = None
        
        for msg in solution_log:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # Handle assistant messages with tool calls
            if role == 'assistant' and 'tool_calls' in msg:
                for tc in msg['tool_calls']:
                    current_tool_call = tc
                    # Format the tool call without result first
                    formatted_log.append(self._format_tool_call(tc))
                    
            # Handle tool responses
            elif role == 'tool' and current_tool_call and msg.get('tool_call_id') == current_tool_call['id']:
                # Extract result from tool response
                try:
                    import json
                    result_dict = json.loads(content)
                    result = result_dict.get('result', '')
                    # If result is a list, join its elements
                    if isinstance(result, list):
                        result = ', '.join(str(x).strip('"""') for x in result)
                    # Clean up the result string
                    result = str(result).strip('"""')
                except:
                    result = content
                
                # Add result and close tool tag
                formatted_log[-1] = self._format_tool_call(current_tool_call, result)
                current_tool_call = None
                
            # Handle regular messages
            if role == 'assistant' and content:
                # Clean up LaTeX formatting
                formatted_log.append(f"<message>\n{content}\n</message>\n")
                
                # Handle tool calls in the message
                if 'tool_calls' in msg:
                    for tc in msg['tool_calls']:
                        current_tool_call = tc
                        formatted_log.append(self._format_tool_call(tc))
                    
        return '\n'.join(formatted_log)

    def insert_reasoning_path(self, metadata_id: int, problem_id: str,
                            solution_log, used_sympy: bool, used_code: bool,
                            used_geometry: bool, is_correct: bool):
        """Insert reasoning path into the CSV file."""
        if not os.path.exists(self.reasoning_path):
            self._create_tables()
            
        df = pd.read_csv(self.reasoning_path, quoting=csv.QUOTE_NONNUMERIC)
        reasoning_path_id = self._get_next_id(self.reasoning_path, 'reasoning_path_id')
        
        formatted_log = self._format_solution_log(solution_log)
        # Only escape newlines and quotes, XML special chars are already escaped
        formatted_log = formatted_log.replace('\n', '\\n')
        
        new_row = pd.DataFrame([{
            'reasoning_path_id': reasoning_path_id,
            'metadata_id': metadata_id,
            'problem_id': problem_id,
            'date_run': datetime.now().isoformat(),
            'solution_log': formatted_log,
            'used_sympy': int(used_sympy),
            'used_code_toolkit': int(used_code),
            'used_geometry_toolkit': int(used_geometry),
            'is_correct': int(is_correct)
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        # Use double quotes and proper escaping for CSV
        df.to_csv(self.reasoning_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def has_attempted(self, problem_id: str) -> bool:
        df = pd.read_csv(self.problem_path)
        return any(df['problem_id'] == problem_id)

    def has_correct_solution(self, problem_id: str) -> bool:
        """Check if a problem has been correctly solved before."""
        df = pd.read_csv(self.reasoning_path)
        return any((df['problem_id'] == problem_id) & (df['is_correct'] == 1))

    def get_accuracy_by_toolkit(self):
        """Get accuracy statistics grouped by toolkit usage."""
        df = pd.read_csv(self.reasoning_path)
        return df.groupby(['used_sympy', 'used_code_toolkit', 'used_geometry_toolkit']).agg({
            'is_correct': ['count', lambda x: x.mean() * 100]
        }).reset_index()

    def get_performance_over_time(self):
        """Get performance trends over time."""
        metadata_df = pd.read_csv(self.metadata_path)
        # Use proper quote handling when reading the CSV
        reasoning_df = pd.read_csv(self.reasoning_path, quoting=csv.QUOTE_NONNUMERIC)
        
        merged_df = pd.merge(metadata_df, reasoning_df, on='metadata_id')
        return merged_df.groupby(['date_run', 'model']).agg({
            'is_correct': ['count', lambda x: x.mean() * 100]
        }).reset_index()

    def print_last_run(self):
        """Print details of the last run from metadata."""
        df = pd.read_csv(self.metadata_path)
        if len(df) == 0:
            self.logger.info("No runs found in metadata.")
            return
        
        last_run = df.iloc[-1]
        self.logger.info("\nLast Run Details:")
        self.logger.info(f"Run ID: {last_run['metadata_id']}")
        self.logger.info(f"Date: {last_run['date_run']}")
        self.logger.info(f"Model: {last_run['model']}")
        self.logger.info(f"Level: {last_run['level']}")
        self.logger.info(f"Number of Problems: {last_run['num_problems']}")
        self.logger.info(f"Dataset: {last_run['dataset']}")
        self.logger.info(f"Multi-step: {'Yes' if last_run['multi_step'] else 'No'}")
        self.logger.info(f"Sympy Toolkit: {'Yes' if last_run['sympy_toolkit'] else 'No'}")
        self.logger.info(f"Code Toolkit: {'Yes' if last_run['code_toolkit'] else 'No'}")
        self.logger.info(f"Geometry Toolkit: {'Yes' if last_run['geometry_toolkit'] else 'No'}")

    def store_results(self, results):
        """Store the overall results of a run in a CSV file.
        
        Args:
            results (dict): Dictionary containing correct, incorrect, skipped counts and accuracy
        """
        results_path = f"{self.base_path}_summary.csv"
        
        # Create the file with headers if it doesn't exist
        if not os.path.exists(results_path):
            pd.DataFrame(columns=[
                'metadata_id', 'date', 'correct', 'incorrect', 'skipped', 'accuracy'
            ]).to_csv(results_path, index=False)
        
        # Get the latest metadata ID
        metadata_df = pd.read_csv(self.metadata_path)
        if len(metadata_df) == 0:
            metadata_id = 0
        else:
            metadata_id = metadata_df['metadata_id'].max()
        
        # Add the results
        df = pd.read_csv(results_path)
        new_row = pd.DataFrame([{
            'metadata_id': metadata_id,
            'date': datetime.now().isoformat(),
            'correct': results['correct'],
            'incorrect': results['incorrect'],
            'skipped': results['skipped'],
            'accuracy': results['accuracy']
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(results_path, index=False)
        if self.logger:
            self.logger.info(f"Results summary saved to {results_path}")
