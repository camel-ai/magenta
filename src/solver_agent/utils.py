import os
import pandas as pd
from colorama import Fore, Style
import logging
import datetime

def get_problem_ids_from_csv(reasoning_csv, problem_csv):
    """
    Get problem IDs that exist in both reasoning_csv and problem_csv.
    
    Args:
        reasoning_csv (str): Path to the reasoning CSV file
        problem_csv (str): Path to the problem CSV file
        
    Returns:
        list: List of problem IDs that exist in both files
    """
    if not os.path.exists(reasoning_csv) or not os.path.exists(problem_csv):
        print(f"{Fore.RED}Error: One or both CSV files not found.{Style.RESET_ALL}")
        return []
    
    try:
        # Read the CSV files
        reasoning_df = pd.read_csv(reasoning_csv)
        problem_df = pd.read_csv(problem_csv)
        
        # Get problem IDs from both files
        reasoning_problem_ids = set(reasoning_df['problem_id'].unique())
        problem_ids = set(problem_df['problem_id'].unique())
        
        # Find common problem IDs
        common_ids = reasoning_problem_ids.intersection(problem_ids)
        
        print(f"{Fore.CYAN}Found {len(common_ids)} problems in both CSV files.{Style.RESET_ALL}")

        # then also extract the problem question and the solution here
        problem_df = problem_df[problem_df['problem_id'].isin(common_ids)]
        problem_df = problem_df[['problem_id', 'level','problem_text', 'category','standard_solution']]

        return list(common_ids), problem_df
    except Exception as e:
        print(f"{Fore.RED}Error reading CSV files: {e}{Style.RESET_ALL}")
        return []



def setup_logging(model_name, log_file_path=None):
    """
    Set up logging configuration with a file handler based on model name or custom log path.
    
    Args:
        model_name: Name of the model to use in the log directory
        log_file_path: Optional custom path for the log file. If provided, this will be used
                       instead of generating a path based on model_name.
    
    Returns:
        logger: Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger("math_solver")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Determine log file path
    if log_file_path:
        # Use the provided custom log file path
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = log_file_path
    elif model_name:
        # Create a log file based on model name
        log_dir = os.path.join("logs", model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamp for the log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    else:
        # No logging to file if neither is provided
        return logger
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    return logger

def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return "unknown"