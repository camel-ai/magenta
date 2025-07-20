#!/usr/bin/env python3
"""
Script to generate reformatted versions of enhanced solution logs.

This script:
1. Reads the input from enhanced_solution.csv and locates the problem_id and enhanced_solution_log for each id
2. Finds the corresponding problem_text from experiment_results_problem.csv
3. Creates a camel-ai ChatAgent using gpt4o-mini to rewrite the chain of thought without mentioning tools
4. Writes the generated solutions to a new CSV called enhanced_solution_reformat.csv
5. Supports parallel processing with job_id and total_jobs parameters
"""

import pandas as pd
import argparse
import tqdm
import logging
import sys
import os
import csv
import json
import glob

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from camel
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.messages import BaseMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from prompt import SYSTEM_MESSAGE, REFORMATTING_PROMPT_TEMPLATE
from agent_factory import create_gpt4o_mini_agent, create_gpt4_1_mini_agent, create_qwen_agent
from utils import save_dataframe_to_csv

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Generate reformatted versions of enhanced solution logs.')
    parser.add_argument('--enhanced_solution', type=str, default='enhanced_solutions.csv', 
                        help='Path to the enhanced_solutions.csv file')
    parser.add_argument('--problem_data', type=str, default=None, 
                        help='Path to the experiment_results_problem.csv file')
    parser.add_argument('--output', type=str, default='enhanced_solution_reformat.csv', 
                        help='Path to the output CSV file')
    parser.add_argument('--log', type=str, default='info', 
                        help='Logging level (debug, info, warning, error, critical)')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='Number of problems to process in each batch for testing purposes')
    parser.add_argument('--max_problems', type=int, default=None, 
                        help='Maximum number of problems to process (for testing)')
    parser.add_argument('--job_id', type=int, default=0,
                        help='Job ID for parallel processing (0-indexed)')
    parser.add_argument('--total_jobs', type=int, default=1,
                        help='Total number of jobs for parallel processing')
    return parser.parse_args()

def setup_logging(log_level: str) -> None:
    """
    Set up logging with the specified level.
    
    Args:
        log_level (str): Logging level (debug, info, warning, error, critical)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logger.setLevel(numeric_level)


def generate_reformatted_solution(
    agent: ChatAgent, 
    problem_text: str, 
    enhanced_solution_log: str
) -> str:
    """
    Generate a reformatted version of the enhanced solution log.
    
    Args:
        agent (ChatAgent): The GPT-4o-mini agent
        problem_text (str): The problem text
        enhanced_solution_log (str): The enhanced solution log
        
    Returns:
        str: The reformatted solution
    """
    # Reset the agent for a fresh conversation
    agent.reset()
    
    # Create the prompt for this problem
    prompt = REFORMATTING_PROMPT_TEMPLATE.format(
        problem_text=problem_text,
        enhanced_solution_log=enhanced_solution_log
    )
    
    # Get the agent's response
    try:
        response = agent.step(BaseMessage.make_user_message(role_name="user", content=prompt))
        reformatted_solution = response.msg.content
        return reformatted_solution
    except Exception as e:
        logger.error(f"Error generating reformatted solution: {e}")
        return ""

def process_batch(
    agent: ChatAgent,
    problem_data: dict,
    batch_df: pd.DataFrame
) -> list:
    """
    Process a batch of problems to generate reformatted solutions.
    
    Args:
        agent (ChatAgent): The GPT-4o-mini agent
        problem_data (dict): Dictionary mapping problem_id to problem_text
        batch_df (pd.DataFrame): DataFrame containing the batch of problems
        
    Returns:
        list: List of reformatted solutions
    """
    reformatted_solutions = []
    
    for _, row in batch_df.iterrows():
        problem_id = row['problem_id']
        enhanced_solution_log = row['enhanced_solution_log']
        
        # Skip processing if enhanced_solution_log is empty or NaN
        if pd.isna(enhanced_solution_log) or not enhanced_solution_log.strip():
            logger.warning(f"Empty or NaN enhanced_solution_log for problem {problem_id}, skipping")
            reformatted_solutions.append(enhanced_solution_log)
            continue
            
        # Get the problem text from the problem data
        if problem_id in problem_data:
            problem_text = problem_data[problem_id]
            
            # Generate the reformatted solution
            logger.info(f"Processing problem {problem_id}")
            reformatted_solution = generate_reformatted_solution(agent, problem_text, enhanced_solution_log)
            
            if reformatted_solution:
                logger.info(f"Successfully generated reformatted solution for problem {problem_id}")
            else:
                logger.warning(f"Failed to generate reformatted solution for problem {problem_id}")
                reformatted_solution = enhanced_solution_log  # Fallback to the original
                
            reformatted_solutions.append(reformatted_solution)
        else:
            logger.warning(f"Problem {problem_id} not found in problem data")
            reformatted_solutions.append(enhanced_solution_log)  # Fallback to the original
    
    return reformatted_solutions


def main() -> None:
    """
    Main function to execute the solution reformatting process.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log)
    
    # Load the enhanced solution data
    logger.info(f"Loading enhanced solution data from {args.enhanced_solution}")
    try:
        # Use more robust CSV parsing settings
        enhanced_solution_df = pd.read_csv(
            args.enhanced_solution, 
            quoting=csv.QUOTE_ALL,
            encoding='utf-8',
            on_bad_lines='skip'  # Skip bad lines instead of raising an error
        )
        logger.info(f"Successfully loaded enhanced solution data with {len(enhanced_solution_df)} rows")
    except Exception as e:
        logger.error(f"Error loading enhanced solution data: {e}")
        return
    
    # Load the problem data
    logger.info(f"Loading problem data from {args.problem_data}")
    if args.problem_data is not None:
        try:
            # Use more robust CSV parsing settings
            problem_df = pd.read_csv(
                args.problem_data,
                quoting=csv.QUOTE_ALL,
                encoding='utf-8',
                on_bad_lines='skip'  # Skip bad lines instead of raising an error
            )
            logger.info(f"Successfully loaded problem data with {len(problem_df)} rows")
        except Exception as e:
            logger.error(f"Error loading problem data: {e}")
            return
    else:
        # read problem data and compile it from merged.jsonl files
        BASE_PATH = "MATH/train"
        problem_df = pd.DataFrame()
        for merged_path in glob.glob(os.path.join(BASE_PATH, "*/merged.jsonl")):
            if "MISC" in merged_path: continue
            df = pd.read_json(
                merged_path,
                lines=True
            )
            df["problem_id"] = df["type"] + "_" + df["id"].astype(str)
            df["problem_text"] = df["problem"]
            problem_df = pd.concat([problem_df, df], ignore_index=True)
        logger.info(f"Successfully loaded problem data with {len(problem_df)} rows")
    
    # Create a dictionary mapping problem_id to problem_text
    problem_data = dict(zip(problem_df['problem_id'], problem_df['problem_text']))
    logger.info(f"Created problem data dictionary with {len(problem_data)} entries")
    
    # Limit the number of problems if specified
    if args.max_problems is not None:
        enhanced_solution_df = enhanced_solution_df.head(args.max_problems)
        logger.info(f"Limited to {args.max_problems} problems")
    
    # Calculate chunk size for parallel processing
    total_items = len(enhanced_solution_df)
    job_id = args.job_id
    total_jobs = args.total_jobs
    
    chunk_size = total_items // total_jobs
    if total_items % total_jobs > 0:
        chunk_size += 1
    
    start_idx = job_id * chunk_size
    end_idx = min((job_id + 1) * chunk_size, total_items)
    
    logger.info(f"Job {job_id+1}/{total_jobs}: Processing items {start_idx+1}-{end_idx} out of {total_items}")
    
    # Get the subset of data for this job
    job_df = enhanced_solution_df.iloc[start_idx:end_idx].copy()
    logger.info(f"Processing {len(job_df)} items in this job")
    
    # Create a modified output filename to include the job ID
    output_file = args.output
    if total_jobs > 1:
        base_name, ext = os.path.splitext(args.output)
        output_file = f"{base_name}_job{job_id}{ext}"
    logger.info(f"Output will be saved to {output_file}")
    
    # Create a copy of the DataFrame for the output
    output_df = job_df.copy()
    
    # Create the GPT-4o-mini agent
    if model == 'GPT-4o-mini':
        logger.info("Creating GPT-4o-mini reformat agent")
        agent = create_gpt4o_mini_agent(SYSTEM_MESSAGE)
    elif model == 'GPT-4-1-mini':
        logger.info("Creating GPT-4-1-mini reformat agent")
        agent = create_gpt4_1_mini_agent(SYSTEM_MESSAGE)
    elif model == 'Qwen/Qwen2.5-7B-Instruct':
        logger.info("Creating Qwen/Qwen2.5-7B-Instruct reformat agent")
        agent = create_qwen_agent(SYSTEM_MESSAGE)
    else:
        raise ValueError(f"Invalid model: {model}")
    
    # Process the problems in batches
    total_problems = len(job_df)
    batch_size = args.batch_size
    num_batches = (total_problems + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Processing {total_problems} problems in {num_batches} batches of size {batch_size}")
    
    for batch_idx in tqdm.tqdm(range(num_batches), desc="Processing batches"):
        # Get the current batch
        start_batch_idx = batch_idx * batch_size
        end_batch_idx = min(start_batch_idx + batch_size, total_problems)
        batch_df = job_df.iloc[start_batch_idx:end_batch_idx]
        
        # Process the batch
        reformatted_solutions = process_batch(agent, problem_data, batch_df)
        
        # Update the output DataFrame
        output_df.loc[batch_df.index, 'enhanced_solution_log_reformatted'] = reformatted_solutions
        
        # Save the intermediate results - with improved CSV handling
        save_dataframe_to_csv(output_df, output_file)
        logger.info(f"Saved intermediate results to {output_file} (batch {batch_idx+1}/{num_batches})")

        # Create the output file for the first problem in the first batch (for debugging)
        if batch_idx > 0 or job_id > 0: continue
        if len(batch_df) == 0: continue
        
        # Get the problem ID for the first item
        problem_id = batch_df['problem_id'].iloc[0] if 'problem_id' in batch_df.columns else "unknown"
        
        # Find the corresponding problem text
        problem_text = problem_data.get(problem_id, "Problem text not found")
        
    
    # Save the final results - with improved CSV handling
    save_dataframe_to_csv(output_df, output_file)
    logger.info(f"Saved final results to {output_file}")
    logger.info(f"Job {job_id+1}/{total_jobs} completed successfully")

if __name__ == "__main__":
    main()
