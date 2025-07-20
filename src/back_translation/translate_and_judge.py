#!/usr/bin/env python3
"""
Generate Chain of Thought (CoT) reasoning for mathematical problems.

This script enhances mathematical problem solutions by generating detailed
Chain of Thought reasoning for each tool call in the solution. It uses the
CAMEL framework to create VLLM Qwen/Qwen2.5-7B-Instruct agents that attempt to solve subproblems,
and verifies the solutions against ground truth using math_verify.
"""

import pandas as pd
import re
import json
import argparse
import tqdm
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Import from project modules
from utils import process_log, format_problem, extract_result_boxed, extract_docstring_from_function, setup_logging
from agent_factory import create_gpt4o_mini_agent, create_gpt4_1_mini_agent, create_qwen_agent

# Import from camel
from camel.agents import ChatAgent
from camel.messages import BaseMessage


from math_verify import parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from prompt import COT_PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE





def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Generate CoT reasoning for mathematical problems.')
    parser.add_argument('--input', type=str, default='experiment_results_reasoning_temp.csv', help='Path to the input CSV file containing solutions.')
    parser.add_argument('--model', type=str, default='GPT-4-1-mini', help='Model to use for generating CoT reasoning.')
    parser.add_argument('--output', type=str, default='enhanced_solution.csv', help='Path to the output CSV file for saving enhanced solutions.')
    parser.add_argument('--log', type=str, default='info', help='Logging level (debug, info, warning, error, critical)')
    parser.add_argument('--num', type=int, default=1500, help='Number of problems to process for testing purposes.')
    parser.add_argument('--job_id', type=int, default=0, help='Job ID for parallel processing (0-indexed)')
    parser.add_argument('--total_jobs', type=int, default=1, help='Total number of jobs for parallel processing')
    return parser.parse_args()



def generate_cot_for_tool_call(
    agent: ChatAgent, 
    model: str,
    problem: str, 
    docstring: str, 
    arguments: str, 
    ground_truth: str, 
    max_retries: int = 3
) -> Tuple[str, bool]:
    """
    Generate Chain of Thought reasoning for a tool call and verify the result.
    
    Args:
        agent (ChatAgent): The Qwen/Qwen2.5-7B-Instruct agent
        problem (str): The subproblem to solve
        ground_truth (str): The ground truth answer to verify against
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        Tuple[str, bool]: The generated CoT reasoning and a boolean indicating success
    """
    
    # Initialize statistics tracking
    statistics = defaultdict(lambda: {'total_attempts': 0, 'success_count': 0})

    # Create an agent for verifier
    if model == 'GPT-4o-mini':
        logger.info("Creating GPT-4o-mini verifier agent")
        verifier_agent = create_gpt4o_mini_agent(system_message = "You are a mathematical problem verifier that determines whether the generated answer is mathematically equivalent to the ground truth. First explain why (or why not) the generated answer is mathmatically equivalent to the ground truth. You DO NOT need to care about the CoT reasoning. Then clearly state True or False at the end, wrapped in \\boxed{{}}, i.e., either \\boxed{{True}} or \\boxed{{False}}.")
    elif model == 'GPT-4-1-mini':
        logger.info("Creating GPT-4-1-mini verifier agent")
        verifier_agent = create_gpt4_1_mini_agent(system_message = "You are a mathematical problem verifier that determines whether the generated answer is mathematically equivalent to the ground truth. First explain why (or why not) the generated answer is mathmatically equivalent to the ground truth. You DO NOT need to care about the CoT reasoning. Then clearly state True or False at the end, wrapped in \\boxed{{}}, i.e., either \\boxed{{True}} or \\boxed{{False}}.")
    elif model == 'Qwen/Qwen2.5-7B-Instruct':
        logger.info("Creating Qwen/Qwen2.5-7B-Instruct verifier agent")
        verifier_agent = create_qwen_agent(system_message = "You are a mathematical problem verifier that determines whether the generated answer is mathematically equivalent to the ground truth. First explain why (or why not) the generated answer is mathmatically equivalent to the ground truth. You DO NOT need to care about the CoT reasoning. Then clearly state True or False at the end, wrapped in \\boxed{{}}, i.e., either \\boxed{{True}} or \\boxed{{False}}.")
    else:
        raise ValueError(f"Invalid model: {model}")
    
    # Try to generate CoT and verify the answer
    for attempt in range(max_retries):
        agent.reset()  # Reset agent for a fresh attempt
        verifier_agent.reset()
        # Generate the prompt for this attempt
        prompt = COT_PROMPT_TEMPLATE.format(tool_name=problem, docstring=docstring, arguments=arguments)
        print(prompt)
        if attempt > 0:
            prompt += f"\n\nThis is attempt {attempt + 1} of {max_retries}. Please be more careful with your calculations and reasoning."
        
        # Get the agent's response
        try:
            response = agent.step(BaseMessage.make_user_message(role_name="user", content=prompt))
            cot_reasoning = response.msg.content
            print("CoT reasoning:", cot_reasoning)
            # Use \\boxed{} to extract the final answer
            final_answer = extract_result_boxed(cot_reasoning)
            print("Final answer:", final_answer)
            # Look for the final answer in the CoT reasoning
            # Typically, the answer would be in LaTeX format enclosed in $ or $$ delimiters
            try:
                # Try to parse the entire response
                
                verifier_prompt = JUDGE_PROMPT_TEMPLATE.format(ground_truth=ground_truth, final_answer=final_answer)

    
                # Create verifier agent response
                verifier_response = verifier_agent.step(
                    BaseMessage.make_user_message(role_name="user", content=verifier_prompt))
                print("Judge response:", verifier_response.msgs[0].content)
                is_correct = bool(extract_result_boxed(verifier_response.msgs[0].content))
                
                statistics[problem]['total_attempts'] += 1
                
                if is_correct:
                    statistics[problem]['success_count'] += 1
                    logger.info(f"Verification successful on attempt {attempt + 1} for tool: {problem}. Correct_answer: {ground_truth}, generated_answer: {parse(cot_reasoning)}")
                    return cot_reasoning, True
                else:
                    logger.info(f"Verification failed for tool: {problem}. Correct_answer: {ground_truth}, generated_answer: {parse(cot_reasoning)}")
            except Exception as e:
                logger.warning(f"Failed to parse or verify answer on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.warning(f"Agent failed to generate response on attempt {attempt + 1}: {e}")
    
    # If all attempts fail, return empty CoT and False
    logger.warning(f"All {max_retries} attempts failed to generate valid CoT")
    return "", False

def enhance_solution_with_cot(
    solution: str, 
    agent: ChatAgent,
    model: str
) -> str:
    """
    Enhance a solution by adding Chain of Thought reasoning for each tool call,
    ensuring duplicate tool calls are eliminated.
    
    Args:
        solution (str): The original solution
        agent (ChatAgent): The agent used for generation
        model (str): The model used for generation
        
    Returns:
        str: The enhanced solution with CoT reasoning
    """
    # Split the solution into components (messages and tools)
    components = re.split(r'(<tool>.*?</tool>|<message>.*?</message>)', solution, flags=re.DOTALL)
    
    # Process components to identify unique tool blocks
    processed_components = []
    seen_tool_blocks = {}
    statistics = defaultdict(lambda: {'total_attempts': 0, 'success_count': 0})
    
    for component in components:
        # Skip empty components
        if not component.strip():
            continue
            
        # If this is a tool block, process it for uniqueness
        if component.strip().startswith('<tool>'):
            # Extract key components
            tool_name_match = re.search(r'<tool_name>(.*?)</tool_name>', component)
            args_match = re.search(r'<args>(.*?)</args>', component)
            result_match = re.search(r'<result>(.*?)</result>', component)
            
            if tool_name_match and args_match and result_match:
                tool_name = tool_name_match.group(1).strip()
                args = args_match.group(1).strip()
                result = result_match.group(1).strip()
                
                # Create a normalized deduplication key
                # Clean up quotes and whitespace for reliable comparison
                clean_args = args.replace('""""', '"').replace('""', '"').strip()
                clean_result = result.strip()
                dedup_key = f"{tool_name}|{clean_args}|{clean_result}"
                
                # If we've seen this tool block before, skip it entirely
                if dedup_key in seen_tool_blocks:
                    continue
                    
                seen_tool_blocks[dedup_key] = True
                
                # Check for error conditions in the tool output
                if any(error in result for error in ["stderr", "No response", "No output from the stdout"]):
                    # For error cases, we just add a simple error message
                    processed_components.append(component)
                    processed_components.append("\n\nError in the tool\n\n")
                    statistics[tool_name]['total_attempts'] += 1
                    continue
                
                # Process for execute_code tool
                if tool_name == "execute_code":
                    result = result[20:]  # Remove ``> Execution Results:``
                    result = result.replace("\\n", "")
                
                # Create a problem description for CoT generation
                sympy_functions = json.load(open("/slurm-storage/xinhua/toolcall-synthetic-datagen/src/back_translation/sympy_toolkit.json", "r"))

                problem = str(tool_name)
                docstring = extract_docstring_from_function(sympy_functions[tool_name])
                arguments = str(args)
                ground_truth = str(result)
                
                # Generate CoT reasoning
                cot_reasoning, success = generate_cot_for_tool_call(agent, model, problem, docstring, arguments, ground_truth)
                print("CoT reasoning:", cot_reasoning)
                # Always count an attempt
                statistics[tool_name]['total_attempts'] += 1
                
                # If CoT generation was successful, add it inside the tool block
                if success and cot_reasoning:
                    # Check if the tool block already has CoT reasoning (from input)
                    if not re.search(r'<cot>', component):
                        # Find the position to insert the CoT (before the closing </tool> tag)
                        tool_end_pos = component.rfind('</tool>')
                        if tool_end_pos > 0:
                            # Insert the CoT before the closing tag
                            formatted_cot = f"\n<cot>\nChain of Thought for {tool_name}:\n{cot_reasoning}\n</cot>\n"
                            modified_component = component[:tool_end_pos] + formatted_cot + component[tool_end_pos:]
                            processed_components.append(modified_component)
                        else:
                            # If we can't find the closing tag, append as is
                            processed_components.append(component)
                    else:
                        # If there's already CoT reasoning, use the original component
                        processed_components.append(component)
                    statistics[tool_name]['success_count'] += 1
                else:
                    # If CoT generation failed, add the original tool block
                    processed_components.append(component)
            else:
                # If we couldn't extract components, keep the original tool block
                processed_components.append(component)
        else:
            # For non-tool blocks (like messages), keep them as is
            processed_components.append(component)
    
    # Combine processed components back into a single solution
    enhanced_solution = ''.join(processed_components)
    
    # Calculate statistics
    for tool_name, stats in statistics.items():
        total_attempts = stats['total_attempts']
        success_count = stats['success_count']
        
        if total_attempts > 0:
            average_attempts = total_attempts / success_count if success_count > 0 else float('inf')
            success_percentage = (success_count / total_attempts) * 100
            logger.info(f"Tool: {tool_name}, Total Attempts: {total_attempts}, Successes: {success_count}, " +
                       f"Average Attempts: {average_attempts:.2f}, Success Percentage: {success_percentage:.2f}%")
        else:
            logger.info(f"Tool: {tool_name}, No attempts were made")
    
    return enhanced_solution

def main():
    """
    Main function to execute the CoT generation process.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log)
    
    # Load the input CSV
    logger.info(f"Loading input CSV from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows from input CSV")
    except Exception as e:
        logger.error(f"Failed to load input CSV: {e}")
        return
    
    # Create a Qwen/Qwen2.5-7B-Instruct agent
    if args.model == 'GPT-4o-mini':
        logger.info("Creating GPT-4o-mini agent")
        agent = create_gpt4o_mini_agent()
    elif args.model == 'GPT-4-1-mini':
        logger.info("Creating GPT-4-1-mini agent")
        agent = create_gpt4_1_mini_agent()
    elif args.model == 'Qwen/Qwen2.5-7B-Instruct':
        logger.info("Creating Qwen/Qwen2.5-7B-Instruct agent")
        agent = create_qwen_agent()
    else:
        logger.error(f"Unsupported model: {args.model}")
        return
        
    # Process each solution
    logger.info("Processing solutions")
    solutions = [solution.replace(r"\\n", "\n").replace("\\\\", "\\") for solution in df["solution_log"]]
    parsed_data_list = process_log(solutions)
    
    # Limit the number of problems to process for testing purposes
    if parsed_data_list:
        parsed_data_list = parsed_data_list[:args.num]
    
    # Calculate chunk for parallel processing
    total_items = len(parsed_data_list)
    job_id = args.job_id
    total_jobs = args.total_jobs
    
    # Calculate start and end indices for this job
    chunk_size = total_items // total_jobs
    start_idx = job_id * chunk_size
    end_idx = (job_id + 1) * chunk_size if job_id < total_jobs - 1 else total_items
    
    logger.info(f"Job {job_id+1}/{total_jobs}: Processing items {start_idx+1}-{end_idx} out of {total_items}")
    
    # Create a new column for enhanced solutions if it doesn't exist
    if "enhanced_solution_log" not in df.columns:
        df["enhanced_solution_log"] = df["solution_log"].copy()
    
    # Create a modified output filename to include the job ID
    output_file = args.output
    if total_jobs > 1:
        output_file = f"{args.output.rsplit('.', 1)[0]}_job{job_id}.{args.output.rsplit('.', 1)[1]}"
    
    # Enhance each solution with CoT reasoning for this job's chunk
    for i, (parsed_data, solution) in tqdm.tqdm(
        enumerate(zip(parsed_data_list[start_idx:end_idx], solutions[start_idx:end_idx])), 
        total=end_idx-start_idx
    ):
        logger.info(f"Enhancing solution {start_idx+i+1}/{total_items}")
        
        try:
            # Enhance the solution with CoT reasoning
            enhanced_solution = enhance_solution_with_cot(solution, agent, args.model)
            # Format the enhanced solution
            formatted_solution = format_problem(enhanced_solution)
            
            # Update the dataframe
            df.loc[start_idx+i, "enhanced_solution_log"] = formatted_solution
            
            logger.info(f"Successfully enhanced solution {start_idx+i+1}")
        except Exception as e:
            logger.error(f"Failed to enhance solution {start_idx+i+1}: {e}")
            # Keep the original solution in case of failure
            df.loc[start_idx+i, "enhanced_solution_log"] = df.loc[start_idx+i, "solution_log"]
    
    # Save the enhanced solutions to the output CSV
    logger.info(f"Saving enhanced solutions to {output_file}")
    try:
        # For parallel jobs, only save the rows that were processed by this job
        if total_jobs > 1:
            output_df = df.iloc[start_idx:end_idx].copy()
            output_df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)
        logger.info("Successfully saved enhanced solutions")
    except Exception as e:
        logger.error(f"Failed to save output CSV: {e}")

if __name__ == "__main__":
    main()
