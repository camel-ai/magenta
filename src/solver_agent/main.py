from dotenv import load_dotenv

from colorama import init, Fore, Style, Back
import pandas as pd
import csv
from database import DatabaseManager

from math_loader import MathLoader, MathProblem
from evaluator import MathEvaluator
from prompt import *
from math_solver import MathSolver
from utils import get_problem_ids_from_csv, setup_logging, get_git_commit_hash



def main(
    args,
    level, 
    start_idx = 0,
    num: int = 10, 
    dataset: str = "intermediate_algebra",
    multi_step: bool = True, 
    sympy_toolkit: bool = False, 
    code_toolkit: bool = False,
    geometry_toolkit: bool = False,
    mode: str = "train",
):
    """
    Run math problem evaluation with specified difficulty level.
    
    Args:
        level: The difficulty level to filter problems by (default: "5")
        num: Number of problems to evaluate (default: 10)
        dataset: Math dataset to use (default: "intermediate_algebra")
        multi_step: Whether to use multi-step conversation (default: True)
        sympy_toolkit: Whether to use sympy toolkit (default: False)
        code_toolkit: Whether to use code toolkit (default: False)
        geometry_toolkit: Whether to use geometry toolkit (default: False)
    """
    
    # Set up logging
    logger = setup_logging(args.model, args.log_file if hasattr(args, 'log_file') else None)
    logger.info(args)

    logger.info(f"Starting evaluation with model: {args.model}")
    logger.info(f"Parameters: level={level}, num={num}, multi_step={multi_step}")
    logger.info(f"Toolkits: sympy={sympy_toolkit}, code={code_toolkit}, geometry={geometry_toolkit}")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize database only if logging is enabled
    db = None
    if args.log:
        db = DatabaseManager(base_path=args.base_path)
        # db.clean_all_tables()  # Start fresh. TODO: Remove later
        
        # Get git commit hash
        commit_hash = get_git_commit_hash()
        
        # Insert metadata at start of run
        metadata_id = db.insert_metadata(args, commit_hash)
    
    # Initialize solver and evaluator
    solver = MathSolver(
        model = args.model, 
        multi_step=multi_step, 
        sympy_toolkit=sympy_toolkit, 
        code_toolkit=code_toolkit, 
        geometry_toolkit=geometry_toolkit, 
        model_name=args.checkpoint_path if hasattr(args, 'checkpoint_path') else None,
        port=args.port,
        vllm_max_tokens=args.vllm_max_tokens,
        logger=logger
    )

    evaluator = MathEvaluator(logger=logger)
    
    # Load problems
    loader = MathLoader(mode=mode, logger=logger)
    
    # Filter problems by IDs from CSV files if requested
    if args.use_csv_problems:
        problem_ids, problem_df = get_problem_ids_from_csv(args.reasoning_path, args.problem_path)
        if problem_ids:
            logger.info(f"Loaded {len(problem_ids)} problem IDs from CSV file: {args.reasoning_path}")
            logger.info(f"Using {len(problem_ids)} problems from CSV files.")
            problems = [
                MathProblem(
                    problem_id=problem['problem_id'].split('_')[1], 
                    problem = problem['problem_text'],
                    solution = problem['standard_solution'],
                    level = problem['level'],
                    problem_type = problem['problem_id'].split('_')[0],
                ) for _,problem in problem_df.iterrows()
            ]
            problems = problems[:num]
        else:
            logger.warning(f"No matching problems found in CSV files. Using default problems.")
            problems = loader.get_problems_by_category_by_level_in_order(dataset, level, start_idx=start_idx, num=num)
    elif level is None:
        problems = loader.get_problems_by_category_in_order(dataset, start_idx=start_idx, num = num)
    else:
        problems = loader.get_problems_by_category_by_level_in_order(dataset, level, start_idx=start_idx, num=num)

    
    if not problems:
        logger.error(f"No problems found for level {level}")
        return
    
    # Track statistics
    correct = 0
    incorrect = 0
    skipped = 0
    
    if args.blacklist_problems is not None:
        # Default path if not provided
        blacklist_path = args.blacklist_problems or "experiment_results_combined_problem.csv"
        try:
            blacklist_df = pd.read_csv(blacklist_path, quoting=csv.QUOTE_ALL, encoding='utf-8', on_bad_lines='skip')
            problem_id_blacklist = set(blacklist_df['problem_id'].astype(str))
        except Exception as e:
            logger.warning(f"Could not load blacklist from {blacklist_path}: {e}")
            problem_id_blacklist = set()


    # Process each problem
    for i, problem in enumerate(problems, 1):
        logger.info(f"Processing problem {problem.id}")
        logger.info("\n" + "=" * 80)
        logger.info(f"Test Case {i} (Problem ID: {problem.id})")
        logger.info(f"Problem: {problem.problem}")
        logger.info(f"Full Solution: {problem.solution}")
        logger.info("-" * 80)
        
        # Check if problem has been correctly solved before
        if args.log and db.has_correct_solution(problem.id) and not args.use_csv_problems:
            logger.info(f"Skipping problem {problem.id} - already solved correctly")
            skipped += 1
            continue
        
        if args.blacklist_problems is not None:
            if str(problem.id) in problem_id_blacklist:
                logger.info(f"Skipping problem {problem.id} - in blacklist")
                skipped += 1
                continue

        logger.info("Solving...")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                solver_answer = solver.solve_math_problem(problem.problem)
                break  # Break out of the retry loop if successful
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Error solving problem {problem.id}: {e}. Maximum retries reached, skipping problem.")
                    break
                logger.warning(f"Error solving problem {problem.id}: {e}. Retry {retry_count}/{max_retries}...")
                solver_answer = None

        # Get tool usage from solver
        used_sympy, used_code, used_geometry = solver.get_tool_usage()
        
        # Get solver log
        solution_log, history = solver.get_solver_log()
        
        # Insert problem if not exists
        if args.log:
            db.insert_problem(
                problem.id,
                problem.problem,
                level,
                dataset,
                problem.solution,
            )
        
        # Check if answer is correct
        is_correct = evaluator.evaluate_solution(
            solver_answer,
            problem.solution
        )
        
        # Log the result
        if is_correct:
            logger.info(f"Problem {problem.id} - Result: CORRECT")
        else:
            logger.info(f"Problem {problem.id} - Result: INCORRECT")
        
        # Insert reasoning path, only insert the correct one
        if args.log and is_correct and (used_sympy or used_code or used_geometry):
            db.insert_reasoning_path(
                metadata_id,
                problem.id,
                history,
                used_sympy,
                used_code,
                used_geometry,
                is_correct
            )
        
        # Update statistics and print result
        if is_correct:
            correct += 1
        else:
            incorrect += 1
        
        # Print running statistics
        logger.info(f"\n{Back.WHITE}{Fore.BLACK}Running Statistics:{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}Correct: {correct}{Style.RESET_ALL}")
        logger.info(f"{Fore.RED}Incorrect: {incorrect}{Style.RESET_ALL}")
        if correct + incorrect > 0:
            accuracy = (correct / (correct + incorrect)) * 100
            logger.info(f"Accuracy: {accuracy:.1f}%")
        else:
            logger.warning("No problems solved yet - all were skipped")
        logger.info(f"Skipped: {skipped}")
        
        # Add a separator between problems
        logger.info("\n" + "=" * 80)
        
    # Print final results
    logger.info(f"\n{Back.WHITE}{Fore.BLACK}Final Results:{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}Correct: {correct}{Style.RESET_ALL}")
    logger.info(f"{Fore.RED}Incorrect: {incorrect}{Style.RESET_ALL}")
    if correct + incorrect > 0:
        accuracy = (correct / (correct + incorrect)) * 100
        logger.info(f"Final Results - Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:.1f}%")
    else:
        logger.warning("No problems were solved - all were skipped")
    logger.info(f"Skipped problems: {skipped}")
    
    results = {
        'correct': correct,
        'incorrect': incorrect,
        'skipped': skipped,
        'accuracy': accuracy if correct + incorrect > 0 else 0
    }
    
    if args.log:
        # Store results in the database
        db.store_results(results)
        logger.info("Results stored in database")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate math problems at a specific difficulty level.')
    parser.add_argument('--level', type=str, default=None, help='Difficulty level of problems (default: 5)')
    parser.add_argument('--start_idx', type=int, default=0, help= 'start_idx of the results (default:0)')
    parser.add_argument('--num', type=int, default=10, help='Number of problems to evaluate (default: 10)')
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model to use (default: gpt-4o-mini)')
    parser.add_argument('--checkpoint_path', type=str, help='Path to a checkpoint model to use for testing')
    parser.add_argument('--multi_step', action='store_true', help='Use multi-step conversation, does not support toolkit')
    parser.add_argument('--sympy_toolkit', action='store_true', help='Use sympy toolkit for evaluation')
    parser.add_argument('--code_toolkit', action='store_true', help='Use code toolkit for evaluation')
    parser.add_argument('--geometry_toolkit', action='store_true', help='Use geometry toolkit for evaluation')
    parser.add_argument('--dataset', type=str, default="intermediate_algebra", help="Math dataset for experiments")
    parser.add_argument('--log', action='store_true', help='Enable logging to the database')
    parser.add_argument('--use_csv_problems', action='store_true', help='Only use problems that exist in both experiment_results_reasoning.csv and experiment_results_problem.csv')
    parser.add_argument('--reasoning_path', type=str, default="experiment_results_reasoning.csv", help="Path to the CSV file containing the reasoning path")
    parser.add_argument('--problem_path', type=str, default="experiment_results_problem.csv", help="Path to the CSV file containing the problem")
    parser.add_argument('--mode', type=str, default="train", help="Mode for loading problems (default: train)")
    parser.add_argument('--port', type=int, default=8000, help="Port for VLLM server")
    parser.add_argument('--vllm_max_tokens', type=int, default=8000, help="Maximum tokens for vLLM model output")
    parser.add_argument('--base_path', type=str, default="experiment_results", help="Base path for CSV files (without extension)")
    parser.add_argument('--log-file', type=str, help="Custom path for the log file")
    parser.add_argument('--blacklist_problems', type=str, default=None, help="Path to the CSV file containing the blacklist of problems")

    args = parser.parse_args()


    main(
        args = args,
        level=args.level,
        start_idx=args.start_idx,
        num=args.num, 
        dataset = args.dataset, 
        multi_step = args.multi_step, 
        sympy_toolkit=args.sympy_toolkit, 
        code_toolkit=args.code_toolkit,
        geometry_toolkit=args.geometry_toolkit,
        mode=args.mode
    )