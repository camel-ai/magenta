from colorama import Fore, Style, Back
from math_verify import parse, verify
import logging


class MathEvaluator:
    """
    A class for evaluating mathematical solutions and presenting results.
    
    This evaluator compares the solver's answer against expected solutions,
    provides formatted output of problems and solutions, and calculates
    accuracy statistics for multiple problems.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the MathEvaluator with an optional logger.
        
        Args:
            logger: Optional logger instance for logging evaluation results
        """
        self.logger = logging.getLogger("math_evaluator")
    
    def evaluate_solution(self, solver_answer, expected_solution):
        """
        Evaluate a solution by comparing it with the expected answer.
        
        This method parses both the expected and actual answers, verifies their
        equivalence, and presents the results with color-coded formatting.
        
        Args:
            solver_answer (str): The answer provided by the solver
            expected_solution (str): The expected solution
            
        Returns:
            tuple: (is_correct, feedback) where is_correct is a boolean and feedback is a string
        """
    

        expected_answer = parse("$" + expected_solution + "$")
        actual_answer = parse(solver_answer) if solver_answer else ""

        is_correct = verify(expected_answer, actual_answer)
        
       
        
        # Print answers
        self.logger.info(f"\n{Back.BLUE}{Fore.WHITE}Expected Answer: {expected_answer}{Style.RESET_ALL}")
        self.logger.info(f"{Back.WHITE}{Fore.BLACK}Solver's Answer: {actual_answer}{Style.RESET_ALL}")
        
        # Print result with color
        result_color = Fore.GREEN if is_correct else Fore.RED
        self.logger.info(f"{result_color}Result: {'✓ Correct' if is_correct else '✗ Incorrect'}{Style.RESET_ALL}")
        
        return is_correct

    def print_problem(self, problem_data, index=None):
        """
        Print problem details with formatted output.
        
        Displays the problem ID, question text, and full solution with
        color-coded sections for better readability.
        
        Args:
            problem_data (dict): Dictionary containing problem information
            index (int, optional): Test case index number
        """
        
        self.logger.info(f"\n{Back.BLUE}{Fore.WHITE}Problem ID: {problem_data['id']}{Style.RESET_ALL}")
        self.logger.info(f"{Back.WHITE}{Fore.BLACK}Problem: {problem_data['problem']}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.MAGENTA}Full Solution: {problem_data['solution']}{Style.RESET_ALL}")
        self.logger.info("-"*80)  # Separator before the solving process

    def print_final_results(self, correct, total):
        """
        Print final accuracy statistics for a set of problems.
        
        Calculates and displays the number of correct/incorrect answers
        and the overall accuracy percentage with color formatting.
        
        Args:
            correct (int): Number of correctly solved problems
            total (int): Total number of problems attempted
        """
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Log the final results
        if self.logger:
            self.logger.info(f"Final Results - Correct: {correct}, Incorrect: {total - correct}, Accuracy: {accuracy:.1f}%")
        
        self.logger.info("\n" + "="*80)  # Section separator
        self.logger.info(f"{Back.YELLOW}{Fore.BLACK}Final Results:{Style.RESET_ALL}")
        self.logger.info(f"{Fore.GREEN}Correct: {correct}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.RED}Incorrect: {total - correct}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.CYAN}Accuracy: {accuracy:.1f}%{Style.RESET_ALL}")
        self.logger.info("="*80)  # Section separator
