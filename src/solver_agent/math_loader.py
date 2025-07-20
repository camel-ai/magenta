import os
import json
import random
from typing import List, Dict, Optional
import logging

class MathProblem:
    def __init__(self, problem_id: str, problem: str, solution: str, level: Optional[str] = None, problem_type: Optional[str] = None):
        """
        Initialize a math problem.
        
        Args:
            problem_id: Unique identifier for the problem
            problem: The math question text
            solution: The solution text
            level: Optional difficulty level
            problem_type: Optional type of problem (e.g., "Intermediate Algebra")
        """
        self.type = problem_type or "MISC"
        self.id = self.type + "_" + problem_id
        self.problem = problem
        self.solution = solution
        self.level = level
        
    
    @classmethod
    def from_json_file(cls, file_path: str, problem_id: str) -> 'MathProblem':
        """Create a MathProblem instance from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            return cls(
                problem_id=problem_id,
                problem=data["problem"],
                solution=data["solution"],
                level=data.get("level"),
                problem_type=data.get("type")
            )

class MathLoader:
    def __init__(self, dataset_path: Optional[str] = None, mode = "train", logger=None):
        """
        Initialize the math problem loader.
        
        Args:
            dataset_path: Path to the dataset directory. If not provided, 
                        uses default path in the project.
            mode: Mode for data loading, either "train" or "test"
            logger: Logger instance for logging
        """
        if dataset_path is None:
            # Use default path
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"..",
                "MATH", mode
            )
        self.dataset_path = dataset_path
        self.mode = mode
        self.problem_files: Dict[str, List[str]] = {}
        self.logger = logger
        self._load_problem_files()
        
        if self.logger:
            self.logger.info(f"Using dataset path: {self.dataset_path}")
        
    def _load_problem_files(self) -> None:
        """Loads the problem files from the dataset directory."""
        dataset_mode_path = os.path.join(self.dataset_path, self.mode)
        if not os.path.exists(dataset_mode_path):
            if self.logger:
                self.logger.warning(f"Dataset path {dataset_mode_path} does not exist. Using dataset path.")
            dataset_mode_path = self.dataset_path
        
        for category in os.listdir(dataset_mode_path):
            category_path = os.path.join(dataset_mode_path, category)
            if os.path.isdir(category_path):
                merged_path = os.path.join(category_path, 'merged.jsonl')
                if os.path.exists(merged_path):
                    self.problem_files[category] = ['merged.jsonl']
                else:
                    self.problem_files[category] = []
                    for file in os.listdir(category_path):
                        if file.endswith('.json'):
                            self.problem_files[category].append(file)
    
    def get_categories(self) -> List[str]:
        """Get list of available problem categories."""
        return list(self.problem_files.keys())
    
    def get_all_problems(self, num: int = 1) -> List[MathProblem]:
        """
        Get a specified number of random math problems from any category.
        """
        all_problems = []
        for category, files in self.problem_files.items():
            category_path = os.path.join(self.dataset_path, category)
            if files == ['merged.jsonl']:
                merged_path = os.path.join(category_path, 'merged.jsonl')
                with open(merged_path, 'r') as f:
                    for i, line in enumerate(f):
                        data = json.loads(line)
                        all_problems.append(MathProblem(
                            problem_id=data.get("id", str(i)),
                            problem=data["problem"],
                            solution=data["solution"],
                            level=data.get("level"),
                            problem_type=data.get("source") or data.get("type", "unknown")
                        ))
            else:
                for file in files:
                    problem_id = file.replace('.json', '')
                    all_problems.append(MathProblem.from_json_file(
                        os.path.join(category_path, file),
                        problem_id
                    ))
        if len(all_problems) == 0:
            return []
        return all_problems[:num]
    
    def get_problems_by_category(self, category: str, num: int = 1) -> List[MathProblem]:
        """
        Get a specified number of math problems from a specific category.
        
        Args:
            category: The category to filter by (e.g., "intermediate_algebra")
            num: Number of problems to return (default: 1)
        
        Returns:
            List of MathProblem instances
        """
        if category not in self.problem_files:
            raise ValueError(f"Category {category} not found.")
        category_path = os.path.join(self.dataset_path, category)
        files = self.problem_files[category]
        problems = []
        if files == ['merged.jsonl']:
            merged_path = os.path.join(category_path, 'merged.jsonl')
            with open(merged_path, 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    problems.append(MathProblem(
                        problem_id=data.get("id", str(i)),
                        problem=data["problem"],
                        solution=data["solution"],
                        level=data.get("level"),
                        problem_type=data.get("source") or data.get("type", "unknown")
                    ))
                    if len(problems) == num:
                        break
            if self.logger:
                self.logger.info(f"Loaded {len(problems)} problems from category {category} (merged.jsonl)")
            return problems
        # Fallback: load individual files
        selected_files = random.sample(files, min(num, len(files)))
        for file in selected_files:
            problems.append(MathProblem.from_json_file(
                os.path.join(category_path, file),
                file.replace('.json', '')
            ))
        if self.logger:
            self.logger.info(f"Loaded {len(problems)} problems from category {category}")
        return problems
    
    def get_problems_by_category_in_order(self, category: str, start_idx: int = 0, num: int = 1) -> List[MathProblem]:
        """
        Get a specified number of math problems from a specific category in sequential order.
        
        Args:
            category: The category to filter by (e.g., "intermediate_algebra")
            start_idx: Starting index in the sorted list of problems (default: 0)
            num: Number of problems to return (default: 1)
        
        Returns:
            List of MathProblem instances
        """
        if category not in self.problem_files:
            raise ValueError(f"Category {category} not found.")
            
        category_path = os.path.join(self.dataset_path, category)
        files = self.problem_files[category]
        
        problems = []
        if files == ['merged.jsonl']:
            merged_path = os.path.join(category_path, 'merged.jsonl')
            
            if self.logger:
                # Count total lines in file for debugging
                total_lines = 0
                with open(merged_path, 'r') as f:
                    for _ in f:
                        total_lines += 1
            
            with open(merged_path, 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    
                    problems.append(MathProblem(
                        problem_id=data.get("id", str(i)),
                        problem=data["problem"],
                        solution=data["solution"],
                        level=data.get("level"),
                        problem_type=data.get("source") or data.get("type", "unknown")
                    ))
            if self.logger:
                self.logger.info(f"Loaded {len(problems)} problems from category {category} (merged.jsonl, in order)")
            return problems

        # Fallback: load individual files in sorted order
        sorted_files = sorted(files)
        end_idx = min(start_idx + num, len(sorted_files))
        selected_files = sorted_files[start_idx:end_idx]
        for file in selected_files:
            problems.append(MathProblem.from_json_file(
                os.path.join(category_path, file),
                file.replace('.json', '')
            ))
        if self.logger:
            self.logger.info(f"Loaded {len(problems)} problems from category {category} (in order)")
        return problems

    def get_problems_by_category_by_level_in_order(self, category: str, level: str, start_idx: int = 0, num: int = 1) -> List[MathProblem]:
        """
        Get a specified number of math problems from a specific category and level in sequential order.
        
        Args:
            category: The category to filter by (e.g., "intermediate_algebra")
            level: The difficulty level to filter by (e.g., "5" will match "Level 5")
            start_idx: Starting index in the sorted list of problems (default: 0)
            num: Number of problems to return (default: 1)
        
        Returns:
            List of MathProblem instances
        
        Raises:
            ValueError: If category not found or start_idx is out of range
        """
        
        # First get all problems from the category in order
        # Count total lines in the merged.jsonl file if it exists
        num_problems = 1000  # Default to a large number
        if self.problem_files[category] == ['merged.jsonl']:
            merged_path = os.path.join(os.path.join(self.dataset_path, category), 'merged.jsonl')
            if os.path.exists(merged_path):
                with open(merged_path, 'r') as f:
                    num_problems = sum(1 for _ in f)
        
            
        all_problems = self.get_problems_by_category_in_order(category, 0, num_problems)
        
        
        # Filter problems by level, handling both "Level X" and "X" formats
        level_to_match = f"Level {level}" if not level.startswith("Level ") else level

        
        
        level_problems = [p for p in all_problems if p.level == level_to_match]
        
        
        if not level_problems:
            if self.logger:
                self.logger.warning(f"No problems found in category {category}, level {level}")
            return []
        
        if start_idx >= len(level_problems):
            if self.logger:
                self.logger.error(f"start_idx {start_idx} is out of range. Max index is {len(level_problems) - 1}")
            raise ValueError(f"start_idx {start_idx} is out of range. Max index is {len(level_problems) - 1}")
        
        # Get the specified slice of problems
        end_idx = min(start_idx + num, len(level_problems))
        
        problems = level_problems[start_idx:end_idx]
        
        return problems

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    loader = MathLoader(logger=logger)
    
    # Print available categories
    logger.info("\nAvailable categories:")
    for category in loader.get_categories():
        logger.info(f"- {category}")
    
    # Get one random problem
    problems = loader.get_all_problems(num=1)
    logger.info("\nRandom problem:")
    logger.info(f"ID: {problems[0].id}")
    logger.info(f"Problem: {problems[0].problem}")
    logger.info(f"Solution: {problems[0].solution}")
    
    # Get problems from specific category
    algebra_problems = loader.get_problems_by_category("intermediate_algebra", num=1)
    logger.info("\nRandom Intermediate Algebra problem:")
    logger.info(f"ID: {algebra_problems[0].id}")
    logger.info(f"Problem: {algebra_problems[0].problem}")
    logger.info(f"Solution: {algebra_problems[0].solution}")
    
    # Get problems in order
    ordered_problems = loader.get_problems_by_category_in_order("intermediate_algebra", start_idx=0, num=2)
    logger.info("\nFirst two Intermediate Algebra problems in order:")
    for prob in ordered_problems:
        logger.info(f"\nID: {prob.id}")
        logger.info(f"Problem: {prob.problem}")
