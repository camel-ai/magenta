"""
Database utilities for Math Reasoning Library
"""

from typing import List, Any, Dict
import json
import os


class DatabaseManager:
    """数据库管理器（简单的文件系统实现）"""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_solution(self, benchmark: str, model: str, problem: Any, solution: str):
        """保存解答到数据库"""
        filename = f"{benchmark}_{model}_solutions.jsonl"
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            "benchmark": benchmark,
            "model": model,
            "problem_id": getattr(problem, 'problem_id', 'unknown'),
            "problem_text": getattr(problem, 'problem_text', ''),
            "solution": solution
        }
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def save_enhanced_solution(self, benchmark: str, original_solution: str, enhanced_solution: str):
        """保存增强后的解答"""
        filename = f"{benchmark}_enhanced_solutions.jsonl"
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            "benchmark": benchmark,
            "original_solution": original_solution,
            "enhanced_solution": enhanced_solution
        }
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def get_solutions(self, benchmark: str) -> List[str]:
        """获取解答数据"""
        solutions = []
        pattern = f"{benchmark}_*_solutions.jsonl"
        
        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"{benchmark}_") and filename.endswith("_solutions.jsonl"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        solutions.append(data['solution'])
        
        return solutions
    
    def get_enhanced_solutions(self, benchmark: str) -> List[str]:
        """获取增强后的解答数据"""
        solutions = []
        filename = f"{benchmark}_enhanced_solutions.jsonl"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    solutions.append(data['enhanced_solution'])
        
        return solutions 