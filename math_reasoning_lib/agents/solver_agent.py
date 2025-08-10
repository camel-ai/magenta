"""
Solver Agent for Math Reasoning Library
"""

from typing import List, Any
from ..core.base_classes import BaseAgent, BaseModel, BaseToolkit, MathProblem


class SolverAgent(BaseAgent):
    """数学问题求解代理"""
    
    def __init__(self, model: BaseModel, toolkits: List[BaseToolkit] = None, config: Any = None):
        super().__init__(model, toolkits)
        self.config = config
    
    def solve(self, problem: MathProblem) -> str:
        """
        求解数学问题
        
        Args:
            problem: 数学问题
            
        Returns:
            str: 解答
        """
        # 构建提示
        prompt = f"""请解答以下数学问题：

问题: {problem.problem_text}

请提供详细的解答过程和最终答案。
"""
        
        # 使用模型生成解答
        try:
            solution = self.model.generate(prompt)
            return solution
        except Exception as e:
            return f"求解失败: {str(e)}" 