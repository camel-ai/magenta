"""
Back Translator for Math Reasoning Library
"""

from typing import Any
from ..core.base_classes import BaseEnhancer, BaseModel


class BackTranslator(BaseEnhancer):
    """反向翻译器，用于数据增强"""
    
    def __init__(self, model: BaseModel, config: Any = None):
        self.model = model
        self.config = config
    
    def enhance(self, solution: str) -> str:
        """
        增强解答
        
        Args:
            solution: 原始解答
            
        Returns:
            str: 增强后的解答
        """
        # 简单实现：添加CoT推理步骤
        prompt = f"""请改进以下数学解答，使其更加清晰和详细：

原始解答:
{solution}

请提供改进后的解答，包含更清晰的推理步骤。
"""
        
        try:
            enhanced = self.model.generate(prompt)
            return enhanced
        except Exception as e:
            # 如果增强失败，返回原始解答
            return solution 