"""
Base Classes for Math Reasoning Library

定义统一的抽象基类和数据结构
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MathProblem:
    """数学问题基础数据结构"""
    problem_id: str
    problem_text: str
    answer: Any
    
    def __str__(self) -> str:
        return f"Problem {self.problem_id}: {self.problem_text}"


class BaseBenchmark(ABC):
    """Benchmark抽象基类"""
    
    @abstractmethod
    def load_problems(self, num_problems: int = 100, **kwargs) -> List[MathProblem]:
        """
        加载问题
        
        Args:
            num_problems: 问题数量
            **kwargs: 其他参数
            
        Returns:
            List[MathProblem]: 问题列表
        """
        pass
    
    @abstractmethod
    def load_test_problems(self, num_problems: int = 100, **kwargs) -> List[MathProblem]:
        """
        加载测试问题
        
        Args:
            num_problems: 问题数量
            **kwargs: 其他参数
            
        Returns:
            List[MathProblem]: 测试问题列表
        """
        pass
    
    def evaluate_solution(self, problem: MathProblem, solution: str) -> Dict[str, Any]:
        """
        评估解答
        
        Args:
            problem: 问题
            solution: 解答
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 默认实现：简单的字符串匹配
        is_correct = str(problem.answer).lower() in solution.lower()
        return {
            "correct": is_correct,
            "problem_id": problem.problem_id,
            "expected_answer": problem.answer
        }
    
    def get_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算性能指标
        
        Args:
            evaluation_results: 评估结果列表
            
        Returns:
            Dict[str, Any]: 性能指标
        """
        if not evaluation_results:
            return {"accuracy": 0.0, "total": 0}
        
        total = len(evaluation_results)
        correct = sum(1 for r in evaluation_results if r.get("correct", False))
        
        return {
            "accuracy": correct / total,
            "total": total,
            "correct": correct
        }


class BaseModel(ABC):
    """模型抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        对话模式生成
        
        Args:
            messages: 对话消息列表
            **kwargs: 其他参数
            
        Returns:
            str: 生成的回复
        """
        pass


class BaseToolkit(ABC):
    """工具包抽象基类"""
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        获取工具列表
        
        Returns:
            List[Dict[str, Any]]: 工具定义列表
        """
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            Any: 执行结果
        """
        pass


class BaseAgent(ABC):
    """智能代理抽象基类"""
    
    def __init__(self, model: BaseModel, toolkits: List[BaseToolkit] = None):
        """
        初始化代理
        
        Args:
            model: 语言模型
            toolkits: 工具包列表
        """
        self.model = model
        self.toolkits = toolkits or []
    
    @abstractmethod
    def solve(self, problem: MathProblem) -> str:
        """
        求解问题
        
        Args:
            problem: 数学问题
            
        Returns:
            str: 解答
        """
        pass


class BaseEnhancer(ABC):
    """数据增强器抽象基类"""
    
    @abstractmethod
    def enhance(self, solution: str) -> str:
        """
        增强解答
        
        Args:
            solution: 原始解答
            
        Returns:
            str: 增强后的解答
        """
        pass


class BaseTrainer(ABC):
    """训练器抽象基类"""
    
    @abstractmethod
    def train(self, training_data: List[Any], output_dir: str, **kwargs) -> str:
        """
        训练模型
        
        Args:
            training_data: 训练数据
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            str: 训练好的模型路径
        """
        pass
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """
        获取训练指标
        
        Returns:
            Dict[str, Any]: 训练指标
        """
        return {}


class BaseEvaluator(ABC):
    """评估器抽象基类"""
    
    @abstractmethod
    def evaluate(self, problems: List[MathProblem], benchmark: BaseBenchmark) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            problems: 测试问题列表
            benchmark: benchmark实例
            
        Returns:
            Dict[str, Any]: 评估指标
        """
        pass 