"""
Math Reasoning Library

统一的数学推理库，支持不同benchmark的端到端处理
"""

__version__ = "0.1.0"
__author__ = "Math Reasoning Team"

from .core.pipeline import MathReasoningPipeline, PipelineResults
from .core.config import PipelineConfig, get_benchmark_config
from .core.base_classes import (
    MathProblem, BaseBenchmark, BaseModel, BaseToolkit, 
    BaseAgent, BaseEnhancer, BaseTrainer, BaseEvaluator
)

__all__ = [
    "MathReasoningPipeline",
    "PipelineResults", 
    "PipelineConfig",
    "get_benchmark_config",
    "MathProblem",
    "BaseBenchmark",
    "BaseModel", 
    "BaseToolkit",
    "BaseAgent",
    "BaseEnhancer",
    "BaseTrainer",
    "BaseEvaluator"
] 