"""
Benchmark Registry for Math Reasoning Library

统一的benchmark注册器，支持动态注册和管理不同的数学推理benchmark
"""

from typing import Dict, Type, Any, List
from ..core.base_classes import BaseBenchmark


class BenchmarkRegistry:
    """Benchmark注册器"""
    
    def __init__(self):
        self._benchmarks: Dict[str, Type[BaseBenchmark]] = {}
        self._register_builtin_benchmarks()
    
    def register(self, name: str, benchmark_class: Type[BaseBenchmark]):
        """
        注册新的benchmark
        
        Args:
            name: benchmark名称
            benchmark_class: benchmark类
        """
        if not issubclass(benchmark_class, BaseBenchmark):
            raise ValueError(f"Benchmark class must inherit from BaseBenchmark")
        
        self._benchmarks[name.lower()] = benchmark_class
    
    def get(self, name: str) -> BaseBenchmark:
        """
        获取benchmark实例
        
        Args:
            name: benchmark名称
            
        Returns:
            BaseBenchmark: benchmark实例
        """
        name = name.lower()
        if name not in self._benchmarks:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(self._benchmarks.keys())}")
        
        return self._benchmarks[name]()
    
    def list_benchmarks(self) -> List[str]:
        """获取所有已注册的benchmark名称"""
        return list(self._benchmarks.keys())
    
    def _register_builtin_benchmarks(self):
        """注册内置的benchmark"""
        try:
            from .math_benchmark import MathBenchmark
            self.register("math", MathBenchmark)
        except ImportError:
            pass
        
        try:
            from .gsm8k_benchmark import GSM8KBenchmark
            self.register("gsm8k", GSM8KBenchmark)
        except ImportError:
            pass
        
        try:
            from .aime_benchmark import AIMEBenchmark
            self.register("aime", AIMEBenchmark)
        except ImportError:
            pass


# 全局注册器实例
benchmark_registry = BenchmarkRegistry()


def register_benchmark(name: str, benchmark_class: Type[BaseBenchmark]):
    """
    便捷函数：注册benchmark
    
    Args:
        name: benchmark名称
        benchmark_class: benchmark类
    """
    benchmark_registry.register(name, benchmark_class)


def get_benchmark(name: str) -> BaseBenchmark:
    """
    便捷函数：获取benchmark
    
    Args:
        name: benchmark名称
        
    Returns:
        BaseBenchmark: benchmark实例
    """
    return benchmark_registry.get(name)


def list_available_benchmarks() -> List[str]:
    """
    便捷函数：列出所有可用的benchmark
    
    Returns:
        List[str]: benchmark名称列表
    """
    return benchmark_registry.list_benchmarks() 