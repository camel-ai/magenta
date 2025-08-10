"""
Benchmarks module for Math Reasoning Library
"""

from .registry import BenchmarkRegistry, register_benchmark, get_benchmark, list_available_benchmarks

__all__ = [
    "BenchmarkRegistry",
    "register_benchmark", 
    "get_benchmark",
    "list_available_benchmarks"
] 