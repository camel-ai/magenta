"""
Model Registry for Math Reasoning Library
"""

from typing import Dict, Type, List
from ..core.base_classes import BaseModel


class MockModel(BaseModel):
    """模拟模型，用于测试"""
    
    def __init__(self, name: str = "mock"):
        self.name = name
    
    def generate(self, prompt: str, **kwargs) -> str:
        return f"Mock response for: {prompt[:50]}..."
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if messages:
            last_msg = messages[-1].get("content", "")
            return f"Mock response for: {last_msg[:50]}..."
        return "Mock response"


class ModelRegistry:
    """模型注册器"""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}
        self._register_builtin_models()
    
    def register(self, name: str, model_class: Type[BaseModel]):
        """注册新模型"""
        self._models[name.lower()] = model_class
    
    def get(self, name: str) -> BaseModel:
        """获取模型实例"""
        name = name.lower()
        if name not in self._models:
            # 如果找不到模型，返回模拟模型
            return MockModel(name)
        
        return self._models[name]()
    
    def list_models(self) -> List[str]:
        """获取所有已注册的模型名称"""
        return list(self._models.keys())
    
    def _register_builtin_models(self):
        """注册内置模型"""
        self.register("mock", MockModel)
        self.register("gpt-4o-mini", MockModel)
        self.register("gpt-3.5-turbo", MockModel) 