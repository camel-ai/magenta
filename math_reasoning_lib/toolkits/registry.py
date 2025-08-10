"""
Toolkit Registry for Math Reasoning Library
"""

from typing import Dict, Type, List, Any
from ..core.base_classes import BaseToolkit


class MockToolkit(BaseToolkit):
    """模拟工具包，用于测试"""
    
    def __init__(self, name: str = "mock"):
        self.name = name
    
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": f"{self.name}_tool",
                "description": f"Mock {self.name} tool",
                "parameters": {}
            }
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        return f"Mock {self.name} execution result"


class ToolkitRegistry:
    """工具包注册器"""
    
    def __init__(self):
        self._toolkits: Dict[str, Type[BaseToolkit]] = {}
        self._register_builtin_toolkits()
    
    def register(self, name: str, toolkit_class: Type[BaseToolkit]):
        """注册新工具包"""
        self._toolkits[name.lower()] = toolkit_class
    
    def get(self, name: str) -> BaseToolkit:
        """获取工具包实例"""
        name = name.lower()
        if name not in self._toolkits:
            # 如果找不到工具包，返回模拟工具包
            return MockToolkit(name)
        
        return self._toolkits[name]()
    
    def list_toolkits(self) -> List[str]:
        """获取所有已注册的工具包名称"""
        return list(self._toolkits.keys())
    
    def _register_builtin_toolkits(self):
        """注册内置工具包"""
        self.register("mock", MockToolkit)
        self.register("sympy", MockToolkit)
        self.register("code_execution", MockToolkit) 