#!/usr/bin/env python3
"""
测试Math Reasoning Library的安装和基本功能
"""

def test_imports():
    """测试基本导入"""
    print("🔍 测试导入...")
    
    try:
        from math_reasoning_lib.core.pipeline import MathReasoningPipeline, PipelineResults
        print("✅ 核心管道导入成功")
    except ImportError as e:
        print(f"❌ 核心管道导入失败: {e}")
        return False
    
    try:
        from math_reasoning_lib.core.config import PipelineConfig, get_benchmark_config
        print("✅ 配置模块导入成功")
    except ImportError as e:
        print(f"❌ 配置模块导入失败: {e}")
        return False
    
    try:
        from math_reasoning_lib.core.base_classes import MathProblem, BaseBenchmark
        print("✅ 基础类导入成功")
    except ImportError as e:
        print(f"❌ 基础类导入失败: {e}")
        return False
    
    try:
        from math_reasoning_lib.benchmarks.registry import register_benchmark
        print("✅ Benchmark注册器导入成功")
    except ImportError as e:
        print(f"❌ Benchmark注册器导入失败: {e}")
        return False
    
    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        from math_reasoning_lib.core.pipeline import MathReasoningPipeline
        from math_reasoning_lib.core.config import PipelineConfig
        from math_reasoning_lib.core.base_classes import MathProblem, BaseBenchmark
        from math_reasoning_lib.benchmarks.registry import register_benchmark
        
        # 创建简单的测试benchmark
        class TestBenchmark(BaseBenchmark):
            def load_problems(self, num_problems=5, **kwargs):
                problems = []
                for i in range(num_problems):
                    problem = MathProblem(
                        problem_id=f"test_{i+1}",
                        problem_text=f"计算 {i+1} + {i+1}",
                        answer=str((i+1) * 2)
                    )
                    problems.append(problem)
                return problems
            
            def load_test_problems(self, num_problems=3, **kwargs):
                return self.load_problems(num_problems, **kwargs)
        
        # 注册测试benchmark
        register_benchmark("test", TestBenchmark)
        print("✅ 测试benchmark注册成功")
        
        # 创建配置
        config = PipelineConfig()
        print("✅ 配置创建成功")
        
        # 创建管道
        pipeline = MathReasoningPipeline(config)
        print("✅ 管道创建成功")
        
        # 测试数据生成（使用模拟模型）
        result = pipeline.run_data_generation(
            benchmark="test",
            model="mock",
            num_problems=3,
            toolkits=["mock"]
        )
        
        print(f"✅ 数据生成测试成功")
        print(f"   - 处理问题数: {result.num_problems}")
        print(f"   - 成功率: {result.success_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_system():
    """测试配置系统"""
    print("\n⚙️ 测试配置系统...")
    
    try:
        from math_reasoning_lib.core.config import PipelineConfig, get_benchmark_config
        
        # 测试预设配置
        math_config = get_benchmark_config("math")
        print("✅ MATH benchmark配置获取成功")
        
        gsm8k_config = get_benchmark_config("gsm8k")
        print("✅ GSM8K benchmark配置获取成功")
        
        # 测试配置创建
        config = PipelineConfig.from_dict(math_config)
        print("✅ 从字典创建配置成功")
        
        # 测试配置保存和加载
        config.save("test_config.yaml")
        loaded_config = PipelineConfig.from_file("test_config.yaml")
        print("✅ 配置保存和加载成功")
        
        # 清理测试文件
        import os
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 Math Reasoning Library 安装测试")
    print("=" * 50)
    
    success = True
    
    # 测试导入
    if not test_imports():
        success = False
    
    # 测试基本功能
    if not test_basic_functionality():
        success = False
    
    # 测试配置系统
    if not test_config_system():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！Math Reasoning Library 安装成功！")
        print("\n📚 快速开始:")
        print("1. 查看 examples/basic_usage.py 了解基本用法")
        print("2. 查看 examples/custom_benchmark.py 了解如何添加自定义benchmark")
        print("3. 查看 README.md 了解完整文档")
    else:
        print("❌ 测试失败，请检查安装")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 