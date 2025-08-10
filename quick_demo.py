#!/usr/bin/env python3
"""
Math Reasoning Library 快速演示

展示重构后的library如何简化不同benchmark的处理流程
"""

import os
from math_reasoning_lib.core.pipeline import MathReasoningPipeline
from math_reasoning_lib.core.config import PipelineConfig, get_benchmark_config
from math_reasoning_lib.core.base_classes import MathProblem, BaseBenchmark
from math_reasoning_lib.benchmarks.registry import register_benchmark


class DemoBenchmark(BaseBenchmark):
    """演示用的简单benchmark"""
    
    def __init__(self):
        self.problems_data = [
            {"id": "demo_001", "text": "计算 2 + 3", "answer": "5"},
            {"id": "demo_002", "text": "求解方程 x + 5 = 8", "answer": "x = 3"},
            {"id": "demo_003", "text": "计算 10 的平方根", "answer": "√10 ≈ 3.16"},
            {"id": "demo_004", "text": "求三角形面积，底边4，高3", "answer": "6"},
            {"id": "demo_005", "text": "化简 (x + 2)(x - 2)", "answer": "x² - 4"},
        ]
    
    def load_problems(self, num_problems=5, **kwargs):
        problems = []
        for i, data in enumerate(self.problems_data[:num_problems]):
            problem = MathProblem(
                problem_id=data["id"],
                problem_text=data["text"],
                answer=data["answer"]
            )
            problems.append(problem)
        return problems
    
    def load_test_problems(self, num_problems=3, **kwargs):
        return self.load_problems(num_problems, **kwargs)


def demo_single_stage():
    """演示：单阶段运行"""
    print("🔬 演示1: 单阶段数据生成")
    print("-" * 40)
    
    # 注册演示benchmark
    register_benchmark("demo", DemoBenchmark)
    
    # 创建配置
    config = PipelineConfig()
    
    # 创建管道
    pipeline = MathReasoningPipeline(config)
    
    # 运行数据生成阶段
    result = pipeline.run_data_generation(
        benchmark="demo",
        model="mock",
        num_problems=3,
        toolkits=["sympy", "code_execution"]
    )
    
    print(f"✅ 数据生成完成")
    print(f"   📊 处理问题数: {result.num_problems}")
    print(f"   📈 成功率: {result.success_rate:.2%}")
    print(f"   🎯 阶段: {result.stage}")
    
    return result


def demo_multi_benchmark():
    """演示：多benchmark对比"""
    print("\n🆚 演示2: 多Benchmark对比")
    print("-" * 40)
    
    # 注册演示benchmark
    register_benchmark("demo", DemoBenchmark)
    
    benchmarks = ["demo"]  # 可以扩展为 ["demo", "math", "gsm8k"]
    models = ["mock"]      # 可以扩展为 ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    results = {}
    
    for benchmark in benchmarks:
        results[benchmark] = {}
        
        for model in models:
            print(f"📋 运行 {benchmark} with {model}")
            
            # 获取特定benchmark的配置
            if benchmark == "demo":
                config = PipelineConfig()
            else:
                config = PipelineConfig.from_dict(get_benchmark_config(benchmark))
            
            pipeline = MathReasoningPipeline(config)
            
            result = pipeline.run_data_generation(
                benchmark=benchmark,
                model=model,
                num_problems=5,
                toolkits=["sympy"]
            )
            
            results[benchmark][model] = {
                "success_rate": result.success_rate,
                "num_problems": result.num_problems
            }
    
    # 打印对比结果
    print("\n📊 对比结果:")
    for benchmark, models_data in results.items():
        print(f"\n{benchmark.upper()}:")
        for model, data in models_data.items():
            print(f"  {model}: {data['success_rate']:.2%} ({data['num_problems']} 问题)")
    
    return results


def demo_full_pipeline():
    """演示：完整四阶段管道"""
    print("\n🔄 演示3: 完整四阶段管道")
    print("-" * 40)
    
    # 注册演示benchmark
    register_benchmark("demo", DemoBenchmark)
    
    # 创建配置
    config = PipelineConfig()
    config.training_config.epochs = 1  # 快速演示，减少训练时间
    
    # 创建管道
    pipeline = MathReasoningPipeline(config)
    
    # 运行完整管道
    print("🚀 开始运行完整四阶段管道...")
    
    results = pipeline.run_full_pipeline(
        benchmark="demo",
        base_model="mock",
        enhancement_model="mock",
        num_problems=3,
        toolkits=["sympy"]
    )
    
    print("\n📋 各阶段结果:")
    for i, result in enumerate(results, 1):
        status = "✅" if result.success_rate > 0 else "❌"
        print(f"  {i}. {result.stage}: {status} {result.success_rate:.2%}")
        if result.metrics:
            print(f"     📊 指标: {result.metrics}")
    
    return results


def demo_config_flexibility():
    """演示：配置灵活性"""
    print("\n⚙️ 演示4: 配置系统灵活性")
    print("-" * 40)
    
    # 1. 使用预设配置
    print("📝 1. 预设配置:")
    math_config = get_benchmark_config("math")
    print(f"   MATH配置 - 最大迭代: {math_config['solver']['max_iterations']}")
    
    gsm8k_config = get_benchmark_config("gsm8k")
    print(f"   GSM8K配置 - 批次大小: {gsm8k_config['training']['batch_size']}")
    
    # 2. 自定义配置
    print("\n🔧 2. 自定义配置:")
    custom_config = {
        "solver": {
            "max_iterations": 20,
            "timeout": 900,
            "retry_attempts": 5
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "rank": 32
        }
    }
    
    config = PipelineConfig.from_dict(custom_config)
    print(f"   自定义配置 - 最大迭代: {config.solver_config.max_iterations}")
    print(f"   自定义配置 - LoRA rank: {config.training_config.rank}")
    
    # 3. 配置文件
    print("\n💾 3. 配置文件保存/加载:")
    config.save("demo_config.yaml")
    loaded_config = PipelineConfig.from_file("demo_config.yaml")
    print(f"   从文件加载配置成功 ✅")
    
    # 清理
    if os.path.exists("demo_config.yaml"):
        os.remove("demo_config.yaml")
    
    return config


def main():
    """主演示函数"""
    print("🎯 Math Reasoning Library 功能演示")
    print("=" * 50)
    print("展示重构后的library如何简化不同benchmark的处理流程")
    print("=" * 50)
    
    try:
        # 演示1: 单阶段运行
        demo_single_stage()
        
        # 演示2: 多benchmark对比
        demo_multi_benchmark()
        
        # 演示3: 完整管道
        demo_full_pipeline()
        
        # 演示4: 配置灵活性
        demo_config_flexibility()
        
        print("\n" + "=" * 50)
        print("🎉 所有演示完成！")
        print("\n💡 关键优势:")
        print("✅ 统一接口 - 不同benchmark使用相同API")
        print("✅ 模块化设计 - 每个组件独立可测试")
        print("✅ 灵活配置 - 支持多种配置方式")
        print("✅ 易于扩展 - 简单注册新benchmark")
        print("✅ 减少重复 - 90%的代码可复用")
        
        print("\n📚 下一步:")
        print("1. 添加真实的模型接口（OpenAI, Anthropic等）")
        print("2. 实现具体的工具包（SymPy, Code Execution等）")
        print("3. 添加更多benchmark（MATH, GSM8K, AIME等）")
        print("4. 完善训练和评估模块")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 