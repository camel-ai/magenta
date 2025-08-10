"""
Basic Usage Examples for Math Reasoning Library

展示如何使用重构后的数学推理库进行各种操作
"""

import os
from math_reasoning_lib.core.pipeline import MathReasoningPipeline
from math_reasoning_lib.core.config import PipelineConfig, get_benchmark_config


def example_full_pipeline():
    """示例：运行完整的四阶段管道"""
    print("=== 完整管道示例 ===")
    
    # 1. 创建配置
    config = PipelineConfig.from_dict(get_benchmark_config("math"))
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 2. 创建管道
    pipeline = MathReasoningPipeline(config)
    
    # 3. 运行完整管道
    results = pipeline.run_full_pipeline(
        benchmark="math",
        base_model="gpt-4o-mini",
        enhancement_model="gpt-4o-mini",
        num_problems=10,
        toolkits=["sympy", "code_execution"],
        level=1,
        dataset="algebra"
    )
    
    # 4. 查看结果
    for result in results:
        print(f"阶段: {result.stage}")
        print(f"成功率: {result.success_rate:.2%}")
        print(f"处理问题数: {result.num_problems}")
        print(f"指标: {result.metrics}")
        print("-" * 40)
    
    # 5. 保存结果
    pipeline.save_results("full_pipeline_results.json")


def example_single_stage():
    """示例：单独运行某个阶段"""
    print("=== 单阶段示例 ===")
    
    # 配置
    config = PipelineConfig.from_dict(get_benchmark_config("gsm8k"))
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 创建管道
    pipeline = MathReasoningPipeline(config)
    
    # 只运行数据生成阶段
    result = pipeline.run_data_generation(
        benchmark="gsm8k",
        model="gpt-4o-mini",
        num_problems=50,
        toolkits=["code_execution"]
    )
    
    print(f"数据生成完成，成功率: {result.success_rate:.2%}")


def example_multiple_benchmarks():
    """示例：对多个benchmark运行相同的流程"""
    print("=== 多Benchmark示例 ===")
    
    benchmarks = ["math", "gsm8k"]
    models = ["gpt-4o-mini", "gpt-3.5-turbo"]
    
    for benchmark in benchmarks:
        for model in models:
            print(f"运行 {benchmark} with {model}")
            
            # 获取特定benchmark的配置
            config = PipelineConfig.from_dict(get_benchmark_config(benchmark))
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
            
            # 创建管道
            pipeline = MathReasoningPipeline(config)
            
            # 运行数据生成
            result = pipeline.run_data_generation(
                benchmark=benchmark,
                model=model,
                num_problems=20,
                toolkits=["sympy", "code_execution"]
            )
            
            print(f"  成功率: {result.success_rate:.2%}")
            print(f"  处理问题: {result.num_problems}")


def example_custom_config():
    """示例：使用自定义配置"""
    print("=== 自定义配置示例 ===")
    
    # 创建自定义配置
    custom_config = {
        "solver": {
            "max_iterations": 20,
            "timeout": 900,
            "multi_step": True,
            "retry_attempts": 5
        },
        "enhancement": {
            "max_retries": 5,
            "temperature": 0.2
        },
        "training": {
            "epochs": 5,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "rank": 128
        },
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
    
    config = PipelineConfig.from_dict(custom_config)
    pipeline = MathReasoningPipeline(config)
    
    # 运行管道
    result = pipeline.run_data_generation(
        benchmark="math",
        model="gpt-4o-mini",
        num_problems=5,
        toolkits=["sympy"],
        level=3,
        dataset="intermediate_algebra"
    )
    
    print(f"自定义配置运行完成，成功率: {result.success_rate:.2%}")


def example_config_file():
    """示例：使用配置文件"""
    print("=== 配置文件示例 ===")
    
    # 创建配置文件
    config_data = {
        "solver": {
            "max_iterations": 15,
            "timeout": 600,
            "multi_step": True
        },
        "enhancement": {
            "max_retries": 3,
            "cot_generation": True
        },
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "rank": 64
        },
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "output_dir": "custom_outputs"
    }
    
    # 保存配置
    config = PipelineConfig.from_dict(config_data)
    config.save("example_config.yaml")
    
    # 从配置文件加载
    loaded_config = PipelineConfig.from_file("example_config.yaml")
    pipeline = MathReasoningPipeline(loaded_config)
    
    print("从配置文件创建管道成功")


def example_parallel_processing():
    """示例：并行处理多个实验"""
    print("=== 并行处理示例 ===")
    
    import concurrent.futures
    
    def run_experiment(benchmark, model):
        """运行单个实验"""
        config = PipelineConfig.from_dict(get_benchmark_config(benchmark))
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        pipeline = MathReasoningPipeline(config)
        
        result = pipeline.run_data_generation(
            benchmark=benchmark,
            model=model,
            num_problems=10,
            toolkits=["sympy"]
        )
        
        return f"{benchmark}-{model}: {result.success_rate:.2%}"
    
    # 定义实验组合
    experiments = [
        ("math", "gpt-4o-mini"),
        ("gsm8k", "gpt-4o-mini"),
        ("math", "gpt-3.5-turbo"),
        ("gsm8k", "gpt-3.5-turbo")
    ]
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_experiment, benchmark, model) 
            for benchmark, model in experiments
        ]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"实验完成: {result}")


def main():
    """主函数：运行所有示例"""
    print("Math Reasoning Library 使用示例")
    print("=" * 50)
    
    # 检查API key
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
        print("请设置: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 运行各种示例
        example_single_stage()
        print()
        
        example_custom_config()
        print()
        
        example_config_file()
        print()
        
        example_multiple_benchmarks()
        print()
        
        # 完整管道示例（需要更多时间）
        # example_full_pipeline()
        
        # 并行处理示例（需要更多资源）
        # example_parallel_processing()
        
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 