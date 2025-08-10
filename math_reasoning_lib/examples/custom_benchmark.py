"""
Custom Benchmark Example for Math Reasoning Library

展示如何创建和注册自定义benchmark
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from math_reasoning_lib.core.pipeline import MathReasoningPipeline
from math_reasoning_lib.core.config import PipelineConfig
from math_reasoning_lib.core.base_classes import BaseBenchmark, MathProblem
from math_reasoning_lib.benchmarks.registry import register_benchmark


@dataclass
class CustomMathProblem(MathProblem):
    """自定义数学问题格式"""
    difficulty: str
    topic: str
    source: str
    hints: Optional[List[str]] = None


class CustomBenchmark(BaseBenchmark):
    """
    自定义benchmark示例
    
    这个示例展示如何创建一个新的benchmark类，
    可以加载自定义格式的数学问题
    """
    
    def __init__(self, data_path: str = "custom_math_data.json"):
        """
        初始化自定义benchmark
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.problems_cache = None
    
    def load_problems(
        self, 
        num_problems: int = 100, 
        difficulty: Optional[str] = None,
        topic: Optional[str] = None,
        **kwargs
    ) -> List[CustomMathProblem]:
        """
        加载问题
        
        Args:
            num_problems: 问题数量
            difficulty: 难度过滤 (easy, medium, hard)
            topic: 主题过滤 (algebra, geometry, calculus, etc.)
            **kwargs: 其他参数
            
        Returns:
            List[CustomMathProblem]: 问题列表
        """
        if self.problems_cache is None:
            self._load_data()
        
        # 过滤问题
        filtered_problems = self.problems_cache
        
        if difficulty:
            filtered_problems = [
                p for p in filtered_problems 
                if p.difficulty.lower() == difficulty.lower()
            ]
        
        if topic:
            filtered_problems = [
                p for p in filtered_problems 
                if p.topic.lower() == topic.lower()
            ]
        
        # 限制数量
        return filtered_problems[:num_problems]
    
    def load_test_problems(
        self, 
        num_problems: int = 100, 
        **kwargs
    ) -> List[CustomMathProblem]:
        """
        加载测试问题
        
        Args:
            num_problems: 问题数量
            **kwargs: 其他参数
            
        Returns:
            List[CustomMathProblem]: 测试问题列表
        """
        # 这里可以加载专门的测试集
        # 为了示例，我们使用训练集的子集
        all_problems = self.load_problems(num_problems * 2, **kwargs)
        return all_problems[num_problems:]  # 使用后半部分作为测试集
    
    def evaluate_solution(
        self, 
        problem: CustomMathProblem, 
        solution: str
    ) -> Dict[str, Any]:
        """
        评估解答
        
        Args:
            problem: 问题
            solution: 解答
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 这里实现自定义的评估逻辑
        # 例如：数值匹配、符号匹配、语义匹配等
        
        # 简单示例：检查解答中是否包含正确答案
        is_correct = str(problem.answer).lower() in solution.lower()
        
        return {
            "correct": is_correct,
            "problem_id": problem.problem_id,
            "difficulty": problem.difficulty,
            "topic": problem.topic,
            "expected_answer": problem.answer,
            "solution_length": len(solution)
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
            return {}
        
        total = len(evaluation_results)
        correct = sum(1 for r in evaluation_results if r["correct"])
        
        # 按难度分组
        by_difficulty = {}
        for result in evaluation_results:
            diff = result["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "correct": 0}
            by_difficulty[diff]["total"] += 1
            if result["correct"]:
                by_difficulty[diff]["correct"] += 1
        
        # 按主题分组
        by_topic = {}
        for result in evaluation_results:
            topic = result["topic"]
            if topic not in by_topic:
                by_topic[topic] = {"total": 0, "correct": 0}
            by_topic[topic]["total"] += 1
            if result["correct"]:
                by_topic[topic]["correct"] += 1
        
        return {
            "overall_accuracy": correct / total,
            "total_problems": total,
            "correct_answers": correct,
            "accuracy_by_difficulty": {
                diff: stats["correct"] / stats["total"] 
                for diff, stats in by_difficulty.items()
            },
            "accuracy_by_topic": {
                topic: stats["correct"] / stats["total"] 
                for topic, stats in by_topic.items()
            }
        }
    
    def _load_data(self):
        """加载数据文件"""
        if not os.path.exists(self.data_path):
            # 如果数据文件不存在，创建示例数据
            self._create_sample_data()
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.problems_cache = []
        for item in data:
            problem = CustomMathProblem(
                problem_id=item["id"],
                problem_text=item["problem"],
                answer=item["answer"],
                difficulty=item["difficulty"],
                topic=item["topic"],
                source=item.get("source", "custom"),
                hints=item.get("hints", [])
            )
            self.problems_cache.append(problem)
    
    def _create_sample_data(self):
        """创建示例数据"""
        sample_data = [
            {
                "id": "custom_001",
                "problem": "求解方程 2x + 5 = 13",
                "answer": "4",
                "difficulty": "easy",
                "topic": "algebra",
                "source": "custom_dataset",
                "hints": ["将常数项移到右边", "除以系数"]
            },
            {
                "id": "custom_002", 
                "problem": "计算圆的面积，其中半径为 5",
                "answer": "78.54",
                "difficulty": "easy",
                "topic": "geometry",
                "source": "custom_dataset",
                "hints": ["使用公式 π * r²"]
            },
            {
                "id": "custom_003",
                "problem": "求函数 f(x) = x² + 3x - 4 的最小值",
                "answer": "-6.25",
                "difficulty": "medium",
                "topic": "calculus",
                "source": "custom_dataset",
                "hints": ["找到导数为零的点", "使用二次函数的顶点公式"]
            },
            {
                "id": "custom_004",
                "problem": "解不等式 3x - 7 > 2x + 1",
                "answer": "x > 8",
                "difficulty": "medium",
                "topic": "algebra", 
                "source": "custom_dataset",
                "hints": ["移项合并同类项"]
            },
            {
                "id": "custom_005",
                "problem": "在三角形ABC中，已知a=5, b=7, C=60°，求边c的长度",
                "answer": "6.08",
                "difficulty": "hard",
                "topic": "trigonometry",
                "source": "custom_dataset",
                "hints": ["使用余弦定理", "c² = a² + b² - 2ab*cos(C)"]
            }
        ]
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)


def example_custom_benchmark():
    """示例：使用自定义benchmark"""
    print("=== 自定义Benchmark示例 ===")
    
    # 1. 注册自定义benchmark
    register_benchmark("custom", CustomBenchmark)
    print("✅ 自定义benchmark已注册")
    
    # 2. 创建配置
    config = PipelineConfig()
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 3. 创建管道
    pipeline = MathReasoningPipeline(config)
    
    # 4. 运行数据生成
    result = pipeline.run_data_generation(
        benchmark="custom",
        model="gpt-4o-mini", 
        num_problems=5,
        toolkits=["sympy"],
        difficulty="easy",
        topic="algebra"
    )
    
    print(f"✅ 数据生成完成")
    print(f"   成功率: {result.success_rate:.2%}")
    print(f"   处理问题: {result.num_problems}")
    
    return result


def example_custom_benchmark_full_pipeline():
    """示例：自定义benchmark的完整管道"""
    print("=== 自定义Benchmark完整管道 ===")
    
    # 注册benchmark
    register_benchmark("custom", CustomBenchmark)
    
    # 创建配置
    config = PipelineConfig()
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # 调整训练配置以适应小数据集
    config.training_config.epochs = 1
    config.training_config.batch_size = 1
    
    # 创建管道
    pipeline = MathReasoningPipeline(config)
    
    # 运行完整管道
    results = pipeline.run_full_pipeline(
        benchmark="custom",
        base_model="gpt-4o-mini",
        num_problems=3,
        toolkits=["sympy"],
        difficulty="easy"
    )
    
    print("✅ 完整管道执行完成")
    for result in results:
        print(f"   {result.stage}: {result.success_rate:.2%}")
    
    return results


def example_benchmark_comparison():
    """示例：比较不同benchmark的性能"""
    print("=== Benchmark比较示例 ===")
    
    # 注册自定义benchmark
    register_benchmark("custom", CustomBenchmark)
    
    benchmarks = ["custom"]  # 可以添加更多benchmark: ["custom", "math", "gsm8k"]
    models = ["gpt-4o-mini"]
    
    results = {}
    
    for benchmark in benchmarks:
        results[benchmark] = {}
        
        for model in models:
            print(f"运行 {benchmark} with {model}")
            
            config = PipelineConfig()
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
            
            pipeline = MathReasoningPipeline(config)
            
            result = pipeline.run_data_generation(
                benchmark=benchmark,
                model=model,
                num_problems=5,
                toolkits=["sympy"]
            )
            
            results[benchmark][model] = {
                "success_rate": result.success_rate,
                "num_problems": result.num_problems,
                "metrics": result.metrics
            }
    
    # 打印比较结果
    print("\n📊 比较结果:")
    for benchmark, models_data in results.items():
        print(f"\n{benchmark.upper()}:")
        for model, data in models_data.items():
            print(f"  {model}: {data['success_rate']:.2%} ({data['num_problems']} 问题)")


def main():
    """主函数"""
    print("自定义Benchmark示例")
    print("=" * 50)
    
    # 检查API key
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
        print("请设置: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 基础示例
        example_custom_benchmark()
        print()
        
        # 比较示例
        example_benchmark_comparison()
        print()
        
        # 完整管道示例（可选，需要更多时间）
        # example_custom_benchmark_full_pipeline()
        
        print("✅ 所有自定义benchmark示例运行完成！")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 