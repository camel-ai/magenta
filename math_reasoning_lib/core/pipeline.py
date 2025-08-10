"""
Core Pipeline for Math Reasoning Library

统一的数学推理管道，支持不同benchmark的端到端处理
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .base_classes import BaseBenchmark, BaseModel, BaseToolkit
from ..benchmarks.registry import BenchmarkRegistry
from ..models.registry import ModelRegistry
from ..toolkits.registry import ToolkitRegistry
from ..agents.solver_agent import SolverAgent
from ..enhancement.back_translator import BackTranslator
from ..training.sft_trainer import SFTTrainer
from ..evaluation.evaluator import Evaluator
from ..utils.logging import setup_logger
from ..utils.database import DatabaseManager

logger = setup_logger(__name__)


@dataclass
class PipelineResults:
    """管道执行结果"""
    stage: str
    benchmark: str
    model: str
    num_problems: int
    success_rate: float
    output_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


class MathReasoningPipeline:
    """
    统一的数学推理管道
    
    支持不同benchmark通过相同的流程：
    数据生成 -> 增强 -> 训练 -> 评估
    """
    
    def __init__(self, config: Union[PipelineConfig, Dict[str, Any], str]):
        """
        初始化管道
        
        Args:
            config: 配置对象、字典或配置文件路径
        """
        if isinstance(config, str):
            self.config = PipelineConfig.from_file(config)
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
            
        self.results: List[PipelineResults] = []
        self.db_manager = DatabaseManager(self.config.database_config)
        
        # 注册组件
        self.benchmark_registry = BenchmarkRegistry()
        self.model_registry = ModelRegistry()
        self.toolkit_registry = ToolkitRegistry()
        
        logger.info(f"Pipeline initialized with config: {self.config}")
    
    def register_benchmark(self, name: str, benchmark_class: type):
        """注册新的benchmark"""
        self.benchmark_registry.register(name, benchmark_class)
        logger.info(f"Registered benchmark: {name}")
    
    def register_model(self, name: str, model_class: type):
        """注册新的模型"""
        self.model_registry.register(name, model_class)
        logger.info(f"Registered model: {name}")
    
    def register_toolkit(self, name: str, toolkit_class: type):
        """注册新的工具包"""
        self.toolkit_registry.register(name, toolkit_class)
        logger.info(f"Registered toolkit: {name}")
    
    def run_data_generation(
        self,
        benchmark: str,
        model: str,
        num_problems: int = 100,
        toolkits: List[str] = None,
        **kwargs
    ) -> PipelineResults:
        """
        阶段1: 数据生成
        
        Args:
            benchmark: benchmark名称
            model: 模型名称
            num_problems: 问题数量
            toolkits: 使用的工具包列表
            **kwargs: 其他参数
            
        Returns:
            PipelineResults: 执行结果
        """
        logger.info(f"Starting data generation: {benchmark} with {model}")
        
        try:
            # 获取benchmark
            benchmark_instance = self.benchmark_registry.get(benchmark)
            
            # 获取模型
            model_instance = self.model_registry.get(model)
            
            # 获取工具包
            toolkit_instances = []
            if toolkits:
                for toolkit_name in toolkits:
                    toolkit_instances.append(self.toolkit_registry.get(toolkit_name))
            
            # 创建求解代理
            solver = SolverAgent(
                model=model_instance,
                toolkits=toolkit_instances,
                config=self.config.solver_config
            )
            
            # 加载问题
            problems = benchmark_instance.load_problems(
                num_problems=num_problems,
                **kwargs
            )
            
            # 求解问题
            solutions = []
            success_count = 0
            
            for i, problem in enumerate(problems):
                logger.info(f"Solving problem {i+1}/{len(problems)}")
                
                try:
                    solution = solver.solve(problem)
                    solutions.append(solution)
                    
                    # 保存到数据库
                    self.db_manager.save_solution(
                        benchmark=benchmark,
                        model=model,
                        problem=problem,
                        solution=solution
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to solve problem {i+1}: {e}")
                    solutions.append(None)
            
            success_rate = success_count / len(problems)
            
            result = PipelineResults(
                stage="data_generation",
                benchmark=benchmark,
                model=model,
                num_problems=len(problems),
                success_rate=success_rate,
                metrics={"solutions_generated": success_count}
            )
            
            self.results.append(result)
            logger.info(f"Data generation completed: {success_rate:.2%} success rate")
            
            return result
            
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            result = PipelineResults(
                stage="data_generation",
                benchmark=benchmark,
                model=model,
                num_problems=0,
                success_rate=0.0,
                errors=[str(e)]
            )
            self.results.append(result)
            return result
    
    def run_enhancement(
        self,
        benchmark: str,
        enhancement_model: str,
        input_data: Optional[str] = None,
        **kwargs
    ) -> PipelineResults:
        """
        阶段2: 数据增强
        
        Args:
            benchmark: benchmark名称
            enhancement_model: 用于增强的模型
            input_data: 输入数据路径，None则从数据库读取
            **kwargs: 其他参数
            
        Returns:
            PipelineResults: 执行结果
        """
        logger.info(f"Starting enhancement for {benchmark} with {enhancement_model}")
        
        try:
            # 获取增强模型
            model_instance = self.model_registry.get(enhancement_model)
            
            # 创建反向翻译器
            back_translator = BackTranslator(
                model=model_instance,
                config=self.config.enhancement_config
            )
            
            # 获取数据
            if input_data:
                solutions = self._load_solutions_from_file(input_data)
            else:
                solutions = self.db_manager.get_solutions(benchmark=benchmark)
            
            # 增强数据
            enhanced_solutions = []
            success_count = 0
            
            for i, solution in enumerate(solutions):
                logger.info(f"Enhancing solution {i+1}/{len(solutions)}")
                
                try:
                    enhanced = back_translator.enhance(solution)
                    enhanced_solutions.append(enhanced)
                    
                    # 保存增强结果
                    self.db_manager.save_enhanced_solution(
                        benchmark=benchmark,
                        original_solution=solution,
                        enhanced_solution=enhanced
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to enhance solution {i+1}: {e}")
                    enhanced_solutions.append(solution)  # 保留原始解答
            
            success_rate = success_count / len(solutions)
            
            result = PipelineResults(
                stage="enhancement",
                benchmark=benchmark,
                model=enhancement_model,
                num_problems=len(solutions),
                success_rate=success_rate,
                metrics={"enhanced_solutions": success_count}
            )
            
            self.results.append(result)
            logger.info(f"Enhancement completed: {success_rate:.2%} success rate")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            result = PipelineResults(
                stage="enhancement",
                benchmark=benchmark,
                model=enhancement_model,
                num_problems=0,
                success_rate=0.0,
                errors=[str(e)]
            )
            self.results.append(result)
            return result
    
    def run_training(
        self,
        base_model: str,
        benchmark: str,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PipelineResults:
        """
        阶段3: 模型训练
        
        Args:
            base_model: 基础模型名称
            benchmark: benchmark名称
            training_config: 训练配置
            **kwargs: 其他参数
            
        Returns:
            PipelineResults: 执行结果
        """
        logger.info(f"Starting training {base_model} on {benchmark}")
        
        try:
            # 合并训练配置
            config = self.config.training_config.copy()
            if training_config:
                config.update(training_config)
            
            # 创建训练器
            trainer = SFTTrainer(
                base_model=base_model,
                config=config
            )
            
            # 获取训练数据
            training_data = self.db_manager.get_enhanced_solutions(benchmark=benchmark)
            
            # 训练模型
            model_path = trainer.train(
                training_data=training_data,
                output_dir=f"outputs/{benchmark}_{base_model}",
                **kwargs
            )
            
            result = PipelineResults(
                stage="training",
                benchmark=benchmark,
                model=base_model,
                num_problems=len(training_data),
                success_rate=1.0,  # 训练完成即成功
                output_path=model_path,
                metrics=trainer.get_training_metrics()
            )
            
            self.results.append(result)
            logger.info(f"Training completed: {model_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            result = PipelineResults(
                stage="training",
                benchmark=benchmark,
                model=base_model,
                num_problems=0,
                success_rate=0.0,
                errors=[str(e)]
            )
            self.results.append(result)
            return result
    
    def run_evaluation(
        self,
        model_path: str,
        benchmark: str,
        num_problems: int = 100,
        **kwargs
    ) -> PipelineResults:
        """
        阶段4: 模型评估
        
        Args:
            model_path: 训练好的模型路径
            benchmark: benchmark名称
            num_problems: 评估问题数量
            **kwargs: 其他参数
            
        Returns:
            PipelineResults: 执行结果
        """
        logger.info(f"Starting evaluation of {model_path} on {benchmark}")
        
        try:
            # 获取benchmark
            benchmark_instance = self.benchmark_registry.get(benchmark)
            
            # 创建评估器
            evaluator = Evaluator(
                model_path=model_path,
                config=self.config.evaluation_config
            )
            
            # 加载测试问题
            test_problems = benchmark_instance.load_test_problems(
                num_problems=num_problems,
                **kwargs
            )
            
            # 评估模型
            metrics = evaluator.evaluate(
                problems=test_problems,
                benchmark=benchmark_instance
            )
            
            result = PipelineResults(
                stage="evaluation",
                benchmark=benchmark,
                model=model_path,
                num_problems=len(test_problems),
                success_rate=metrics.get("accuracy", 0.0),
                metrics=metrics
            )
            
            self.results.append(result)
            logger.info(f"Evaluation completed: {metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            result = PipelineResults(
                stage="evaluation",
                benchmark=benchmark,
                model=model_path,
                num_problems=0,
                success_rate=0.0,
                errors=[str(e)]
            )
            self.results.append(result)
            return result
    
    def run_full_pipeline(
        self,
        benchmark: str,
        base_model: str,
        enhancement_model: Optional[str] = None,
        num_problems: int = 100,
        toolkits: List[str] = None,
        **kwargs
    ) -> List[PipelineResults]:
        """
        运行完整的四阶段管道
        
        Args:
            benchmark: benchmark名称
            base_model: 基础模型名称
            enhancement_model: 增强模型名称，默认使用base_model
            num_problems: 问题数量
            toolkits: 工具包列表
            **kwargs: 其他参数
            
        Returns:
            List[PipelineResults]: 各阶段执行结果
        """
        logger.info(f"Starting full pipeline: {benchmark} with {base_model}")
        
        pipeline_results = []
        
        # 阶段1: 数据生成
        result1 = self.run_data_generation(
            benchmark=benchmark,
            model=base_model,
            num_problems=num_problems,
            toolkits=toolkits,
            **kwargs
        )
        pipeline_results.append(result1)
        
        # 阶段2: 数据增强
        enhancement_model = enhancement_model or base_model
        result2 = self.run_enhancement(
            benchmark=benchmark,
            enhancement_model=enhancement_model,
            **kwargs
        )
        pipeline_results.append(result2)
        
        # 阶段3: 模型训练
        result3 = self.run_training(
            base_model=base_model,
            benchmark=benchmark,
            **kwargs
        )
        pipeline_results.append(result3)
        
        # 阶段4: 模型评估
        if result3.output_path:
            result4 = self.run_evaluation(
                model_path=result3.output_path,
                benchmark=benchmark,
                num_problems=num_problems,
                **kwargs
            )
            pipeline_results.append(result4)
        
        logger.info(f"Full pipeline completed for {benchmark}")
        return pipeline_results
    
    def get_results(self) -> List[PipelineResults]:
        """获取所有执行结果"""
        return self.results
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        import json
        
        results_data = []
        for result in self.results:
            results_data.append({
                "stage": result.stage,
                "benchmark": result.benchmark,
                "model": result.model,
                "num_problems": result.num_problems,
                "success_rate": result.success_rate,
                "output_path": result.output_path,
                "metrics": result.metrics,
                "errors": result.errors
            })
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def _load_solutions_from_file(self, file_path: str) -> List[Any]:
        """从文件加载解答数据"""
        # 实现文件加载逻辑
        pass 