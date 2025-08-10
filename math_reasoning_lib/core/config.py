"""
Configuration Management for Math Reasoning Library

统一的配置管理，支持YAML/JSON配置文件和代码配置
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverConfig:
    """求解器配置"""
    max_iterations: int = 10
    timeout: int = 300  # 秒
    multi_step: bool = True
    enable_verification: bool = True
    retry_attempts: int = 3


@dataclass
class EnhancementConfig:
    """数据增强配置"""
    max_retries: int = 3
    enable_verification: bool = True
    cot_generation: bool = True
    back_translation: bool = True
    temperature: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 100
    warmup_steps: int = 100


@dataclass
class EvaluationConfig:
    """评估配置"""
    timeout: int = 300
    batch_size: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 1024
    enable_detailed_metrics: bool = True


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "math_reasoning"
    username: Optional[str] = None
    password: Optional[str] = None
    file_path: str = "math_reasoning.db"  # for sqlite


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class PipelineConfig:
    """管道总配置"""
    solver_config: SolverConfig = field(default_factory=SolverConfig)
    enhancement_config: EnhancementConfig = field(default_factory=EnhancementConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    
    # 输出目录
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """从配置文件加载"""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """从字典创建配置"""
        config = cls()
        
        # 更新各个子配置
        if 'solver' in data:
            config.solver_config = SolverConfig(**data['solver'])
        
        if 'enhancement' in data:
            config.enhancement_config = EnhancementConfig(**data['enhancement'])
        
        if 'training' in data:
            config.training_config = TrainingConfig(**data['training'])
        
        if 'evaluation' in data:
            config.evaluation_config = EvaluationConfig(**data['evaluation'])
        
        if 'database' in data:
            config.database_config = DatabaseConfig(**data['database'])
        
        if 'logging' in data:
            config.logging_config = LoggingConfig(**data['logging'])
        
        # 更新其他配置
        for key in ['openai_api_key', 'anthropic_api_key', 'mistral_api_key', 
                   'output_dir', 'cache_dir']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'solver': self.solver_config.__dict__,
            'enhancement': self.enhancement_config.__dict__,
            'training': self.training_config.__dict__,
            'evaluation': self.evaluation_config.__dict__,
            'database': self.database_config.__dict__,
            'logging': self.logging_config.__dict__,
            'openai_api_key': self.openai_api_key,
            'anthropic_api_key': self.anthropic_api_key,
            'mistral_api_key': self.mistral_api_key,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
        }
    
    def save(self, config_path: Union[str, Path]):
        """保存配置到文件"""
        config_path = Path(config_path)
        data = self.to_dict()
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_default_config() -> PipelineConfig:
    """创建默认配置"""
    return PipelineConfig()


def create_config_template(output_path: Union[str, Path]):
    """创建配置模板文件"""
    config = create_default_config()
    config.save(output_path)


# 预定义配置模板
MATH_BENCHMARK_CONFIG = {
    "solver": {
        "max_iterations": 15,
        "timeout": 600,
        "multi_step": True,
        "enable_verification": True,
        "retry_attempts": 3
    },
    "enhancement": {
        "max_retries": 3,
        "enable_verification": True,
        "cot_generation": True,
        "temperature": 0.1
    },
    "training": {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "rank": 64,
        "max_seq_length": 4096
    }
}

GSM8K_BENCHMARK_CONFIG = {
    "solver": {
        "max_iterations": 10,
        "timeout": 300,
        "multi_step": True,
        "enable_verification": True,
        "retry_attempts": 2
    },
    "enhancement": {
        "max_retries": 2,
        "enable_verification": True,
        "cot_generation": True,
        "temperature": 0.0
    },
    "training": {
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "rank": 32,
        "max_seq_length": 2048
    }
}

def get_benchmark_config(benchmark_name: str) -> Dict[str, Any]:
    """获取特定benchmark的预设配置"""
    configs = {
        "math": MATH_BENCHMARK_CONFIG,
        "gsm8k": GSM8K_BENCHMARK_CONFIG,
    }
    
    return configs.get(benchmark_name.lower(), MATH_BENCHMARK_CONFIG) 