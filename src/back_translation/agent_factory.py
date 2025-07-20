
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

def create_gpt4o_mini_agent(system_message = "You are a mathematical problem solver.") -> ChatAgent:
    """
    Create a GPT-4o-mini agent using the CAMEL framework.
    
    Returns:
        ChatAgent: Configured GPT-4o-mini agent
    """
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(
            temperature=0.5,
            top_p=0.95,
            max_tokens=4000
        ).as_dict(),
    )
    return ChatAgent(system_message=system_message, model=model)

def create_gpt4_1_mini_agent(system_message = "You are a mathematical problem solver.") -> ChatAgent:
    """
    Create a GPT-4o-mini agent using the CAMEL framework.
    
    Returns:
        ChatAgent: Configured GPT-4o-mini agent
    """
    model_config = ChatGPTConfig(
        temperature=0,
        max_tokens=15000,
    )
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict=model_config.as_dict(),
    )
    return ChatAgent(system_message=system_message, model=model)

def create_qwen_agent(system_message = "You are a mathematical problem solver.") -> ChatAgent:
    """
    Create a Qwen/Qwen2.5-7B-Instruct agent using the CAMEL framework.
    
    Returns:
        ChatAgent: Configured Qwen/Qwen2.5-7B-Instruct agent
    """
    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="Qwen/Qwen2.5-7B-Instruct",
        url=f"http://localhost:8964/v1",
        model_config_dict={
            "temperature": 0.2,
            "max_tokens": 4000
        },
    )
    return ChatAgent(system_message=system_message, model=model)