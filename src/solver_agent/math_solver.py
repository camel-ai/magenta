from socket import timeout
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, OllamaConfig, ChatGPTConfig, GroqConfig, SambaCloudAPIConfig
from camel.types import StorageType, RoleType
from camel.agents import ChatAgent
from camel.messages import BaseMessage, HermesFunctionFormatter, ShareGPTConversation, ShareGPTMessage, FunctionCallingMessage
from camel.toolkits import CodeExecutionToolkit, MathToolkit, SymPyToolkit
try:
    from camel.toolkits import GeometryToolkit
    GEOMETRY_TOOLKIT_AVAILABLE = True
except ImportError:
    GEOMETRY_TOOLKIT_AVAILABLE = False
try:
    from camel.toolkits import MathCodeToolkit
    MATH_CODE_TOOLKIT_AVAILABLE = True
except ImportError:
    MATH_CODE_TOOLKIT_AVAILABLE = False
from camel.interpreters import SubprocessInterpreter
from camel.toolkits import FunctionTool
from pydantic_core._pydantic_core import ValidationError
from camel.models.model_manager import ModelProcessingError
import os
import sys
from colorama import init, Fore, Style, Back
import time
import logging
import re

from math_loader import MathLoader
from evaluator import MathEvaluator
from prompt import *
from schema import Schema

class MathSolver:
    """
    A class for solving mathematical problems using AI-powered conversation and various toolkits.
    
    This class uses a combination of SymPy toolkit for symbolic mathematics and a code execution toolkit
    for running Python code. It employs a multi-turn conversation approach with an AI model to break down
    and solve complex mathematical problems.
    
    Attributes:
        MAX_TURNS (int): Maximum number of conversation turns allowed (default: 10)
        tools (list): List of available tools from various toolkits
        multi_step (bool): Whether to use multi-step problem solving approach
        sympy_toolkit (SymPyToolkit): Optional SymPy toolkit for symbolic mathematics
        math_code_toolkit (CodeExecutionToolkit or MathCodeToolkit): Optional toolkit for executing Python code
        model: The AI model used for problem solving
        agent (ChatAgent): The chat agent that manages conversation
        tools_output_data (list): Storage for tool outputs
        evaluator (MathEvaluator): Evaluator for checking solution correctness
    """
    
    MAX_TURNS = 15  # Maximum number of conversation turns
    
    def __init__(self, model, sympy_toolkit = False, code_toolkit = False, geometry_toolkit = False, multi_step = True, model_name = None, port = 8000, vllm_max_tokens = 8000, logger=None):
        # Note: max_tokens parameter is kept for backward compatibility but renamed to vllm_max_tokens in main.py
        """
        Initialize the MathSolver with specified toolkits and settings.
        
        Args:
            sympy_toolkit (bool): Whether to enable SymPy toolkit
            code_toolkit (bool): Whether to enable code execution toolkit
            geometry_toolkit (bool): Whether to enable geometry toolkit
            multi_step (bool): Whether to use multi-step problem solving approach
            model_name (str, optional): Name or ID of a specific model to use (e.g., fine-tuned model ID)
        """
        self.tools = []
        self.multi_step = multi_step
        self.used_sympy = False
        self.used_code_toolkit = False
        self.used_geometry = False
        self.port = port
        self.vllm_max_tokens = vllm_max_tokens  # This is vllm_max_tokens in main.py

        
        if sympy_toolkit:
            self.sympy_toolkit = SymPyToolkit(timeout=180)
            self.tools += [*self.sympy_toolkit.get_tools()]
        if code_toolkit:
            if MATH_CODE_TOOLKIT_AVAILABLE:
                self.math_code_toolkit = MathCodeToolkit(sandbox='subprocess', verbose=True, import_white_list=['sympy'], output_cap=2000, timeout=180)
            else:
                self.math_code_toolkit = CodeExecutionToolkit()
                if hasattr(self, 'logger'):
                    self.logger.info("Using CodeExecutionToolkit instead of MathCodeToolkit (not available)")
            self.tools += [*self.math_code_toolkit.get_tools()]
        if geometry_toolkit:
            if GEOMETRY_TOOLKIT_AVAILABLE:
                self.geometry_toolkit = GeometryToolkit(timeout=180)
                self.tools += [*self.geometry_toolkit.get_tools()]
            else:
                if hasattr(self, 'logger'):
                    self.logger.warning("GeometryToolkit not available in current camel version")
                else:
                    print("Warning: GeometryToolkit not available in current camel version")

        self.initialize_agent(model)
        
        self.tools_output_data = []
        self.evaluator = MathEvaluator()
        self.logger = logger
        
    def initialize_agent(self, model):
        """Initialize the ChatAgent based on the model name."""
        

        # Decide which model to load based on the name
        if model == "gpt-4o-mini":
            model_config = ChatGPTConfig(
                temperature=0,
                max_tokens=15000,
            )
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
                model_config_dict=model_config.as_dict(),
            )
        elif model == "gpt-4.1-mini":
            model_config = ChatGPTConfig(
                temperature=0,
                max_tokens=15000,
            )
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4_1_MINI,
                model_config_dict=model_config.as_dict(),
            )
        else:
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM,
                model_type=model,
                url=f"http://localhost:{self.port}/v1",
                model_config_dict={"temperature": 0, "max_tokens": self.vllm_max_tokens},  # vllm_max_tokens from main.py
            )

        if self.tools:
            self.agent = ChatAgent(model=self.model,
                                system_message=SYSTEM_PROMPT,
                                tools=self.tools,
                                )
        else:
            self.agent = ChatAgent(model=self.model,
                                system_message="You are a helpful assistant that solve math problems. Please wrapped your final answer in \\boxed{}.",
                                )
    
    def solve_math_problem(self, problem_text: str) -> str:
        """
        Solve a mathematical problem using multi-turn conversation or single-step approach.
        
        This method implements the core problem-solving logic. In multi-step mode, it:
        1. Creates a plan for solving the problem
        2. Executes the plan step by step
        3. Refines the approach based on intermediate results
        4. Continues until a solution is found or MAX_TURNS is reached
        
        Args:
            problem_text (str): The mathematical problem to solve
            
        Returns:
            str: The solution, typically in LaTeX format wrapped in \\boxed{}
        """
        self.agent.reset()
        self.used_sympy = False
        self.used_code_toolkit = False
        self.used_geometry = False

        if not self.multi_step:
            # Use single-step approach
            result = self.generate_response(problem_text, use_schema=False)
            if self.logger:
                self.logger.info(f"No multi_step, just generate response: {result}")
            return result

        # Initial prompt
        planning_prompt = PLANNING_PROMPT.format(problem_text = problem_text)

        if self.logger:
            self.logger.info(f"Planning_Prompt: {planning_prompt}")

        plan = self.agent.step(
            BaseMessage.make_user_message(
                content = planning_prompt, 
                role_name="User"
                ), 
                remove_tool_calls = True
                ).msgs[0].content
        
        prompt = EXECUTE_PROMPT.format(plan = plan, problem_text=problem_text)
        
        if self.logger:
            self.logger.info(f"Execute_Prompt: {prompt}")

        response = self.agent.step(
            BaseMessage.make_user_message(
                content = prompt, 
                role_name="User"
            )
        ).msgs[0].content
        
        if self.logger:
            self.logger.info(f"Initial response: {response}")

        turns = 0
        while turns < self.MAX_TURNS:
            # Get next step from model
            turns += 1
            if self.logger:
                self.logger.info(f"Turns: {turns}")
                
            # Check if we have a final answer (contains \\boxed{})
            if response is not None and "\\boxed{" in response:
                return response
                
            # Extract next_action from response if it exists
            next_action = self.extract_next_action(response)
            
            # If we're at max turns, force a final answer
            if not response:
                return "\\boxed{Unable to solve within tool use limit.} "

            if turns == self.MAX_TURNS - 1:
                prompt = WRAPUP_PROMPT
            else:
                # Ask for next step
                prompt = FOLLOWUP_PROMPT

            if self.logger:
                self.logger.info(f"Followup_Prompt: {prompt}")
            response = self.generate_response(prompt,use_schema=False)
            self.print_result(response)
            # Add a small delay to avoid rate limits
            time.sleep(1)
        
        # If we hit max turns without a boxed answer, wrap the last response
        self.print_result(response)
        return "\\boxed{Unable to solve within turn limit}"
    

    def generate_response(self, prompt, use_schema = False):
        """
        Generate a response from the AI model with optional schema validation.
        
        This method handles the interaction with the AI model, including retries on failures
        and optional schema validation of the response.
        
        Args:
            prompt (str): The prompt to send to the model
            use_schema (bool): Whether to validate response against Schema
            
        Returns:
            Union[Schema, str]: Parsed response if use_schema=True, else raw response string
        """
        response = None
        retry = 10
        while response is None and retry > 0:
            try:
                if use_schema:
                    response = self.agent.step(
                        BaseMessage.make_user_message(
                            content = prompt, 
                            role_name="User"
                            ), 
                            response_format=Schema
                            ).msgs[0].parsed
                else:
                    response = self.agent.step(
                        BaseMessage.make_user_message(
                            content = prompt, 
                            role_name="User"
                            ), 
                            ).msgs[0].content
            except ValidationError as e:
                retry -= 1 
            except ModelProcessingError as e:
                retry -= 1
        return response

    def extract_next_action(self, response):
        """
        Extract the next action from the model's response.
        
        Args:
            response (str): The model's response
            
        Returns:
            str: The extracted next action, or None if not found
        """
        if response is None:
            return None
            
        # Try to extract next action using the new format [NEXT_ACTION: ]
        next_action_match = re.search(r"\[NEXT_ACTION: (.*?)\]", response)
        if next_action_match:
            return next_action_match.group(1).strip()
            
        return None

    def print_result(self, response):
        """
        Print the response from the model.
        
        Args:
            response (str): The response from the model
        """
        if self.logger:
            self.logger.info(f"Response: {response}")

    def get_tool_usage(self):
        """Analyze agent's memory to detect actual tool usage by checking tool names"""
        used_sympy = self.used_sympy
        used_code_toolkit = self.used_code_toolkit
        used_geometry = self.used_geometry
        
        # Get tool names dynamically from the toolkit classes
        sympy_tools_list = []
        geometry_tools_list = []
        code_tools_list = []
        
        # If toolkits were initialized, get their tool names
        if hasattr(self, 'sympy_toolkit'):
            sympy_tools_list = [tool.get_function_name() for tool in self.sympy_toolkit.get_tools()]
        
        if hasattr(self, 'geometry_toolkit'):
            geometry_tools_list = [tool.get_function_name() for tool in self.geometry_toolkit.get_tools()]
            
        if hasattr(self, 'math_code_toolkit'):
            code_tools_list = [tool.get_function_name() for tool in self.math_code_toolkit.get_tools()]

        history = self.agent.memory.get_context()[0]
        
        # Check each message for tool calls
        for msg in history:
            if isinstance(msg, dict) and 'tool_calls' in msg:
                for call in msg['tool_calls']:
                    
                    if 'function' in call:
                        tool_name = call['function'].get('name', '').lower()
                        
                        # Check if tool name matches any sympy tools
                        if tool_name in [t.lower() for t in sympy_tools_list]:
                            used_sympy = True
                        # Check if tool name matches any code tools
                        if tool_name in [t.lower() for t in code_tools_list]:
                            used_code_toolkit = True
                        # Check if tool name matches any geometry tools
                        if tool_name in [t.lower() for t in geometry_tools_list]:
                            used_geometry = True
        
        return used_sympy, used_code_toolkit, used_geometry

    def write_to_output(self, output_file):
        """
        Write conversation history and tool outputs to a CSV file.
        
        This method collects all messages from the agent's memory, formats them
        as CSV, and appends them to an existing output file or creates a new one.
        
        Args:
            output_file (str): Path to the output CSV file
        """
        messages = [record.memory_record.message for record in self.agent.memory.retrieve()]
        csv_lines = []
        for msg in messages:
            if isinstance(msg, dict) and 'tool_calls' in msg:
                for call in msg['tool_calls']:
                    if 'function' in call:
                        tool_name = call['function'].get('name', '')
                        csv_lines.append(f"Tool Call,{tool_name}\n")
            if isinstance(msg, str):
                csv_lines.append(f"Message,{msg}\n")
        
        # Read existing CSV file if it exists, otherwise initialize an empty list
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    existing_data = f.readlines()
            except IOError:
                existing_data = []
        else:
            existing_data = []

        # Append the new data
        existing_data.extend(csv_lines)

        # Save back to the CSV file
        with open(output_file, "w") as f:
            f.writelines(existing_data)

        if self.logger:
            self.logger.info(f"Data appended and saved to {output_file}")

    def get_solver_log(self):
        """Get the full solver log from agent's memory with formatted tool calls"""
        if hasattr(self, 'agent') and self.agent.memory:
            history = self.agent.memory.get_context()[0]
            formatted_log = []
            for msg in history:
                if isinstance(msg, dict):
                    if 'tool_calls' in msg:
                        for tool_call in msg['tool_calls']:
                            if 'function' in tool_call:
                                func = tool_call['function']
                                name = func.get('name', '')
                                args = func.get('arguments', '')
                                formatted_log.append(f"<tool>\n  <tool_name>{name}</tool_name>\n  <args>{args}</args>\n</tool>\n")
                    if 'content' in msg:
                        formatted_log.append(f"<message>\n{msg['content']}\n</message>\n")
                else:
                    formatted_log.append(f"<message>\n{str(msg)}\n</message>\n")
            return '\n'.join(formatted_log), history
        return "", []

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    math_solver = MathSolver(model="gpt-4o-mini", sympy_toolkit=True, code_toolkit=True, geometry_toolkit=True, logger=logger)
    problem_text = "Solve for x: 2x + 5 = 11"
    solution = math_solver.solve_math_problem(problem_text)
    if math_solver.logger:
        math_solver.logger.info(f"Solution: {solution}")

if __name__ == "__main__":
    main()
