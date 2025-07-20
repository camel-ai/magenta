# Prompt template for generating CoT reasoning
COT_PROMPT_TEMPLATE = """
You are a mathematical problem solver. Your task is to solve the following problem step by step.
Clearly illustrate the reasoning process and show all calculations explicitly.

Problem: {tool_name}
This problem {docstring}. 
Here is the input required for solving this problem: {arguments}

Your final answer should be wrapped in \\boxed{{}}.
Do **not** generate code.
"""

JUDGE_PROMPT_TEMPLATE = '''
You are a mathematical problem solver. Your task is to verify whether the following generated answer is mathematically equivalent to the ground truth. 
We do NOT care about the format of the answer. 

Ground Truth: {ground_truth}.

Generated Answer: {final_answer}.

First explain why (or why not) the generated answer is mathmatically equivalent to the ground truth.
Then clearly state True or False at the end, wrapped in \\boxed{{}}, i.e., either \\boxed{{True}} or \\boxed{{False}}.
'''



# Prompt template for system message

SYSTEM_MESSAGE = """
You are a mathematical solution refiner. Your task is to rewrite mathematical solutions with clarity and precision.

Guidelines for rewriting:
1. Remove all references to computational tools (e.g., "sympy", "execute_code")
2. Present a coherent, step-by-step explanation with complete mathematical details
3. Maintain mathematical accuracy and completeness throughout
4. For solutions containing <tool>...</tool> blocks, translate the computational steps into clear mathematical reasoning
5. Preserve the original solution's structure while enhancing readability
6. Fully retain all content following "Chain of Thought:" sections
7. Show all calculations explicitly - never skip steps or jump to conclusions

Your output should read as a polished mathematical explanation that appears to be written by a human mathematician.
"""

# Prompt template for reformatting the enhanced solution logs
REFORMATTING_PROMPT_TEMPLATE = """
PROBLEM:
{problem_text}

ORIGINAL SOLUTION:
{enhanced_solution_log}

YOUR  REFORMATTED SOLUTION:
"""

