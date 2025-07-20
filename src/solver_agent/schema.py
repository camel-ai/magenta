from pydantic import BaseModel, Field, validator


class Schema(BaseModel):
    """
    Schema for structuring the AI model's responses in the math problem solving process.
    
    This schema defines the expected format of responses from the AI model, including
    the execution result, reflection on the process, and next steps. Each field serves
    a specific purpose in the iterative problem-solving approach.
    
    Attributes:
        execution_result (str): The result of executing this step of the plan. If you obtain the final solution, 
            reply in LaTeX format wrapped in \\boxed{}. The response should contain a detailed step-by-step solution to the problem.
        reflection (str): Analysis of what worked well and what didn't in the current step. 
            Include specific details about tool usage and effectiveness.
        next_action (str): The next action to be taken, will be fed directly to the model in the next step. 
            Should be specific and actionable.
        refined_plan (str): The refined plan based on reflection: what to do next. 
            Should include clear steps and rationale for each step.
    """
    execution_result: str = Field(
        description="The result of executing this step of the plan. If you obtain the final solution, "
                    "reply in LaTeX format wrapped in \\boxed{}. The response should contain a "
                    "detailed step-by-step solution to the problem."
    )
    reflection: str = Field(
        description="Analysis of what worked well and what didn't in the current step. "
                    "Include specific details about tool usage and effectiveness."
    )
    next_action: str = Field(
        description="The next action to be taken, will be fed directly to the model in the next step. "
                    "Should be specific and actionable."
    )
    refined_plan: str = Field(
        description="The refined plan based on reflection: what to do next. "
                    "Should include clear steps and rationale for each step."
    )
