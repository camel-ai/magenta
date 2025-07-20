
import re
from collections import defaultdict
import logging

def setup_logging(log_level: str) -> logging.Logger:
    """
    Set up logging with the specified level.
    
    Args:
        log_level (str): Logging level (debug, info, warning, error, critical)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

# Function to format a problem text by joining lines, replacing special characters, and adding a period at the end if necessary
def format_problem(problem_text):
    """
    Formats a problem text by joining lines, replacing special characters, and adding a period at the end if necessary.
    
    Args:
        problem_text (str): The problem text to be formatted.
    
    Returns:
        str: The formatted problem text.
    """
    lines = problem_text.splitlines()
    lines = [' '.join(line.split()) for line in lines]
    problem_text = '\\n'.join(lines)
    problem_text = problem_text.replace('\\[', '[').replace('\\]', ']')
    problem_text = problem_text.replace('\\(', '(').replace('\\)', ')')
    if not problem_text.endswith(('.', '!', '?')):
        problem_text += '.'
    problem_text = problem_text.replace('"', '""')
    return problem_text

# Function to parse a text by extracting specific patterns and replacing special characters
def parse_text(text):
    """
    Parses a text by extracting specific patterns and replacing special characters.
    
    Args:
        text (str): The text to be parsed.
    
    Returns:
        str: The parsed text.
    """
    text = text.replace('""', '"')
    patterns = {
        "Execution_result": r'"execution_result":"(.*?)","reflection"',
        "Reflection": r'"reflection":"(.*?)","refined_plan"',
        "Refined_plan": r'"refined_plan":"(.*?)","next_action"',
        "Next_action": r'"next_action":"(.*?)"}'
    }
    extracted_parts = {key: re.search(pattern, text, re.DOTALL).group(1).replace("\\n", "\n").strip() if re.search(pattern, text, re.DOTALL) else "" for key, pattern in patterns.items()}
    text_result = ""
    for section, content in extracted_parts.items():
        if content == "":
            return text
        if section == "Execution_result":
            continue
        else:
            if len(section.split("_")) > 1:
                section_ = section.split("_")[0] + " " + section.split("_")[1]
            else:
                section_ = section
            if section == "Next_action":
                text_result += f"{section_}:\n{content}"
            else:
                text_result += f"{section_}:\n{content}\n\n"
    return text_result

# Function to process a log by parsing solutions and extracting relevant information
def process_log(solutions):
    """
    Processes a log by parsing solutions and extracting relevant information.
    
    Args:
        solutions (list): A list of solutions to be processed.
    
    Returns:
        list: A list of parsed data.
    """
    parsed_data_list = []
    for solution in solutions:
        message_pattern = re.compile(r"<message>(.*?)</message>", re.DOTALL)
        tool_pattern = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
        pattern = re.compile(r"<(message|tool)>(.*?)</(message|tool)>", re.DOTALL)
        messages = message_pattern.findall(solution)
        tools = tool_pattern.findall(solution)
        blocks = pattern.findall(solution)
        parsed_data = defaultdict(list)
        for msg in messages:
            parsed_data["message"].append(msg.strip())
        for tool in tools:
            tool_name_pattern = re.compile(r"<tool_name>(.*?)</tool_name>", re.DOTALL)
            args_pattern = re.compile(r"<args>(.*?)</args>", re.DOTALL)
            result_pattern = re.compile(r"<result>(.*?)</result>", re.DOTALL)
            cot_pattern = re.compile(r"<cot>(.*?)</cot>", re.DOTALL)
            tool_name = tool_name_pattern.findall(tool)
            args = args_pattern.findall(tool)
            result = result_pattern.findall(tool)
            cot = cot_pattern.findall(tool)
            parsed_data["tool"].append({"tool_name": tool_name, "args": args, "result": result, "cot": cot})
        for block in blocks:
            if block[0] == "message":
                msg = parse_text(block[1].strip())
                parsed_data["all"].append({"type": "message", "content": msg})
            elif block[0] == "tool":
                if "Error in the tool" in block[1]:
                    continue
                # print(block[1])
                tool_name_pattern = re.compile(r"<tool_name>(.*?)</tool_name>", re.DOTALL)
                args_pattern = re.compile(r"<args>(.*?)</args>", re.DOTALL)
                result_pattern = re.compile(r"<result>(.*?)</result>", re.DOTALL)
                tool_name = tool_name_pattern.findall(block[1])[0]
                args = args_pattern.findall(block[1])[0]
                result = result_pattern.findall(block[1])[0]
                parsed_data["all"].append({"type": "tool", "tool_name": tool_name, "args": args, "result": result, "raw_string": block[1]})
        parsed_data_list.append(parsed_data)
    return parsed_data_list

def extract_result_boxed(text: str) -> str:
    r"""Extract content from \\boxed{} environments.

        Args:
            text (str): The input text to process.

        Returns:
            Optional[str]: Content inside \\boxed{} if found, else None.
        """
        # Find the start of the boxed content
    boxed_pattern = "\\boxed{"
    if boxed_pattern not in text:
        logger.debug("No \\boxed{} content found in the response")
        return None

    start_idx = text.find(boxed_pattern) + len(boxed_pattern)
    if start_idx >= len(text):
        logger.debug("Malformed \\boxed{} (no content after opening)")
        return None

    # Use stack-based approach to handle nested braces
    stack = 1  # Start with one opening brace
    end_idx = start_idx
    escape_mode = False

    for i in range(start_idx, len(text)):
        char = text[i]

        # Handle escape sequences
        if escape_mode:
            escape_mode = False
            continue

        if char == '\\':
            escape_mode = True
            continue

        if char == '{':
            stack += 1
        elif char == '}':
            stack -= 1

        if stack == 0:  # Found the matching closing brace
            end_idx = i
            break

    # Check if we found a complete boxed expression
    if stack != 0:
        logger.debug("Unbalanced braces in \\boxed{} content")
        return None

    # Extract the content
    content = text[start_idx:end_idx].strip()
    logger.debug(f"Extracted boxed content: {content}")
    return content


def extract_docstring_from_function(function_code: str) -> str:
    """
    Extract the docstring from a function code string.
    
    Args:
        function_code (str): The function code as a string
        
    Returns:
        str: The docstring if found, otherwise an empty string
    """
    # Regular expression to match triple-quoted docstrings
    docstring_pattern = r'r?"""(.*?)"""'
    match = re.search(docstring_pattern, function_code, re.DOTALL)
    
    if match:
        return match.group(1)  # Return just the content inside the quotes
    return ""

def save_dataframe_to_csv(df, output_path):
    """
    Save a DataFrame to a CSV file with proper handling of newlines and special characters.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to the output CSV file
    """
    import json
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Text columns that need special handling
    text_columns = ['problem_text', 'solution_log', 'enhanced_solution_log', 'enhanced_solution_log_reformatted']
    
    # For text columns, convert to JSON strings to properly handle newlines and special characters
    for col in text_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(
                lambda x: json.dumps(str(x)) if pd.notna(x) else json.dumps("")
            )
    
    # Save to CSV with minimal quoting (only quote text fields)
    df_copy.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
    )
    
    logger.info(f"Saved DataFrame to {output_path} with {len(df_copy)} rows and {len(df_copy.columns)} columns")
