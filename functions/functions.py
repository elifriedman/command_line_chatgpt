import math
import re

def validate_math_expression(expression):
    # Regex pattern to match valid math expressions
    pattern = r'^[\d\s\(\)\+\-\*\/\%\^\.]+|math\.[a-zA-Z_]+\(.*\)$'

    # Check if the expression matches the pattern
    match = re.match(pattern, expression)

    return bool(match)


def calculator(expression=None):
    if not validate_math_expression(expression):
        raise ValueError(f"'{expression}' might be dangerous! If not, then improve your code")
    return eval(expression)

