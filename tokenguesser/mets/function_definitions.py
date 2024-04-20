import json
import math
import operator
from typing import List, Tuple

build_params_file_path = "mets/math_tree_generation_config.json"

with open(build_params_file_path, "r") as json_file:
    build_params_dict = json.load(json_file)


def is_an_integer(number) -> bool:
    return math.isclose(number, round(number), abs_tol=1e-9)


def safe_factorial(n: int) -> int:
    if n < 0:
        raise ValueError(f"Factorial needs non-negative inputs, received {n=}.")
    if n > 20:
        raise OverflowError(f"Factorial tried to generate a number bigger than 64 bits, received {n=}.")
    if not is_an_integer(n):
        raise TypeError(f"Factorial needs integer inputs, received {n=}")
    return math.factorial(int(n))


def safe_pow(base, exp):
    if exp < 0:
        raise ValueError(f"Power needs non-negative exponents, received {exp=}.")
    return math.pow(base, exp)


def safe_fib(n: int) -> int:
    if not is_an_integer(n):
        raise TypeError(f"Fibonacci needs integer inputs, received {n=}.")
    if n < 0:
        raise ValueError(f"Fibonacci needs non-negative inputs, received {n=}.")
    if n > 90:
        raise OverflowError(f"Fibonacci tried to generate a number bigger than 64 bits, received {n=}.")
    a, b = 0, 1
    for _ in range(int(n)):
        a, b = b, a + b
    return a


def safe_gcd(a: int, b: int) -> int:
    if not (is_an_integer(a) and is_an_integer(b)):
        raise TypeError(f"GCD needs integer inputs, received {a=} and {b=}.")
    return math.gcd(int(a), int(b))


def safe_comb(n: int, k: int) -> int:
    if not (is_an_integer(n) and is_an_integer(k)):
        raise TypeError(f"Combination needs integer inputs, received {n=} and {k=}.")
    if n < 0 or k < 0:
        raise ValueError(f"Combination needs non-negative inputs, received {n=} and {k=}.")
    if n > 1000 or k > 1000:
        raise ValueError(f"Combination tried to generate a number bigger than 64 bits, received {n=} and {k=}.")
    return math.comb(int(n), int(k))


# Define a dictionary mapping function names to their properties
functions = {

    "ADD": {
        "func": operator.add,
        "name": "ADD",
        "arity": 2,
        "complexity": 1.0,
        "output_input_type_map": {
            "Z": [("Z", "Z")],
            "R": [("R", "R"), ("Z", "R"), ("R", "Z")]
        }
    },
    "SUB": {
        "func": operator.sub,
        "name": "SUB",
        "arity": 2,
        "complexity": 1.0,
        "output_input_type_map": {
            "Z": [("Z", "Z")],
            "R": [("R", "R"), ("Z", "R"), ("R", "Z")]
        }
    },
    "MULT": {
        "func": operator.mul,
        "name": "MULT",
        "arity": 2,
        "complexity": 2.0,
        "output_input_type_map": {
            "Z": [("Z", "Z")],
            "R": [("R", "R"), ("Z", "R"), ("R", "Z")]
        }
    },
    "FLOOR_DIV": {
        "func": operator.floordiv,
        "name": "FLOOR_DIV",
        "arity": 2,
        "complexity": 3.0,
        "output_input_type_map": {
            "Z": [("Z", "Z"), ("R", "R"), ("Z", "R"), ("R", "Z")],
            "R": []
        }
    },
    "FIB": {
        "func": safe_fib,
        "name": "FIB",
        "arity": 1,
        "complexity": 3.0,
        "output_input_type_map": {
            "Z": [("Z",), ],
            "R": []
        }
    },
    "POW": {
        "func": safe_pow,
        "name": "POW",
        "arity": 2,
        "complexity": 3.0,
        "output_input_type_map": {
            "Z": [("Z", "Z"), ],
            "R": [("R", "R"), ("Z", "R"), ("R", "Z"), ]
        }
    },
    "FACTORIAL": {
        "func": safe_factorial,
        "name": "FACTORIAL",
        "arity": 1,
        "complexity": 3.0,
        "output_input_type_map": {
            "Z": [("Z",), ],
            "R": []
        }
    },
    "TRIANGULAR": {
        "func": lambda n: n * (n + 1) // 2,
        "name": "TRIANGULAR",
        "arity": 1,
        "complexity": 2.0,
        "output_input_type_map": {
            "Z": [("Z",), ],
            "R": [("R",), ]
        }
    },
}

leaves = {
    "N": {
        "func": lambda x: x,
        "name": "N",
        "output_input_type_map": {
            "Z": [("Z",)],
            "R": [("R",)]
        }
    },
    "NUMBER": {
        "func": lambda x, y: int(str(x) + str(y)),
        "name": "NUMBER",
        "output_input_type_map": {
            "Z": [("Z", "Z")],
            "R": []
        }
    },
    **{
        str(i): {
            "func": lambda *args, i_=i: i_,
            "name": str(i),
            "output_input_type_map": {
                "Z": [("Z",), ("R",)],
                "R": []
            }
        } for i in range(0, 10)
    }
}

tmp = {
}

_function_definitions = dict()
_function_definitions.update(functions)
_function_definitions.update(leaves)
_function_definitions.update(tmp)


class MathematicalFunction:
    def __init__(self, func: callable = None, name: str = None, arity: int = None,
                 complexity: float = None, output_input_type_map: dict = None):
        self._func = func
        self._name = name
        self._arity = arity  # How many arguments a function takes, i.e log_a(b) has arity of 2.
        self._complexity = complexity
        self._output_input_type_map = output_input_type_map
        self._lower_bound = build_params_dict.get("Number Bounds").get("NUMBER_LOWER_BOUND")
        self._upper_bound = build_params_dict.get("Number Bounds").get("NUMBER_UPPER_BOUND")

    def validate_inputs(self, inputs: List[Tuple]):
        results: List = []
        # print(self._name, inputs, end=" ")
        for args in inputs:
            result = self._func(*args)
            if not (self._lower_bound < result < self._upper_bound):
                raise OverflowError(f"result={result} went out of bounds.")
            results.append(result)
        # print(results)
        return results

    def get_arity(self):
        return self._arity

    def get_name(self):
        return self._name

    def get_output_input_type_map(self):
        return self._output_input_type_map

    def __repr__(self):
        return self._name


def get_function_names() -> List[str]:
    return [func_info["name"] for func_info in functions.values()]


def get_all_function_names() -> List[str]:
    return [func_info["name"] for func_info in _function_definitions.values()]


def get_function(name: str) -> MathematicalFunction:
    if name in _function_definitions:
        definition = _function_definitions[name]
        return MathematicalFunction(**definition)
    else:
        raise ValueError(f"Function '{name}' not defined.")


def get_function_definitions():
    return _function_definitions
