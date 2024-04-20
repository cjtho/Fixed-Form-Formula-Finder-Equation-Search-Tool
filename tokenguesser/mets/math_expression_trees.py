import json
import logging
import math
import random
import time
from typing import List, Dict, Tuple, Union

from graphviz import Digraph

from .function_definitions import MathematicalFunction, get_function, get_function_definitions

build_params_file_path = "mets/math_tree_generation_config.json"

with open(build_params_file_path, "r") as json_file:
    build_params_dict = json.load(json_file)


class Node:
    def __init__(self, math_func: MathematicalFunction = None, children: List['Node'] = None, parent: 'Node' = None):
        self._math_func = math_func
        self._children = children if children is not None else []
        self._parent = parent

    def validate_tree(self, values: List[Tuple]) -> List:
        # If the function is a leaf (no children), it directly applies to the values.
        if not self.get_children():
            return self.get_math_func().validate_inputs(values)

        # For non-leaf nodes, collect results from children and apply this node's function.
        child_results: List[List] = []
        for child in self.get_children():
            child_result = child.validate_tree(values)
            child_results.append(child_result)

        result = self.get_math_func().validate_inputs(list(zip(*child_results)))

        # Pruning.
        is_node_equivalent_to_integer: bool = is_an_integer(result[0]) and all(x == result[0] for x in result)
        if is_node_equivalent_to_integer:
            self.replace(digit_tree(round(result[0])))
        else:
            for child, child_result in zip(self.get_children(), child_results):
                if all(math.isclose(x, y, abs_tol=1e-8) for x, y in zip(child_result, result)):
                    # self failed to modify the result after the transformation; might as well replace with child
                    child.set_parent(None)
                    self.replace(child)
                    break

        return result

    def get_math_func(self) -> MathematicalFunction:
        return self._math_func

    def set_math_func(self, func: callable) -> None:
        self._math_func = func

    def get_children(self) -> List['Node']:
        return self._children

    def set_children(self, children: List['Node']) -> None:
        for child in children:
            child.set_parent(self)
        self._children = children

    def get_parent(self) -> 'Node':
        return self._parent

    def set_parent(self, parent: 'Node') -> None:
        self._parent = parent

    def replace(self, node: 'Node') -> None:
        # Assign the new node's parent and children.
        node.set_parent(self.get_parent())

        if self.get_parent():  # If the node isn't root
            # Replace the old node in the parent's children list with the new node.
            index = self.get_parent().get_children().index(self)
            self.get_parent().get_children()[index] = node
        else:
            self.set_math_func(node.get_math_func())
            self.set_children(node.get_children())

    def get_prefix_traversal_tokens(self) -> List[str]:
        tokens = [self.get_math_func().get_name()]
        for child in self.get_children():
            tokens.extend(child.get_prefix_traversal_tokens())
        return tokens

    def is_valid_token(self):
        return self.get_math_func().get_arity() is not None

    def get_terms_count(self) -> int:
        count = 0
        if self.is_valid_token():
            count += 1
        for child in self.get_children():
            count += child.get_terms_count()
        return count

    def __repr__(self):
        return f"{self.get_prefix_traversal_tokens()}"


class FunctionSampler:
    def __init__(self):
        self.function_definitions = get_function_definitions()

    def get_function_of_type(self, return_type: str) -> MathematicalFunction:
        candidates = {name: defn for name, defn in self.function_definitions.items()
                      if defn["output_input_type_map"].get(return_type) and defn.get("arity")}
        chosen_function_name = random.choices(list(candidates.keys()))[0]
        return get_function(chosen_function_name)


def get_input_sequence(complexity: float):
    sequence_start_value = build_params_dict.get("Sequence Bounds").get("SEQUENCE_START_VALUE")
    min_length = build_params_dict.get("Sequence Bounds").get("SEQUENCE_MIN_LENGTH")
    max_length = build_params_dict.get("Sequence Bounds").get("SEQUENCE_MAX_LENGTH")
    length = int(linear_interpolate(min_length, max_length, complexity))
    return [(i,) for i in range(sequence_start_value, length)]


def get_terms_count(complexity: float):
    min_terms = build_params_dict.get("Terms Bounds").get("MIN_TERMS")
    max_terms = build_params_dict.get("Terms Bounds").get("MAX_TERMS")
    terms_count = int(linear_interpolate(min_terms, max_terms, complexity))
    return terms_count


def linear_interpolate(a, b, t):
    return a + (b - a) * t


def is_an_integer(number) -> bool:
    return math.isclose(number, round(number), abs_tol=1e-9)


def digit_function(digit: int):
    return get_function(str(digit))


def number_function():
    return get_function("NUMBER")


def identity_function():
    return get_function("N")


def digit_tree(number: Union[List[int], int]) -> Node:
    if isinstance(number, int):
        is_negative: bool = number < 0
        numbers = list(map(int, str(abs(number))))
    else:
        is_negative = False  # because it only receives lists from the randomizer
        numbers = number

    numbers.reverse()
    root = Node()
    dummy = root

    if is_negative:  # Construct 0 - number
        root.set_math_func(get_function("SUB"))
        left_child = Node()
        left_child.set_math_func(get_function("0"))
        right_child = Node()
        root.set_children([left_child, right_child])
        root = right_child

    current_node = root
    for i, digit in enumerate(numbers):
        if i >= len(numbers) - 1:
            current_node.set_math_func(digit_function(digit))
        else:
            current_node.set_math_func(number_function())
            left_child = Node()
            right_child = Node(math_func=digit_function(digit))
            current_node.set_children([left_child, right_child])
            current_node = left_child

    return dummy


def random_digit_tree():
    if random.random() < build_params_dict.get("Digit Tree Information").get("PROBABILITY_OF_NEGATIVE_CONSTANT"):
        return digit_tree(-1)

    random_digits: List[int] = [random.randint(1, 9)]
    i = 1
    while (
            (random.random() < build_params_dict.get("Digit Tree Information").get("PROBABILITY_OF_ADD_DIGIT"))
            and i < build_params_dict.get("Digit Tree Information").get("MAXIMUM_DIGITS")
    ):
        random_digits.append(random.randint(0, 9))
        i += 1
    return digit_tree(random_digits)


def generate_tree(functions: FunctionSampler, function_terms: int):
    root = Node()
    initial_desired_outputs = ["Z"]
    frontier = [(root, initial_desired_outputs)]

    # Set functions nodes.
    while function_terms > 0:
        # Randomly select a node from the frontier to work on
        idx = random.randint(0, len(frontier) - 1)
        node, desired_outputs = frontier[idx]
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        frontier.pop()

        # Choose a function that can produce a desired output type and assign it to the node
        desired_output = random.choice(desired_outputs)
        func = functions.get_function_of_type(desired_output)
        node.set_math_func(func)
        function_terms -= 1

        # Prepare children nodes based on the arity of the selected function
        arity = func.get_arity()
        children = [Node() for _ in range(arity)]
        node.set_children(children)
        possible_inputs = func.get_output_input_type_map().get(desired_output)
        possible_inputs = list(zip(*possible_inputs))
        for child, possible_input in zip(children, possible_inputs):
            frontier.append((child, list(set(possible_input))))

    # Set leaves.
    variable_is_placed: bool = False
    for idx, (node, _) in enumerate(frontier):
        is_idx_last_and_not_first = (0 < idx >= len(frontier) - 1)
        if is_idx_last_and_not_first and (not variable_is_placed):
            node.set_math_func(identity_function())
        elif random.random() < build_params_dict.get("Digit Tree Information").get("PROBABILITY_OF_DIGIT_TREE"):
            rdt = random_digit_tree()
            node.replace(rdt)
        else:
            node.set_math_func(identity_function())
            variable_is_placed = True

    return root


def visualize_tree(tree: Node, graph=None, node_id=0):
    if graph is None:
        graph = Digraph()
        graph.node(str(node_id), label=str(tree.get_math_func()))

    current_node_id = node_id
    if tree.get_children():
        for i, child in enumerate(tree.get_children()):
            child_node_id = node_id * 10 + i + 1  # Unique ID for each child
            graph.node(str(child_node_id), label=str(child.get_math_func()))
            graph.edge(str(current_node_id), str(child_node_id))
            visualize_tree(child, graph, child_node_id)

    if node_id == 0:
        graph.render(filename="math_tree.dv",
                     directory="math_visualisations",
                     format="pdf",
                     view=True)


def generate_sequence_function_pair(visualize: bool = False,
                                    debugging_mode: bool = False,
                                    *,
                                    seq_complexity: float = 0.0,
                                    term_complexity: float = 0.0, ):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Adjust logging level based on debugging_mode
    if not debugging_mode:
        logging.disable(logging.CRITICAL)

    outputs = tree = None

    function_handler = FunctionSampler()
    inputs = get_input_sequence(seq_complexity)
    function_terms_count = get_terms_count(term_complexity)

    logging.debug(
        f"Sequence length: {len(inputs)}, Terms length: {function_terms_count}")
    logging.debug(f"Initial sequence: {inputs}")

    solution_found = False
    attempt = 0
    while not solution_found:
        attempt += 1
        tree = generate_tree(function_handler, function_terms_count)
        try:
            outputs = tree.validate_tree(inputs)  # trims tree and modifies its structure in place
            terms_count = tree.get_terms_count()
            terms_lower_bound = (build_params_dict.get("Terms Bounds").get("TERMS_COUNT_TOLERANCE")
                                 * function_terms_count)

            is_enough_terms: bool = terms_count >= terms_lower_bound
            if not is_enough_terms:
                raise ValueError(f"Not enough terms: {tree.get_terms_count()} / {function_terms_count}")

            if any(not is_an_integer(output) for output in outputs):
                raise TypeError(f"Output was not integer: {outputs}")

            logging.debug(f"Successful sequence generated on attempt {attempt}: {outputs}")
            solution_found = True
            break
        except (ZeroDivisionError, TypeError, OverflowError, ValueError) as e:
            logging.debug(f"Failed attempt {attempt} because: {e}")

    if solution_found:
        inputs = [a for a, *_ in inputs]
        if visualize:
            visualize_tree(tree)
    else:
        raise Exception("Failed to find a solution within the specified attempts.")

    outputs = [int(output) for output in outputs]
    return inputs, outputs, tree
