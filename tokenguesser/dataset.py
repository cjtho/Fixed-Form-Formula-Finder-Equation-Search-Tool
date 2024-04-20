import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from mets.math_expression_trees import generate_sequence_function_pair


def build_sub_dataset(desired_size: int, complexity: float, vocabulary) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
    seen = set()
    stagnation_count = 0
    term_complexity = complexity
    while len(seen) < desired_size:
        # Generate our raw data
        _, sequence_of_integers, tree = generate_sequence_function_pair(
            seq_complexity=random.uniform(0, 1),
            term_complexity=term_complexity
        )

        # Generate token existence vector
        token_list = tree.get_prefix_traversal_tokens()
        tokens_counts = [0] * len(vocabulary)
        for token in token_list:
            if token in vocabulary:
                tokens_counts[vocabulary[token]] = 1

        # Add our data
        sequence_of_integers, tokens_counts = map(tuple, (sequence_of_integers, tokens_counts))
        prev_length = len(seen)
        seen.add((sequence_of_integers, tokens_counts))
        new_length = len(seen)

        # If struggling to find new data, up the complexity
        if new_length <= prev_length:
            stagnation_count += 1
        else:
            stagnation_count = 0

        if stagnation_count >= 100:
            term_complexity = np.clip(term_complexity + 0.01, 0, 1)
            stagnation_count = 0

    X = []
    y = []
    for sequence_of_integers, tokens_counts in seen:
        X.append(sequence_of_integers)
        y.append(tokens_counts)

    return X, y


def build_dataset(desired_sizes: List[int], complexities: List[float], vocabulary, preprocessor):
    X = []
    y = []
    for desired_size, complexity in zip(desired_sizes, complexities):
        X_, y_ = build_sub_dataset(desired_size, complexity, vocabulary)
        X.extend(X_)
        y.extend(y_)

    # Preprocess data into padded blocks
    X, mask = preprocessor.preprocess(X, use_static_embedding=False)

    # Convert to tensors
    X = tf.convert_to_tensor(X)
    mask = tf.convert_to_tensor(mask)
    y = tf.convert_to_tensor(y)

    # X = tf.reshape(X, (*X.shape, 1))
    return [X, mask], y
