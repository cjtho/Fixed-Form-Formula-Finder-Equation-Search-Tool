"""
Motivation is to 'ask' many distinguishing questions about an integer,
then train an encoder such that it can guess the correct answer.
"""

import json
import math
import random
from typing import List

import numpy as np
import sympy
import tensorflow as tf


def number_to_base_n_vector(number: int, base: int, normalize: bool = True) -> List[float]:
    digits: List[float] = []

    # Be on the safe side
    if isinstance(number, int):
        pass
    elif isinstance(number, float) and number.is_integer():
        pass
    else:
        return digits

    # Apply sign bit
    sign_bit = 1.0 if number < 0 else 0.0

    # Special case
    if number == 0:
        return [sign_bit, 0.0]

    number = abs(number)
    while number:
        number, digit = divmod(number, base)
        digits.append(digit)

    # Min-Max scaling
    if normalize:
        min_digit, max_digit = 0, base - 1
        digits = [(digit - min_digit) / (max_digit - min_digit) for digit in digits]

    return [sign_bit] + digits


def log_scaling(n: int) -> float:
    n = abs(n)
    return math.log10(n + 1)


def sign(n: int) -> int:
    if n == 0:
        return 0
    if n < 0:
        return 1
    if n > 0:
        return 2


def distance_to_nearest_prime(n: int) -> int:
    n = abs(n)
    if n <= 2:
        return abs(n - 2)
    elif sympy.isprime(n):
        return 0
    else:
        distance_below = n - sympy.prevprime(n)
        distance_above = sympy.nextprime(n) - n
        return min(distance_below, distance_above)


def distance_to_nearest_fibonacci(n: int) -> float:
    n = abs(n)
    i = 0
    while True:
        fib_num = sympy.fibonacci(i)
        if fib_num >= n:
            break
        i += 1
    distance_below = n - sympy.fibonacci(i - 1) if i > 0 else float('inf')
    distance_above = fib_num - n
    return math.log10(min(distance_below, distance_above) + 1)


def distance_to_nearest_triangular(n: int) -> float:
    n = abs(n)
    x = (-1 + math.sqrt(1 + 8 * n)) / 2
    x_int = int(x)
    nearest_below = x_int * (x_int + 1) // 2
    nearest_above = (x_int + 1) * (x_int + 2) // 2
    return math.log10(min(n - nearest_below, nearest_above - n) + 1)


def distance_to_nearest_square(n: int) -> float:
    n = abs(n)
    sqrt_n = math.isqrt(n)
    nearest_below = sqrt_n ** 2
    nearest_above = (sqrt_n + 1) ** 2
    return math.log10(min(n - nearest_below, nearest_above - n) + 1)


def reversed_digits_log_scaling(n: int) -> float:
    n = abs(n)
    n = int(str(n)[::-1])
    return math.log10(n + 1)


def leading_digit(n: int) -> int:
    return n % 10


def sum_digits(n: int) -> int:
    n = abs(n)
    return sum(map(int, str(n)))


def inverse_log_scaling(n: int) -> int:
    n = abs(n)
    return 10 ** n - 1


def generate_dataset(amount: int, mask_token: float, base: int, property_config: List[dict]):
    X, y_labels = [], {prop["name"]: [] for prop in property_config}

    for _ in range(amount):
        sign_ = random.choice([-1, 1])
        magnitude = random.randint(0, 2 ** random.randint(0, 63))  # log random distribution
        number_rand = sign_ * magnitude
        integer_representation = number_to_base_n_vector(number_rand, base, normalize=True)
        X.append(integer_representation)
        for prop in property_config:
            y_labels[prop["name"]].append(prop["generate_label"](number_rand))

    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max(len(x) for x in X), value=mask_token,
                                                      dtype="float64", padding="post")
    X = tf.convert_to_tensor(X, dtype="float64")
    X = tf.reshape(X, (*X.shape, 1))

    for prop in property_config:
        y_data = tf.convert_to_tensor(y_labels[prop["name"]], dtype="float64")
        y_labels[prop["name"]] = tf.reshape(y_data, (*y_data.shape, 1))

    return X, y_labels


def build_model(mask_token: float, embedding_size: int, property_config: List[dict]):
    # Inputs & Masking
    inputs = tf.keras.layers.Input(shape=(None, 1))
    x = tf.keras.layers.Masking(mask_value=mask_token)(inputs)

    # Encoding
    encoder_layer = tf.keras.layers.SimpleRNN(units=embedding_size, return_sequences=False)
    encoding = tf.keras.layers.Bidirectional(encoder_layer, merge_mode="sum")(x)

    # Decoding (Complexity)
    x = tf.keras.layers.Dense(units=128, activation="relu")(encoding)

    # Decoding (Outputs)
    outputs = []
    losses = {}
    loss_weights = {}
    for prop in property_config:
        output = tf.keras.layers.Dense(units=prop["units"], activation=prop["activation"], name=prop["name"])(x)
        outputs.append(output)
        losses[prop["name"]] = prop["loss"]
        loss_weights[prop["name"]] = prop["loss_weight"]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model, losses, loss_weights, encoder_layer


def main():
    # Hyper Parameters
    with open("integer_embedding_config.json", "r") as file:
        hyper_parameter_data = json.load(file)
    embedding_size = hyper_parameter_data["embedding_size"]
    mask_token = hyper_parameter_data["mask_token"]
    base = hyper_parameter_data["base"]

    # Training Parameters
    equalize_loss_weight: bool = False
    attempts_to_improve = 25
    intra_epochs = 10
    training_size_per_epoch = 10_000
    batch_size = 1024

    property_config = [
        {
            "name": "magnitude_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 1.5,
            "generate_label": log_scaling
        },
        {
            "name": "sign_output",
            "units": 3,
            "activation": "softmax",
            "loss": "sparse_categorical_crossentropy",
            "loss_weight": 1.0,
            "generate_label": sign
        },
        {
            "name": "distance_to_nearest_prime_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 0.25,
            "generate_label": distance_to_nearest_prime
        },
        {
            "name": "distance_to_nearest_fibonacci_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 1.0,
            "generate_label": distance_to_nearest_fibonacci
        },
        {
            "name": "distance_to_nearest_triangular_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 1.0,
            "generate_label": distance_to_nearest_triangular
        },
        {
            "name": "distance_to_nearest_square_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 1.0,
            "generate_label": distance_to_nearest_square
        },
        {
            "name": "reversed_digits_magnitude_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 1.0,
            "generate_label": reversed_digits_log_scaling
        },
        {
            "name": "leading_digit_output",
            "units": 10,
            "activation": "softmax",
            "loss": "sparse_categorical_crossentropy",
            "loss_weight": 1.0,
            "generate_label": leading_digit
        },
        {
            "name": "sum_digits_output",
            "units": 1,
            "activation": "relu",
            "loss": "mean_squared_error",
            "loss_weight": 1.0,
            "generate_label": sum_digits
        },
    ]

    if equalize_loss_weight:
        for prop in property_config:
            prop["loss_weight"] = 1.0

    # Normalize loss weights
    total_loss_weight = sum(prop["loss_weight"] for prop in property_config)
    for prop in property_config:
        prop["loss_weight"] /= total_loss_weight

    print("\n>> Sample Data\n")
    print(generate_dataset(5, mask_token, base, property_config))

    # Create model
    model, losses, loss_weights, encoder_layer = build_model(mask_token, embedding_size, property_config)

    # Building
    model.build(input_shape=(None, 1))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses, loss_weights=loss_weights)

    # Fitting
    print("\n>> Training\n")
    best_loss = np.inf
    attempts_remaining = attempts_to_improve
    while attempts_remaining > 0:
        X_train, y_train = generate_dataset(training_size_per_epoch, mask_token, base, property_config)
        history = model.fit(X_train, y_train, epochs=intra_epochs, batch_size=batch_size, verbose=0)
        new_loss = history.history["loss"][-1]
        if new_loss < best_loss:
            best_loss = new_loss
            attempts_remaining = attempts_to_improve
        else:
            attempts_remaining -= 1
        print(f">> ({attempts_remaining:2}) Current Loss: {new_loss:10.4f} "
              f"{'<' if new_loss <= best_loss else '>'} Best Loss: {best_loss:10.4f}")

    # Testing
    print("\n>> Prediction\n")
    X_test, y_test = generate_dataset(3, mask_token, base, property_config)
    y_pred = model.predict(X_test)

    for index, prop in enumerate(property_config):
        actual_values = y_test[prop["name"]].numpy().flatten()
        predicted_values = y_pred[index]
        print()
        print(f"--- {prop['name']} ({prop['activation']}) ---".center(65))
        print()
        for actual, predicted in zip(actual_values, predicted_values):
            print(f"Actual: {actual}, Predicted: {np.argmax(predicted), np.max(predicted)}")

    weights = encoder_layer.get_weights()
    np.save("static_embedding_encoder_weights.npy", weights)


if __name__ == "__main__":
    main()
