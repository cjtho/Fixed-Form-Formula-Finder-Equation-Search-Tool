import json
import tensorflow as tf


def create_token_guesser_model(vocabulary_size: int,
                               sequence_encoding_layers: int,
                               sequence_encoding_dim: int):
    with open("integerembeddings/integer_embedding_config.json", "r") as file:
        hyper_parameter_data = json.load(file)

    embedding_size = hyper_parameter_data["embedding_size"]
    mask_token = hyper_parameter_data["mask_token"]

    # Model inputs
    encoder_inputs = tf.keras.layers.Input(shape=(None, None), name="encoder_inputs")
    mask_inputs = tf.keras.layers.Input(shape=(None,), name="mask_inputs", dtype="bool")

    # --- SCOPE OF INTEGER VECTORS ---

    # Reshape input to add a channel dimension, making it compatible with sequence models
    x = tf.expand_dims(encoder_inputs, -1)

    # Apply masking of the base-N integer vectors
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Masking(mask_value=mask_token)
    )(x)

    # Process base-N integer vectors sequentially
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=embedding_size, return_sequences=False),
            merge_mode="concat"),
    )(x)

    # Final layer to refine integer vectors into desired embedding
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=embedding_size, activation="relu")
    )(x)

    # --- SCOPE OF SEQUENCE OF INTEGER EMBEDDINGS ---

    # Apply sequence-level mask to control which sequences are processed
    x = tf.keras.layers.Lambda(lambda t: t)(x, mask=mask_inputs)

    # Process the entire sequence of embeddings
    for _ in range(sequence_encoding_layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(sequence_encoding_dim, return_sequences=True),
            merge_mode="concat",
        )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(sequence_encoding_dim, return_sequences=False),
        merge_mode="concat",
    )(x)

    # Output layer to predict the vocabulary token
    x = tf.keras.layers.Dense(sequence_encoding_dim, activation="linear")(x)
    decoding_output = tf.keras.layers.Dense(vocabulary_size, activation="sigmoid")(x)

    # Construct the model
    model = tf.keras.models.Model(inputs=[encoder_inputs, mask_inputs], outputs=decoding_output)

    # Build the model
    model.build(input_shape=(None, None))

    return model
