import json
import tensorflow as tf


def create_token_guesser_model(vocabulary_size: int,
                               num_embedding_seq_layers: int, embedding_seq_dim: int,
                               num_embedding_feed_forward_layers: int, embedding_feed_forward_dim: int,
                               num_sequence_encoding_seq_layers: int, sequence_encoding_seq_dim: int,
                               encoder_output_dim: int,
                               num_decoding_feed_forward_layers: int, decoding_feed_forward_dim: int):
    """
    Constructs a deep learning model for predicting masked tokens in sequences of integers.
    Each integer in the sequence is embedded and processed through a series of neural network layers.

    Parameters:
    - vocabulary_size: Size of the output vocabulary; defines the number of classes in the final output.
    - num_embedding_seq_layers: Number of bidirectional LSTM layers to process each integer vector as a sequence.
    - embedding_seq_dim: Dimensionality of the LSTM units in the embedding sequence layers.
    - num_embedding_feed_forward_layers: Number of dense layers applied after the embedding sequence layers.
    - embedding_feed_forward_dim: Dimensionality of the dense layers following the embedding sequence.
    - num_sequence_encoding_seq_layers: Number of bidirectional LSTM layers for encoding the sequence of embedded vectors.
    - sequence_encoding_seq_dim: Dimensionality of the LSTM units in the sequence encoding layers.
    - encoder_output_dim: Dimensionality of the final encoder output, influencing the sequence representation.
    - num_decoding_feed_forward_layers: Number of dense layers in the decoding stage of the model.
    - decoding_feed_forward_dim: Dimensionality of each dense layer in the decoding process.

    Returns:
    - A TensorFlow Keras model that takes batches of integer sequences and outputs a prediction for the vocabulary class of each token.
    """

    with open("integerembeddings/integer_embedding_config.json", "r") as file:
        hyper_parameter_data = json.load(file)

    embedding_size = hyper_parameter_data["embedding_size"]
    mask_token = hyper_parameter_data["mask_token"]

    # Model Inputs
    encoder_inputs = tf.keras.layers.Input(shape=(None, None), name="encoder_inputs")
    mask_inputs = tf.keras.layers.Input(shape=(None,), name="mask_inputs", dtype="bool")

    # --- ENCODING INTEGER VECTORS ---

    # Reshape input to add a channel dimension, making it compatible with sequence models
    x = tf.expand_dims(encoder_inputs, -1)

    # Apply masking to ignore specific tokens during training
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Masking(mask_value=mask_token)
    )(x)

    # A deep sequence to sequence model applied to the integer vectors
    for _ in range(num_embedding_seq_layers):
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=embedding_seq_dim, return_sequences=True),
                merge_mode="concat"),
        )(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=embedding_seq_dim, return_sequences=False),
            merge_mode="concat"),
    )(x)

    # A deep feed forward neural network to further refine integer vectors
    for _ in range(num_embedding_feed_forward_layers):
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=embedding_feed_forward_dim, activation="relu")
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(0.4)
        )(x)

    # Final layer to refine integer vectors into desired embedding size
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=embedding_size, activation="relu")
    )(x)

    # --- ENCODING SEQUENCE OF VECTORS ---

    # Apply sequence-level mask to control which sequences are processed
    x = tf.keras.layers.Lambda(lambda t: t)(x, mask=mask_inputs)

    # Pre-encoding layers to process sequences of embedding vectors
    for _ in range(num_sequence_encoding_seq_layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(sequence_encoding_seq_dim, return_sequences=True),
            merge_mode="concat"
        )(x)

    # Final sequence model to encode entire sequence of embedded integer vectors into desired encoder dim
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(encoder_output_dim, return_sequences=False),
        merge_mode="concat",
    )(x)

    # A deep feed forward neural network to further refine integer vectors
    for _ in range(num_decoding_feed_forward_layers):
        x = tf.keras.layers.Dense(decoding_feed_forward_dim, activation="softplus")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    # Output layer to predict the vocabulary token
    x = tf.keras.layers.Dense(decoding_feed_forward_dim, activation="relu")(x)
    decoding_output = tf.keras.layers.Dense(vocabulary_size, activation="sigmoid")(x)

    # Compile the model
    model = tf.keras.models.Model(inputs=[encoder_inputs, mask_inputs], outputs=decoding_output)

    return model
