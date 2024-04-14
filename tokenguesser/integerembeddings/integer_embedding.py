import copy
import json
import random

import numpy as np
import tensorflow as tf

from integerembeddings.static_embedding_training import number_to_base_n_vector


class IntegerSequenceEmbeddingPreprocessor:
    """Maps a batch of sequences of integer representations to a batch of sequences of embeddings."""

    def __init__(self, load_model: bool = True):
        with open("integerembeddings/integer_embedding_config.json", "r") as file:
            hyper_parameter_data = json.load(file)
        weights = np.load("integerembeddings/static_embedding_encoder_weights.npy", allow_pickle=True)

        self.embedding_size = hyper_parameter_data["embedding_size"]
        self.mask_token = hyper_parameter_data["mask_token"]
        self.base = hyper_parameter_data["base"]
        if load_model:
            self.mask_layer = tf.keras.layers.Masking(self.mask_token)
            inner_layer = tf.keras.layers.SimpleRNN(units=self.embedding_size, return_sequences=False)
            inner_layer.set_weights(weights)
            self.encoder_layer = tf.keras.layers.Bidirectional(inner_layer, merge_mode="sum")

    def _unpadded_batch_sequence_integer_to_padded_batch_sequence_vectors(self, bsi):
        """
        :param bsi: Batch sequences of integers.
        :return: Batch sequences of vectors, all padded.
        """

        # Build the batch of padded sequences of integers
        bpsi, mask = self.pad_sequence_of_integers(bsi)

        # Build the batch of padded sequences of vectors
        bpsv, longest_vector_length = self.convert_integers_to_vectors(bpsi)

        # Build the batch of padded sequences of padded vectors
        bpspv = self.pad_integer_representations(bpsv, longest_vector_length)

        # Convert to friendly tensor
        bpspv = tf.convert_to_tensor(bpspv, dtype=tf.float64)
        return bpspv, mask

    @staticmethod
    def pad_sequence_of_integers(batch_of_sequences_of_integers):
        longest_sequence_length = max(len(sequence) for sequence in batch_of_sequences_of_integers)
        bpsi = tf.keras.preprocessing.sequence.pad_sequences(batch_of_sequences_of_integers,
                                                             maxlen=longest_sequence_length,
                                                             padding="post",
                                                             value=float("inf"),
                                                             dtype="float64")
        mask = tf.constant(bpsi == float("inf"), dtype=tf.bool)
        return bpsi, mask

    def convert_integers_to_vectors(self, batch_of_padded_sequences_of_integers, *, base=None):
        bpsv = []
        base_n = self.base if base is None else base
        longest_vector_length = float("-inf")
        for padded_sequence in batch_of_padded_sequences_of_integers:
            sequence_of_vectors = []
            for integer in padded_sequence:
                vector_representation = number_to_base_n_vector(integer, base_n)
                longest_vector_length = max(longest_vector_length, len(vector_representation))
                sequence_of_vectors.append(vector_representation)
            bpsv.append(sequence_of_vectors)
        return bpsv, longest_vector_length

    def pad_integer_representations(self, batch_of_padded_sequences_of_vectors, longest_vector_length, *,
                                    mask_token=None):
        new_batch = copy.deepcopy(batch_of_padded_sequences_of_vectors)
        pad_val = self.mask_token if mask_token is None else mask_token
        for i, padded_sequence in enumerate(new_batch):
            new_batch[i] = (
                tf.keras.preprocessing.sequence.pad_sequences(padded_sequence,
                                                              maxlen=longest_vector_length,
                                                              padding="post",
                                                              value=pad_val,
                                                              dtype="float64")
            )
        return new_batch

    def _apply_embedding(self, inputs):
        batch_size, sequence_length, integer_vector_length = inputs.shape
        x = tf.reshape(inputs, (batch_size * sequence_length, integer_vector_length, 1))

        # Mask the inner values of the vector representations
        x = self.mask_layer(x)

        # Apply our encoder to each vector representation to get its embedding
        x = self.encoder_layer(x)

        x = tf.reshape(x, (batch_size, sequence_length, self.embedding_size))
        return x

    def preprocess(self, X, *, dynamic_training=False):
        X_padded, mask = self._unpadded_batch_sequence_integer_to_padded_batch_sequence_vectors(X)
        if dynamic_training:
            return X_padded, mask
        X_embedded = self._apply_embedding(X_padded)
        return X_embedded, mask


if __name__ == "__main__":
    random.seed(0)
    pp = IntegerSequenceEmbeddingPreprocessor()

    X = [[random.randint(-20, 20) for _ in range(random.randint(1, 10))] for _ in range(3)]
    print(X)

    X_1 = pp.preprocess(X)
    print(X_1)
