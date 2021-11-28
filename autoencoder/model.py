import keras
from keras import layers


class AutoEncoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def build_model(self):
        input_img = keras.Input(shape=(28 * 28,))

        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(28 * 28, activation='sigmoid')(encoded)

        # This model maps an input to its reconstruction
        autoencoder = keras.Model(input_img, decoded)
        encoder = keras.Model(input_img, encoded)

        # This is our encoded (32-dimensional) input
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        # Retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # Create the decoder model
        decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

        return autoencoder