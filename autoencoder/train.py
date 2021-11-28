import keras
from keras.datasets import mnist
import numpy as np
import model


class Train:
    def __init__(self):
        (self.x_train, _), (self.x_test, _) = mnist.load_data()

        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.x_train = self.x_train.reshape((len(self.x_train), np.prod(self.x_train.shape[1:])))
        self.x_test = self.x_test.reshape((len(self.x_test), np.prod(self.x_test.shape[1:])))

        print(self.x_train.shape)
        print(self.x_test.shape)

        self.autoencoder = model.AutoEncoder(28 * 28, 32).build_model()

    def train_model(self):
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.autoencoder.fit(self.x_train, self.x_train,
                             epochs=50,
                             batch_size=256,
                             shuffle=True,
                             validation_data=(self.x_test, self.x_test))


if __name__ == "__main__":
    Train().train_model()
