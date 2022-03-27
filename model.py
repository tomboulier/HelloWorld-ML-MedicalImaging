from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from utils import enable_gpu_computing
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image as keras_image

enable_gpu_computing()


class Model:
    def __init__(self, settings):
        self.img_width = settings.img_width
        self.img_height = settings.img_height
        self._model = None
        self.build()

    def build(self):
        # build the Inception V3 network, use pretrained weights from ImageNet
        # remove top fully connected layers by include_top=False
        base_model = applications.InceptionV3(weights='imagenet',
                                              include_top=False,
                                              input_shape=(self.img_width, (self.img_height), 3))

        # Add new layers on top of the model
        # build a classifier model to put on top of the convolutional model
        # This consists of a global average pooling layer and a fully connected layer with 256 nodes
        # Then apply dropout and sigmoid activation
        model_top = Sequential()
        model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:],
                                             data_format=None)),
        model_top.add(Dense(256, activation='relu'))
        model_top.add(Dropout(0.5))
        model_top.add(Dense(1, activation='sigmoid'))
        model = KerasModel(inputs=base_model.input, outputs=model_top(base_model.output))
        # Compile model using Adam optimizer with common values and binary cross entropy loss
        # Use low learning rate (lr) for transfer learning
        model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e08, decay=0.0),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self._model = model

    def train(self, dataset, settings):
        history = self._model.fit(dataset.train_generator,
                                  steps_per_epoch=dataset.train_samples_number // settings.batch_size,
                                  epochs=settings.epochs,
                                  validation_data=dataset.validation_generator,
                                  validation_steps=dataset.validation_samples_number // settings.batch_size)

        return History(history)

    def predict(self, image_path: str):
        """Classification of the image given by its path, `image_path`"""
        image = keras_image.load_img(image_path, target_size=(self.img_width, self.img_height))
        array = keras_image.img_to_array(image)  # convert image to numpy array, so Keras can render a prediction
        prediction = Prediction(image)
        # expand array from 3 dimensions (height, width, channels) to 4 dimensions (batch size, height, width, channels)
        # rescale pixel values to 0-1
        x = np.expand_dims(array, axis=0) * 1. / 255
        # get prediction on test image
        prediction.score = self._model.predict(x)

        return prediction


class Prediction:
    def __init__(self, image):
        self.image = image
        self.score = -1

    def plot(self):
        """Plots the image with the predicted score in the title"""
        plt.figure()
        plt.imshow(self.image)
        plt.title(f'Predicted: {self.to_string()}')
        plt.show()
        
    def to_string(self):
        if self.score < 0.5:
            return 'Chest X-ray'
        else:
            return 'Abd X-ray'


class History:
    """A wrapper for the History class of TensorFlow"""

    def __init__(self, history):
        self._history = history

    def plot(self):
        """Plots how the training process occured, with accuracy and loss for test and validation sets."""
        plt.figure()
        plt.plot(self._history.history['accuracy'], 'orange', label='Training accuracy')
        plt.plot(self._history.history['val_accuracy'], 'blue', label='Validation accuracy')
        plt.plot(self._history.history['loss'], 'red', label='Training loss')
        plt.plot(self._history.history['val_loss'], 'green', label='Validation loss')
        plt.legend()
        plt.show()
