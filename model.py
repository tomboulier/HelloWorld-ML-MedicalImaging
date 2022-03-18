from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam


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
        return self._model.fit(dataset.train_generator,
                               steps_per_epoch=dataset.train_samples_number // settings.batch_size,
                               epochs=settings.epochs,
                               validation_data=dataset.validation_generator,
                               validation_steps=dataset.validation_samples_number // settings.batch_size)

    def predict(self, image):
        return self._model.predict(image)
