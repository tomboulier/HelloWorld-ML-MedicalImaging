"""The classifier: Inception V3 pre-trained on ImageNet, with a new binary head.

The architecture is the one of the original notebook. Two things moved inside
the model so that it receives raw images in ``[0, 255]``: the rescaling to
``[0, 1]`` and the data augmentation, which Keras only applies during training.
"""

from __future__ import annotations

import keras

from hello_world_medical_imaging.config import Augmentation


def augmentation_layers(augmentation: Augmentation) -> keras.Sequential:
    """Random transformations of the training images, from the configured ranges.

    >>> layers = augmentation_layers(Augmentation(horizontal_flip=False))
    >>> len(layers.layers)
    4
    """
    layers: list[keras.Layer] = [
        keras.layers.RandomRotation(augmentation.rotation_degrees / 360),
        keras.layers.RandomZoom(augmentation.zoom_range),
        keras.layers.RandomTranslation(augmentation.shift_range, augmentation.shift_range),
        keras.layers.RandomShear(augmentation.shear_range),
    ]
    if augmentation.horizontal_flip:
        layers.append(keras.layers.RandomFlip("horizontal"))
    return keras.Sequential(layers, name="augmentation")


def build_model(
    image_size: int,
    learning_rate: float,
    augmentation: Augmentation | None = None,
    pretrained: bool = True,
) -> keras.Model:
    """Build and compile the transfer-learning classifier.

    Args:
        image_size: Side of the square input images, at least 75 pixels.
        learning_rate: Learning rate of the Adam optimizer (kept low for transfer learning).
        augmentation: Random transformations applied to training images, if any.
        pretrained: Load ImageNet weights (downloaded once by Keras). ``False`` gives
            random weights, which is what the tests use to stay offline and fast.

    Returns:
        A compiled model that outputs the probability of the abdomen class.
    """
    base_model = keras.applications.InceptionV3(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    if augmentation is not None:
        x = augmentation_layers(augmentation)(x)
    x = base_model(x)
    # Classifier on top of the convolutional base: global average pooling,
    # a fully connected layer with 256 nodes, dropout and a sigmoid output.
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="abdomen_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="hello_world_medical_imaging")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
