# Code from the following article:
# https://link.springer.com/article/10.1007/s10278-018-0079-6

from data import DataSet
from settings import Settings
from utils import enable_gpu_computing
from model import Model

enable_gpu_computing()
settings = Settings(settings_filepath='settings.toml')
dataset = DataSet(settings)

######################################################################
# Build the model
######################################################################

model = Model(settings)

######################################################################
# Image preprocessing and augmentation
######################################################################


train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Rescale pixel values to 0-1 to aid CNN processing
    shear_range=0.2,  # 0-1 range for shearing
    zoom_range=0.2,  # 0-1 range for zoom
    rotation_range=20,  # 0-180 range, degrees of rotation
    width_shift_range=0.2,  # 0-1 range horizontal translation
    height_shift_range=0.2,  # 0-1 range vertical translation
    horizontal_flip=True)  # set True or False

# Rescale pixel values to 0-1 to aid CNN processing
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Defining the training and validation generator
# Class mode is set to 'binary' for a 2-class problem
# Generator randomly shuffles and presents images in batches to the network
train_generator = train_datagen.flow_from_directory(dataset.train_directory_path,
                                                    target_size=(settings.img_height, settings.img_width),
                                                    batch_size=settings.batch_size,
                                                    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(dataset.validation_directory_path,
                                                         target_size=(settings.img_height, settings.img_width),
                                                         batch_size=settings.batch_size,
                                                         class_mode='binary')

######################################################################
# Fitting the model
######################################################################

# Fine-tune the pretrained Inception V3 model using the data generator
# Specify steps per epoch (number of samples/batch_size)
history = model.fit(train_generator,
                    steps_per_epoch=dataset.train_samples_number // settings.batch_size,
                    epochs=settings.epochs,
                    validation_data=validation_generator,
                    validation_steps=dataset.validation_samples_number // settings.batch_size)

######################################################################
# Plotting the model fitting convergence
######################################################################

# import matplotlib library, and plot training cuve
import matplotlib.pyplot as plt

print(history.history.keys())
plt.figure()
plt.plot(history.history['accuracy'], 'orange', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.show()

######################################################################
# Evaluating the model performance
######################################################################

# import numpy and keras preprocessing libraries
import numpy as np
from keras.preprocessing import image

# load, resize, and display test images
img_path = 'Open_I_abd_vs_CXRs/TEST/chest2.png'
img_path2 = 'Open_I_abd_vs_CXRs/TEST/abd2.png'
img = image.load_img(img_path, target_size=((settings.img_width), (settings.img_height)))
img2 = image.load_img(img_path2, target_size=((settings.img_width), (settings.img_height)))
plt.imshow(img)
plt.show()

# convert image to numpy array, so Keras can render a prediction
img = image.img_to_array(img)

# expand array from 3 dimensions (height, width, channels) to 4 dimensions (batch size, height, width, channels)
# rescale pixel values to 0-1
x = np.expand_dims(img, axis=0) * 1. / 255

# get prediction on test image
score = model.predict(x)

print('Predicted:', score, 'Chest X-ray' if score < 0.5 else 'Abd X-ray')
# display and render a prediction for the 2nd image
plt.imshow(img2)
plt.show()
img2 = image.img_to_array(img2)
x = np.expand_dims(img2, axis=0) * 1. / 255
score2 = model.predict(x)
print('Predicted:', score2, 'Chest X-ray' if score2 < 0.5 else 'Abd X-ray')
