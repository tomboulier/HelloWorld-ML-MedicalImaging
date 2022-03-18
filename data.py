from wget import download
from os import system
from os.path import join, abspath, exists
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import count_files_in_folder


class DataSet:
    def __init__(self, settings):
        self.url = settings.dataset_url
        self.directory_name = settings.dataset_directory_name
        self.load()
        self.image_width = settings.img_width
        self.image_height = settings.img_height
        self.image_generator = ImageGenerator(settings)

    def load(self):
        """
            Download dataset from url (if not already done), then unzip the resulting file
        """
        if exists(self.directory_name):
            return
        download(url=self.url, out='dataset.zip')
        system('unzip dataset.zip')
        system('rm dataset.zip')

    @property
    def directory_path(self):
        return abspath(self.directory_name)

    @property
    def train_directory_path(self):
        return join(self.directory_path, 'TRAIN')

    @property
    def validation_directory_path(self):
        return join(self.directory_path, 'VAL')

    @property
    def train_samples_number(self):
        return count_files_in_folder(self.train_directory_path)

    @property
    def validation_samples_number(self):
        return count_files_in_folder(self.validation_directory_path)

    @property
    def train_generator(self):
        return self.image_generator.flow_from_directory(self.train_directory_path)

    @property
    def validation_generator(self):
        return self.image_generator.flow_from_directory(self.validation_directory_path)


class ImageGenerator(ImageDataGenerator):
    """
    A data generator aims at having a bigger dataset without having
    to load everything in RAM.
    See this article: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3

    Here, the idea is to have a bigger image set with image augmentation
    (see https://towardsdatascience.com/image-augmentation-for-deep-learning-histogram-equalization-a71387f609b2)

    TODO: use this article to get our architecture cleaner?
    """
    def __init__(self, settings):
        super().__init__(
            rescale=settings.rescale,
            shear_range=settings.shear_range,
            zoom_range=settings.zoom_range,
            rotation_range=settings.rotation_range,
            width_shift_range=settings.width_shift_range,
            height_shift_range=settings.height_shift_range,
            horizontal_flip=settings.horizontal_flip)
        self.img_height = settings.img_height
        self.img_width = settings.img_width
        self.batch_size = settings.batch_size

    def build_from_directory(self, directory_path):
        # Class mode is set to 'binary' for a 2-class problem
        # Generator randomly shuffles and presents images in batches to the network
        return self.flow_from_directory(directory_path,
                                        target_size=(self.img_height, self.img_width),
                                        batch_size=self.batch_size,
                                        class_mode='binary')
