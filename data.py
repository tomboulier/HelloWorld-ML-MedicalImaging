from wget import download
from os import system
from os.path import join, abspath, exists
from utils import count_files_in_folder


class DataSet:
    def __init__(self, settings):
        self.url = settings.dataset_url
        self.directory_name = settings.dataset_directory_name
        self.load()

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
