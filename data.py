from wget import download
from os import system


class DataSet:
    def __init__(self, settings):
        self.url = settings.dataset_url

    def load(self):
        # Download dataset from url, and unzip the resulting file
        download(url=self.url, out='dataset.zip')
        system('unzip dataset.zip')
        system('rm dataset.zip')
