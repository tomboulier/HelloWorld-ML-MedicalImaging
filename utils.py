from os import walk
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def enable_gpu_computing():
    """
    Compatibility for GPU computing
    See this issue: https://stackoverflow.com/questions/59340465/how-to-solve-no-algorithm-worked-keras-error
    """
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)


def count_files_in_folder(directory):
    """Count the number of files in a directory"""

    total = 0

    for root, dirs, files in walk(directory):
        total += len(files)

    return total
