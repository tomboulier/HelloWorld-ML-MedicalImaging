# This module is a wrapper of Dynaconf
# see https://www.dynaconf.com/ on how to use Dynaconf
from dynaconf import Dynaconf


class Settings(Dynaconf):
    def __init__(self, settings_filepath):
        super().__init__(settings_files=settings_filepath)
