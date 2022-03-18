from data import DataSet
from settings import Settings
from pytest import approx

test_settings = Settings(settings_filepath='test_settings.toml')
expected_url = 'https://raw.githubusercontent.com/paras42/Hello_World_Deep_Learning/master/Open_I_abd_vs_CXRs.zip'
dataset = DataSet(test_settings)

def test_if_Settings_class_works():
    assert test_settings.dataset_url == expected_url
    assert test_settings.rescale == approx(1./255)
    assert test_settings.horizontal_flip

def test_if_dataset_has_same_url_as_in_settings():
    assert dataset.url == expected_url

def test_train_and_validation_samples_number_properties():
    assert dataset.train_samples_number == 65
    assert dataset.validation_samples_number == 10
