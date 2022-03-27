# Code from the following article:
# https://link.springer.com/article/10.1007/s10278-018-0079-6

from data import DataSet
from settings import Settings
from model import Model


def main():
    settings = Settings(settings_filepath='settings.toml')
    dataset = DataSet(settings)
    model = Model(settings)
    history = model.train(dataset, settings)
    history.plot()

    # Evaluating the model performance
    model.predict('Open_I_abd_vs_CXRs/TEST/chest2.png').plot()
    model.predict('Open_I_abd_vs_CXRs/TEST/abd2.png').plot()


if __name__ == '__main__':
    main()
