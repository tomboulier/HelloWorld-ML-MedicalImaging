"""Hello World Deep Learning in Medical Imaging, from notebook to bedside.

The original tutorial (Lakhani et al., Journal of Digital Imaging, 2018,
https://doi.org/10.1007/s10278-018-0079-6) trains a chest-versus-abdomen
radiograph classifier in a Jupyter notebook. This package reorganises the same
pipeline into small, testable modules:

- :mod:`.config` reads the experiment configuration (paths, hyperparameters,
  random seed) from a TOML file;
- :mod:`.data` downloads the images and turns them into datasets;
- :mod:`.model` builds the transfer-learning model;
- :mod:`.training` trains a model on datasets it receives as arguments;
- :mod:`.predict` classifies a single image with a trained model;
- :mod:`.cli` wires everything together for the command line.
"""
