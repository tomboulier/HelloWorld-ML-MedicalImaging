# 0001 — Inject the data source into training

Date: 2026-11-06

## Status

Accepted

## Context

In the notebook, and in the first extraction into modules, the training code
knew where the images were: it downloaded the archive, listed the folders and
built the Keras generators itself. Testing the training step therefore meant
downloading 75 radiographs and the ImageNet weights, and there was no way to
train on anything else without editing the training code.

Training needs two things: a compiled model and batches of `(images, labels)`.
It does not need to know where either comes from.

## Decision

`train()` in `training.py` takes the model and the two datasets as arguments.
The caller decides where they come from:

- `cli.py` downloads the archive and reads the `TRAIN` and `VAL` folders with
  `load_image_datasets()`;
- the tests call `synthetic_datasets()`, which generates noise images with the
  same shape and labels.

Both sources honour one contract, stated in `data.py`: batches of RGB images
with values in `[0, 255]` and binary labels. The model itself rescales to
`[0, 1]` and applies augmentation, so every source stays as simple as "read
images".

There is no `DataSource` base class or interface: two functions with the same
return type are enough for the needs of this repository (YAGNI). The entry
point `cli.py` is the only module that imports every other one; it is the
place where the pieces are assembled (the *composition root*).

## Consequences

- The integration test trains one epoch in about half a minute, offline, with
  random weights, because nothing in `training.py` touches the network or the
  disk.
- Replacing the data source (another folder layout, a DICOM reader, a database)
  means adding one function in `data.py` and calling it from `cli.py`; the
  model and the training code do not change.
- Anyone reading `train()` sees its whole input in the signature.
- The cost is that `cli.py` has to pass a few more arguments around, and that a
  new data source has to respect the contract by discipline, since nothing
  enforces it at import time. The test
  `test_synthetic_datasets_have_the_same_contract_as_real_ones` checks it for
  the two existing sources.
