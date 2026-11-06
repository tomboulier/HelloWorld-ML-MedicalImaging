[![CI](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/actions/workflows/ci.yml/badge.svg)](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/actions/workflows/ci.yml)
[![Documentation](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/actions/workflows/docs.yml/badge.svg)](https://tomboulier.github.io/HelloWorld-ML-MedicalImaging/)

# Hello World Deep Learning in Medical Imaging, from notebook to bedside

This repository takes the tutorial *Hello World Deep Learning in Medical
Imaging* (Lakhani et al., Journal of Digital Imaging, 2018,
[doi:10.1007/s10278-018-0079-6](https://doi.org/10.1007/s10278-018-0079-6);
original code and data at
[paras42/Hello_World_Deep_Learning](https://github.com/paras42/Hello_World_Deep_Learning))
and walks it, in six steps, from a Jupyter notebook to reproducible, tested and
documented code. The model is a chest-versus-abdomen radiograph classifier
built on Inception V3.

The companion page of the teaching video, with the six steps and their
readings, is published at
<https://tomboulier.github.io/HelloWorld-ML-MedicalImaging/>.

> This is a teaching example. It is neither a medical device nor compliant
> with any standard for one.

## Installation

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
(pip works too, see below).

```bash
git clone https://github.com/tomboulier/HelloWorld-ML-MedicalImaging.git
cd HelloWorld-ML-MedicalImaging
uv sync
```

`uv sync` creates `.venv` with the exact dependency versions recorded in
`uv.lock`. With pip instead: `python -m venv .venv && . .venv/bin/activate && pip install -e .`
(same direct dependencies, but transitive ones are not pinned).

## Usage

Train with the settings of `config.toml` (the images are downloaded on first run):

```bash
uv run hello-world-medical-imaging
```

Settings (paths, image size, epochs, learning rate, random seed, data
augmentation) are in `config.toml`; pass another file with `--config`. The
trained model, the training curves and the predictions on the two test images
go to `outputs/` and to the log.

Run the checks:

```bash
uv run pytest                 # unit tests, integration test, doctests
uv run ruff check .           # lint
uv run mypy                   # types
uv run mkdocs build --strict  # documentation (needs: uv sync --group docs)
```

With Docker, or Apptainer on a cluster: see
[docs/usage.md](docs/usage.md).

## Layout

```
src/hello_world_medical_imaging/   the package: config, data, model, training, predict, cli
tests/                             pytest suite (the integration test trains one epoch on synthetic images)
docs/                              MkDocs site: roadmap (companion page), usage, testing, automation, ADRs
notebooks/                         the original tutorial notebook, as the starting point
config.toml                        experiment settings
Dockerfile                         image that runs the training and the tests
```

## Data

The 75 radiographs are not stored in this repository. They are downloaded at
run time from the original tutorial repository, which describes them as
de-identified images from [Open-i](https://openi.nlm.nih.gov/) and states no
licence for them. Nothing from the dataset is committed here, including
notebook outputs.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (and [AGENTS.md](AGENTS.md) for coding agents).
