# Usage

How to install, configure and run the training, on a workstation or on a cluster.

## Install

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tomboulier/HelloWorld-ML-MedicalImaging.git
cd HelloWorld-ML-MedicalImaging
uv sync
```

`uv sync` creates a virtual environment in `.venv` with the exact versions
recorded in `uv.lock`. Direct dependencies are pinned in `pyproject.toml`;
`uv.lock` pins every transitive dependency too, so two people running `uv sync`
get the same environment.

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install -e .`
(transitive dependencies are then not pinned).

## Configure

`config.toml` holds everything that used to be a global variable in the
notebook: where the data is downloaded, image size, batch size, number of
epochs, learning rate, data augmentation ranges and the random seed. Copy it,
edit the copy, and pass it with `--config`:

```bash
uv run hello-world-medical-imaging --config my-experiment.toml
```

The seed is applied to Python, NumPy and TensorFlow. On CPU, two runs with the
same seed give the same numbers; on GPU, some operations stay non-deterministic
unless TensorFlow is told otherwise.

## Run

```bash
uv run hello-world-medical-imaging
```

On first run the 75 radiographs are downloaded from the original tutorial
repository into `data/` (ignored by Git), and the ImageNet weights of Inception
V3 are downloaded by Keras into its cache. The trained model
(`outputs/model.keras`), the training curves (`outputs/training_curves.png`)
and the scores of the two test images (in the log) are the results.

## Run the same image with Docker on a workstation

The `Dockerfile` builds an image with Python 3.12 and the locked dependencies.
Build it once:

```bash
docker build --tag hello-world-medical-imaging .
```

Train, keeping the data and the outputs on the host:

```bash
docker run --rm \
  --volume "$PWD/data:/app/data" \
  --volume "$PWD/outputs:/app/outputs" \
  hello-world-medical-imaging
```

Run the tests inside the image (this is what the CI does):

```bash
docker run --rm hello-world-medical-imaging pytest
```

Podman, which needs no daemon, accepts the same commands with `podman` in
place of `docker`.

## Run the same image with Apptainer on a cluster

Clusters usually do not allow Docker but provide
[Apptainer](https://apptainer.org/), which runs Docker images from a registry
without a daemon or root rights. The GRICAD computing centre in Grenoble, for
example, documents this use
(<https://gricad-doc.univ-grenoble-alpes.fr/>).

Apptainer needs the image in a registry. If the image has been pushed, for
example to the GitHub Container Registry under the name
`ghcr.io/tomboulier/helloworld-ml-medicalimaging`, then on the cluster:

```bash
apptainer exec docker://ghcr.io/tomboulier/helloworld-ml-medicalimaging:latest \
  hello-world-medical-imaging --config config.toml
```

`apptainer exec` mounts the current directory, so `config.toml`, `data/` and
`outputs/` are read and written where you launch the command. Add `--nv` to
use the node's GPUs. Pushing the image to a registry is not automated by this
repository; see [Automation and releases](automation.md).

Without a registry, build the image on a machine that has Docker and convert
it to a single file that you copy to the cluster:

```bash
apptainer build hello-world-medical-imaging.sif docker-daemon://hello-world-medical-imaging:latest
```
