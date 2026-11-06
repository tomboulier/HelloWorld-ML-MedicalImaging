# Testing

The test suite lives in `tests/` and runs with `uv run pytest`. It has three kinds of tests:

- **unit tests** of the extracted functions: configuration loading, archive
  download and unpacking (from a local `file://` archive, no network), dataset
  loading from a folder of generated images, model shapes;
- **one integration test** (`tests/test_training.py`) that builds the real
  architecture with random weights at 75 x 75 pixels, trains it for one epoch
  on synthetic noise images, predicts one image and plots the curves. It runs
  in about half a minute on a laptop CPU, with no download;
- **doctests** in the docstrings, run by pytest through `--doctest-modules`.

Ruff (lint and formatting) and mypy (types, in strict mode) are configured in
`pyproject.toml` and run in the same CI workflow.

## A test written before the code

Test-driven development means writing a failing test that describes the
behaviour, then the smallest code that makes it pass, then tidying up. Here is
how `download_dataset` was written. The behaviour wanted: unpack a zip archive
into a destination directory, and do nothing if that directory already exists.

The test came first, with a tiny archive built in a temporary directory so
that it needs no network:

```python
def test_download_dataset_unpacks_the_archive(dataset_archive: Path, tmp_path: Path) -> None:
    destination = tmp_path / "unpacked" / "Open_I_abd_vs_CXRs"

    result = download_dataset(dataset_archive.as_uri(), destination)

    assert result == destination
    assert (destination / "TRAIN" / "openI_CXR" / "0.png").exists()
    assert (destination / "TEST" / "abd2.png").exists()
```

Running `uv run pytest tests/test_data.py` at that point fails with
`ImportError: cannot import name 'download_dataset'`. That is the expected
first step: the test states the contract before any code exists. Then the
function is written until the test passes:

```python
def download_dataset(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
        urllib.request.urlretrieve(url, archive.name)
        with zipfile.ZipFile(archive.name) as zipped:
            zipped.extractall(destination.parent)
    return destination
```

The second test, "do nothing when the destination exists", was added the same
way, with an archive URL that does not exist so that any attempt to download
fails loudly. The third, "fail if the archive does not contain the expected
folder", made the `FileNotFoundError` branch appear. The final version in
`src/hello_world_medical_imaging/data.py` adds logging and docstrings, nothing
else.

## What a green test suite does and does not tell you

A passing suite says that the behaviours someone thought of writing down still
hold. It says nothing about the behaviours nobody wrote down. A coverage
percentage counts lines that were executed by some test, not lines whose
behaviour was checked; it is a tool to find untested code, not a proof of
robustness. The integration test here trains on noise: it shows that the
pipeline runs, not that the model is any good.
