# Automation and releases

Three GitHub Actions workflows live in `.github/workflows/`.

## Continuous integration (`ci.yml`)

On every push and pull request:

1. install the locked dependencies with uv;
2. `ruff check` and `ruff format --check`;
3. `mypy`;
4. `pytest`, which runs the unit tests, the integration test and the doctests;
5. in a second job, build the Docker image and run `pytest` inside it, so that
   the `Dockerfile` is verified too.

## Documentation (`docs.yml`)

The site in `docs/` is built with MkDocs (`mkdocs build --strict`, so a broken
link fails the build) on every push and pull request. On the `main` branch the
result is published to GitHub Pages at
<https://tomboulier.github.io/HelloWorld-ML-MedicalImaging/>.

Publishing needs Pages to be enabled once by a repository administrator:
*Settings > Pages > Build and deployment > Source: GitHub Actions*.

## Versions and changelog (`release.yml`)

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:` for a new behaviour, `fix:` for a bug fix, `docs:`, `test:`,
`refactor:`, `ci:`, `build:` and `chore:` for the rest, and `!` or a
`BREAKING CHANGE:` footer for incompatible changes. From these,
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
computes the next [Semantic Versioning](https://semver.org/) number (a `fix:`
bumps the patch number, a `feat:` the minor number, a breaking change the
major number), writes it in `pyproject.toml`, adds the section to
`CHANGELOG.md` (which follows [Keep a Changelog](https://keepachangelog.com/)),
commits, tags and creates a GitHub release.

The release workflow is started by hand (*Actions > Release > Run workflow*)
rather than on every push to `main`, so that a maintainer decides when a
version exists. To see what it would do without changing anything:

```bash
uv sync --group release
uv run semantic-release --noop version
```

## Publishing to PyPI (not done here)

This package is not published to PyPI: it is a teaching example, and
publishing would put a radiograph classifier into a package index where people
install things without reading them. If it were to be published, the steps
would be:

1. choose a name that is free on PyPI and set it in `pyproject.toml`;
2. build the wheel and the source distribution (`uv build`);
3. configure a *trusted publisher* on PyPI for this repository and workflow,
   so that no API token has to be stored in GitHub;
4. add a job to `release.yml`, after the version is tagged, that uploads the
   files (for example with the `pypa/gh-action-pypi-publish` action);
5. test the whole chain against TestPyPI first.

The Carpentries Incubator lesson
[Publishing a Python package](https://carpentries-incubator.github.io/python-packaging-publishing/)
walks through these steps.

## Container registry (not done here)

The CI builds the image but does not push it. Pushing it to the GitHub
Container Registry (`ghcr.io/tomboulier/helloworld-ml-medicalimaging`) would
let a cluster pull it with Apptainer (see [Usage](usage.md)); it is a
`docker/login-action` plus a `docker push` in the `container` job, with the
`packages: write` permission. It is left as a decision for the maintainer.
