# From notebook to bedside: a roadmap

This page accompanies the video *Best Practices in Software Engineering for
Robust and Maintainable Medical AI*. The video is a map, not a tutorial: each
of the six steps below points to existing lessons and readings rather than
teaching them. This repository shows every step carried out on one small
example, the *Hello World Deep Learning in Medical Imaging* tutorial of
Lakhani et al. ([doi:10.1007/s10278-018-0079-6](https://doi.org/10.1007/s10278-018-0079-6),
code at [paras42/Hello_World_Deep_Learning](https://github.com/paras42/Hello_World_Deep_Learning)),
a Jupyter notebook that classifies radiographs as chest or abdomen.

The roadmap is not about certifying a medical device. It is about what has to
be true of the code before anyone relies on it. Tools are named below as
examples, never as recommendations.

## 1. Version control

**Done here.** The [original notebook](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/tree/main/notebooks)
is in the repository as the starting point, restored from the history with its
outputs stripped. Every change since is a commit written as a
[Conventional Commit](https://www.conventionalcommits.org/), so that the
history reads as the roadmap itself (`git log --oneline`). Data, model weights
and training outputs are excluded by
[`.gitignore`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/.gitignore).

**What to retain.** A history is a lab notebook that cannot be lost or
rewritten by accident; it is only useful if each entry says what changed and
why. Large files (images, weights) do not belong in Git: tools such as
[DVC](https://dvc.org) or [Git LFS](https://git-lfs.com) version them next to
the code without bloating it.

**Lessons and readings.**
[Version Control with Git](https://swcarpentry.github.io/git-novice/), Software Carpentry ·
[Pro Git](https://git-scm.com/book/en/v2).

## 2. Reproducible environment

**Done here.** Dependencies are pinned in
[`pyproject.toml`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/pyproject.toml)
and `uv.lock`. Paths, hyperparameters and the random seed left the code for
[`config.toml`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/config.toml).
A [`Dockerfile`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/Dockerfile)
runs the training and the tests; [Usage](usage.md) shows the same image run
with Docker on a workstation and with Apptainer on a cluster.

**What to retain.** "It works on my machine" is a statement about the machine.
Writing down the exact versions, taking the magic numbers out of the code and
packaging the whole thing in an image are three steps towards someone else, or
you in a year, getting the same result.

**Lessons and readings.**
[Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510), Wilson et al. 2017 ·
[The Turing Way](https://book.the-turing-way.org/) ·
[Python packaging](https://carpentries-incubator.github.io/python_packaging), Carpentries Incubator ·
[Docker documentation](https://docs.docker.com/get-started/).

## 3. Architecture

**Done here.** The notebook became four small modules in
[`src/hello_world_medical_imaging/`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/tree/main/src/hello_world_medical_imaging):
`data.py` (loading), `model.py` (the network), `training.py` (fitting) and
`predict.py`. The training function receives the model and the datasets as
arguments instead of building them itself (dependency injection), so the tests
swap the radiographs for synthetic images without touching the model
([ADR 0001](adr/0001-inject-the-data-source-into-training.md)). Short
[`AGENTS.md`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/AGENTS.md) and
[`CONTRIBUTING.md`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/CONTRIBUTING.md)
state the architecture, the test commands and two guard-rails: keep it simple
(KISS), and do not build what is not needed yet (YAGNI).

**What to retain.** The point of architecture is to decide what depends on
what. When the part that trains does not know where the data comes from, you
can test it in seconds and replace the data source later. Any further layer
has to earn its place.

**Lessons and readings.**
[Clean Architecture](https://www.oreilly.com/library/view/clean-architecture-a/9780134494272/), Robert C. Martin ·
[Hexagonal architecture](https://alistair.cockburn.us/hexagonal-architecture/), Alistair Cockburn ·
[SOLID](https://en.wikipedia.org/wiki/SOLID) ·
[Design patterns](https://en.wikipedia.org/wiki/Design_Patterns) ·
[Better Architecture, Better Software, Better Research](https://doi.org/10.1109/MCSE.2025.3573887), Druskat et al. 2025 ·
[Intermediate Research Software Development](https://carpentries-incubator.github.io/python-intermediate-development/), Carpentries Incubator ·
[Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/), Software Carpentry ·
[Research Software Engineering with Python](https://alan-turing-institute.github.io/rse-course/html/index.html), Alan Turing Institute.

## 4. Tests

**Done here.** [`tests/`](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/tree/main/tests)
holds pytest unit tests of the extracted functions and one integration test
that trains the real architecture for one epoch on synthetic images, offline.
Ruff (lint) and mypy (types) are configured in `pyproject.toml`.
[Testing](testing.md) shows a test written before its code, and says what a
green suite does and does not prove.

**What to retain.** A test is a written-down expectation that a machine checks
on every change. Writing it first forces you to say what the code should do
before you say how. A coverage percentage is a way to find untested code, not
a measure of robustness.

**Lessons and readings.**
[Python Testing and Continuous Integration](https://carpentries-incubator.github.io/python-testing), Carpentries Incubator ·
[Test-Driven Development: By Example](https://www.oreilly.com/library/view/test-driven-development/0321146530/), Kent Beck ·
[pytest](https://docs.pytest.org/) ·
[Ruff](https://docs.astral.sh/ruff/) ·
[mypy](https://mypy.readthedocs.io/).

## 5. Documentation

**Done here.** The [README](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging#readme)
says how to install and run. Every function has a docstring (PEP 257), several
with doctests that pytest runs. Messages go through the `logging` module
instead of `print`. The first architecture decision record, in
[`docs/adr/`](adr/index.md), records the dependency injection choice in the
format proposed by Michael Nygard.

**What to retain.** Documentation is several things for several readers: a
how-to for the user, a reference for the developer, an explanation for the
reviewer. Docstrings with examples stay true because they are tested;
decision records stay useful because they are never rewritten.

**Lessons and readings.**
[Diátaxis](https://diataxis.fr/) ·
[PEP 257](https://peps.python.org/pep-0257/) ·
[doctest](https://docs.python.org/3/library/doctest.html) ·
[Logging HOWTO](https://docs.python.org/3/howto/logging.html) ·
[Better Research Software](https://carpentries-incubator.github.io/better-research-software/), Carpentries Incubator ·
[Architecture decision records](https://adr.github.io/).

## 6. Automation

**Done here.** A GitHub Actions
[workflow](https://github.com/tomboulier/HelloWorld-ML-MedicalImaging/blob/main/.github/workflows/ci.yml)
runs Ruff, mypy, pytest and the doctests on every push and pull request, and
builds the container image. A second workflow builds this site with MkDocs and
publishes it on GitHub Pages. The version number and `CHANGELOG.md` are derived
from the Conventional Commits with python-semantic-release; nothing is
published to PyPI, [Automation and releases](automation.md) describes how it
would be done.

**What to retain.** Whatever is checked by hand is checked sometimes. A
workflow that runs the checks on every change turns "we should test this"
into "this was tested", and leaves a record of it.

**Lessons and readings.**
[GitHub Actions](https://docs.github.com/en/actions) ·
[Publishing a Python package](https://carpentries-incubator.github.io/python-packaging-publishing/), Carpentries Incubator ·
[GitHub Pages](https://pages.github.com/) ·
[python-semantic-release](https://python-semantic-release.readthedocs.io/) ·
[Semantic Versioning](https://semver.org/) ·
[Keep a Changelog](https://keepachangelog.com/).

## Working with coding agents

Much of the code in this repository was written with a coding agent, under the
instructions of `AGENTS.md` and the checks of the CI. The references below
describe the practice from several angles; none of them is a recommendation.

- Bridgeford et al., *Twelve quick tips for AI-assisted coding in science*,
  PLOS Computational Biology, 2026, <https://doi.org/10.1371/journal.pcbi.1014428>.
- O'Brien and Eisty, *Scientific Software in the Age of Vibe Coding*,
  Computing in Science & Engineering, 2026, <https://doi.org/10.1109/MCSE.2026.3661610>.
- Elle O'Brien, *Who is scientific code for? Maintaining human-readable landmarks in agent-written code*, 2026, <https://arxiv.org/abs/2607.25975>.
- Birgitta Böckeler, *Understanding Spec-Driven-Development: Kiro, spec-kit, and Tessl*, 2025,
  <https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html>;
  Thoughtworks, *Spec-driven development: Unpacking one of 2025's key new AI-assisted engineering practices*, 2025,
  <https://www.thoughtworks.com/en-us/insights/blog/agile-engineering-practices/spec-driven-development-unpacking-2025-new-engineering-practices>.
- Examples of tools for specification-driven work:
  [spec-kit](https://github.com/github/spec-kit),
  [BMAD](https://github.com/bmad-code-org/BMAD-METHOD),
  [Kiro](https://kiro.dev/docs/specs/),
  [Tessl](https://tessl.io/).
- Instructions files for agents: [AGENTS.md](https://agents.md),
  [CLAUDE.md](https://code.claude.com/docs/en/memory),
  [Agent Skills](https://agentskills.io/).
- Policies and guides:
  [The Carpentries' GenAI policy](https://docs.carpentries.org/policies/genai-policy.html);
  [Responsible use of AI in research software](https://www.software.ac.uk/guide/responsible-use-ai-research-software), Institute for Research Software;
  [Developing Research Software with Generative AI](https://carpentries-incubator.github.io/software-development-with-genai/), Carpentries Incubator.
- Measured effects: Becker et al. (METR), *Measuring the Impact of Early-2025 AI on
  Experienced Open-Source Developer Productivity*, 2025, <https://arxiv.org/abs/2507.09089>,
  and METR's follow-up of February 2026, <https://metr.org/blog/2026-02-24-uplift-update/>.

## Background reading

- Hannay et al., *How do scientists develop and use scientific software?*, 2009, <https://doi.org/10.1109/SECSE.2009.5069155>.
- Merali, *Computational science: ...Error*, Nature, 2010, <https://doi.org/10.1038/467775a>.
- Sandve et al., *Ten simple rules for reproducible computational research*, PLOS Computational Biology, 2013, <https://doi.org/10.1371/journal.pcbi.1003285>.
- Wilson et al., *Best practices for scientific computing*, PLOS Biology, 2014, <https://doi.org/10.1371/journal.pbio.1001745>.
- McDermott et al., *Reproducibility in machine learning for health research: Still a ways to go*, Science Translational Medicine, 2021, <https://doi.org/10.1126/scitranslmed.abb1655>.
- Heil et al., *Reproducibility standards for machine learning in the life sciences*, Nature Methods, 2021, <https://doi.org/10.1038/s41592-021-01256-7>.
- Varoquaux and Cheplygina, *Machine learning for medical imaging: methodological failures and recommendations for the future*, npj Digital Medicine, 2022, <https://doi.org/10.1038/s41746-022-00592-y>.
