# DOLFINx time slab solver

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/UCL/dxss/actions/workflows/tests.yml/badge.svg)](https://github.com/UCL/dxss/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/UCL/dxss/graph/badge.svg?token=1O6E05lrHn)](https://codecov.io/gh/UCL/dxss)
[![Linting](https://github.com/UCL/dxss/actions/workflows/linting.yml/badge.svg)](https://github.com/UCL/dxss/actions/workflows/linting.yml)
[![Documentation](https://github.com/UCL/dxss/actions/workflows/docs.yml/badge.svg)](https://github-pages.ucl.ac.uk/dxss/)
[![Licence][licence-badge]](./LICENCE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/dxss
[conda-link]:               https://github.com/conda-forge/dxss-feedstock
[pypi-link]:                https://pypi.org/project/dxss/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/dxss
[pypi-version]:             https://img.shields.io/pypi/v/dxss
[licence-badge]:             https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

`dxss` provides DOLFINx solvers on space-time finite element spaces which use a partition of the time interval to decompose the spatio-temporal domain into a collection of _time slabs_.

This project is developed in collaboration with the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## Documentation

Documentation can be viewed at https://github-pages.ucl.ac.uk/dxss/

## About

### Project team

Current members

- Erik Burman ([burmanerik](https://github.com/burmanerik))
- Sam Cunliffe ([samcunliffe](https://github.com/samcunliffe))
- Deepika Garg ([deepikagarg20](https://github.com/deepikagarg20))
- Krishnakumar Gopalakrishnan ([krishnakumarg1984](https://github.com/krishnakumarg1984))
- Matt Graham ([matt-graham](https://github.com/matt-graham))
- Janosch Preuss ([janoschpreuss](https://github.com/janoschpreuss))

Former members

- Anastasis Georgoulas ([ageorgou](https://github.com/ageorgou))
- Jamie Quinn ([JamieJQuinn](https://github.com/JamieJQuinn))

### Research software engineering contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Built with

- [FEniCSx](https://fenicsproject.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

## Getting started

### Prerequisites

Compatible with Python 3.9 and 3.10. [Requires DOLFINx v0.6.0 or above to be installed](https://github.com/FEniCS/dolfinx#installation).

### Installation

To install the latest development using `pip` run

```sh
pip install git+https://github.com/UCL/dxss.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/dxss.git
```

and then install in editable mode by running

```sh
pip install -e .
```

from the root of your clone of the repository.

### Running tests

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

from the root of the repository, or to run tests with Python 3.9 specifically run

```sh
tox -e test-py39
```

substituting `py39` for `py310` to run tests with Python 3.10.

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building documentation

HTML documentation can be built locally using `tox` by running

```sh
tox -e docs
```

from the root of the repository with the output being written to `docs/_build/html`.

## Acknowledgements

This work was funded by a grant from the the Engineering and Physical Sciences Research Council (EPSRC).
