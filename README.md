# On-the-fly distributed python workflows

[![PyPI](https://img.shields.io/pypi/v/otf.svg)](https://pypi.org/project/otf/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/otf)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Tests](https://github.com/till-varoquaux/otf/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/till-varoquaux/otf/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/till-varoquaux/otf/branch/main/graph/badge.svg?token=ahhI117oFg)](https://codecov.io/gh/till-varoquaux/otf)

OTF is a framework to programatically write, run and debug workflows.

## Installing

OTF is currently in pre-alpha. If you really want to play with it you can
install the latest build via:

```bash
$ pip install -i https://test.pypi.org/simple/ otf
```

## Testing

We use [tox](https://tox.wiki/en/latest/) and
[poetry](https://python-poetry.org/) to manage dependencies. You can run the
tests from the project direcotry (on a machine where tox and python 3.10 are
already installed) by calling:

```bash
$ tox
```

### Development environment

To setup your dev environment you should install the extra dev-tools package in
poetry.

```bash
$ poetry install -E dev-tools
```

This will install all the tools required to work on the code base including:

* [Grip](https://github.com/joeyespo/grip): Instant preview of the `README.md` file
```bash
$ poetry run grip . 0.0.0.0:8080
```

* [Pytest-watch](https://github.com/joeyespo/pytest-watch): Continuous pytest runner
```bash
$ poetry run ptw
```

* [Black](https://black.readthedocs.io/en/stable/index.html): Code formatter
```bash
$ poetry run black --line-length=80 .
```

* [Isort](https://pycqa.github.io/isort): Sort the python imports
```bash
$ poetry run isort .
```

* [Flake8](https://flake8.pycqa.org/en/latest): Linting and style
```bash
$ poetry run flake8 .
```

* [Mypy](http://mypy-lang.org/): Static type checker
```bash
$ poetry run mypy src tests
```

## Road to alpha0

The `alpha 0` version of OTF will support sending python closures over the
wire. Although we will not have our own serialization format we will support
*exploding* functions to a representations that plays nicely with `pickle`.

- [x] Parse functions
- [ ] Capture closures
- [ ] Unparse functions
- [ ] Explode/implode functions
