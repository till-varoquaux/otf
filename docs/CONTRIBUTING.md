
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

