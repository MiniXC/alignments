Since I don't expect to update this package often, there are a few common tasks that I'll need to do before releasing a new version. This document will help me remember what those tasks are.

## Update the version number
The first thing I need to do is update the version number in the following files:
- `pyproject.toml`
- `README.md`

## Run the tests & coverage - then generate the badge
For this we need the following packages:
- `pytest`
- `coverage`
- `coverage-badge`
- `pytest-local-badge`
Install command:
```bash
pip install pytest coverage coverage-badge pytest-local-badge
```
I need to make sure that the tests are passing and that the coverage is at an acceptable level. I'll run the tests and coverage, then generate the badge to include in the `README.md` file.
```bash
coverage run -m pytest -vs --local-badge-output-dir badges/
coverage-badge -f -o badges/coverage.svg
```