# Releasing

The repository is set up so that publishing is a single tag push.

## Normal release (GitHub Release)

1. Bump `version` in `pyproject.toml` and `__version__` in `streamcount/__init__.py`.
2. Move the `[Unreleased]` notes in `CHANGELOG.md` under the new version heading.
3. Commit, then tag and push:

```bash
git tag v0.2.0
git push origin main --tags
```

The `Release` workflow lints, tests, builds the sdist + wheel, and attaches both to a
GitHub Release with generated notes.

## PyPI (optional, trusted publishing)

1. On PyPI, add a trusted publisher for this repository with:
   - workflow: `release.yml`
   - environment: `pypi`
2. In GitHub → Settings → Secrets and variables → Actions → Variables, add `PUBLISH_TO_PYPI = true`.
3. The next tag push publishes to PyPI automatically.

Before the first PyPI upload, make sure any README images use **absolute URLs**
(`https://github.com/<owner>/streamcount/raw/main/assets/...`) — relative paths render on
GitHub but break on PyPI.

## Local pre-flight check

```bash
pip install build twine
python -m build
twine check dist/*
python -m zipfile -l dist/*.whl | head        # inspect wheel contents
tar -tzf dist/*.tar.gz | head -30             # inspect sdist contents
```

The sdist carries docs, examples, scripts and assets; the wheel carries the package only.
Neither ever contains `runs/`, `.env` files or model weights.
