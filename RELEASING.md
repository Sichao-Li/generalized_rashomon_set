# Releasing

Maintainer notes for publishing `generalized_rashomon_set` to PyPI.

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. Create a GitHub release from the tag. Publishing the release triggers
   `.github/workflows/publish.yml`, which builds the sdist and wheel and uploads
   them to PyPI via trusted publishing (no API token is stored in the repo).

One-time setup: the maintainer must add a trusted publisher on PyPI
(project `generalized_rashomon_set`, owner `Sichao-Li`, repository
`generalized_rashomon_set`, workflow `publish.yml`, environment `pypi`) and
create the `pypi` environment in the GitHub repository settings.
