# Release Checklist

Follow this checklist when preparing and publishing a release.

1. Bump the version in `setup.py` or the project's version file.
2. Run the full test suite locally:

```bash
tox
```

3. Build distributions locally and inspect them:

```bash
python -m pip install --upgrade build
python -m build
ls dist/
```

4. Create a signed tag (optional) and push it:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

5. After pushing the tag:
   - The repository will publish to TestPyPI automatically.
   - Verify the test release (install from TestPyPI if desired).
   - Approve the `production` environment on GitHub to trigger PyPI publishing.

6. Confirm the PyPI release appears on https://pypi.org/project/schwabgym/ and update the `CHANGELOG.md` if present.

Security: Do not commit API tokens to the repository. Use GitHub Secrets `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN`.
