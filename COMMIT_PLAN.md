# Commit Plan for SchwabGym

This file lists a recommended, logical series of commits to organize, document, and prepare the repository for development and CI. Execute each step on a feature branch (example: `dev/cleanup-and-ci`) and create a PR when ready.

Suggested branch name:

```
git checkout -b dev/cleanup-and-ci
```

Commit series (apply in order):

1) chore: update README with project structure
   - Description: Adds a clear `Project Structure` section to `README.md` so contributors can quickly find key modules and tests.
   - Files changed: `README.md`
   - Example:
     - `git add README.md`
     - `git commit -m "chore(readme): add project structure overview"

2) docs: add commit plan and contribution notes
   - Description: Add `COMMIT_PLAN.md` (this file) describing further commits and suggested workflow.
   - Files changed: `COMMIT_PLAN.md`
   - Example:
     - `git add COMMIT_PLAN.md`
     - `git commit -m "docs: add commit plan and contributor checklist"`

3) style: add formatting and pre-commit tooling
   - Description: Add `pyproject.toml` or update `setup.cfg` to include formatting settings (Black, isort), and add a `.pre-commit-config.yaml` to enforce formatting before commits.
   - Files changed: `pyproject.toml` (new) or `setup.cfg`, `.pre-commit-config.yaml`
   - Rationale: Keep code style consistent and reduce PR churn.

4) test: ensure tests run and fix failing tests
   - Description: Run `pytest`, fix or mark flaky tests, ensure `requirements.txt` has test deps. Update `tests/` if needed.
   - Files changed: changes to tests or supporting fixtures (if required)

5) ci: add GitHub Actions workflow for tests
   - Description: Add `.github/workflows/ci.yml` to run tests on Python matrix (3.8, 3.9, 3.10/3.11 as desired), cache pip, run pytest.
   - Files changed: `.github/workflows/ci.yml`

6) docs: tidy and expand examples
   - Description: Improve `examples/` to include minimal runnable examples and update `LIVE_TRADING.md` with deploy notes.
   - Files changed: `examples/*`, `LIVE_TRADING.md`

7) feat: small refactors & public API exports
   - Description: Make any small API improvements to module exports (for example, confirm `__all__` in `schwabgym/__init__.py`), ensure `setup.py` metadata is correct.
   - Files changed: `schwabgym/__init__.py`, `setup.py`

8) chore: bump version and tag release
   - Description: Update `setup.py` / version file and tag release: `v0.x.y`
   - Files changed: `setup.py` or project version file

Notes on applying commits safely:

- Work on a branch: `git checkout -b dev/cleanup-and-ci`
- Stage and commit logically grouped changes. Keep commits small and focused.
- Run tests locally: `pytest -q`
- Run linters/formatters: `black .` `isort .`
- To create a signed/tidy PR, push the branch and open a PR against `dev` or `main`.

Example sequence of commands to implement the first two commits (already applied locally by the assistant):

```bash
git checkout -b dev/cleanup-and-ci
git add README.md COMMIT_PLAN.md
git commit -m "chore(readme): add project structure overview"
git commit -m "docs: add commit plan and contributor checklist"
git push -u origin dev/cleanup-and-ci
```

If you'd like, I can:

- Create the `pyproject.toml` and `.pre-commit-config.yaml` and commit them.
- Add a minimal GitHub Actions CI workflow and run tests locally.
- Make small API or packaging fixes (one logical change per commit).

Tell me which of the proposed subsequent commits you'd like me to implement next and I will proceed.
