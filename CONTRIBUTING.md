# Contribution guidelines

- We very much welcome reports of any and all [bugs](https://github.com/UCL/dxss/issues/new/choose)!

## Medium-term plans and milestones

- Add draft work items to the [project board](https://github.com/orgs/UCL/projects/36), and discuss!
  - All members of the Burman group should have access.
  - Feel free to request access if you don't have it.

## Pull requests

- Open draft PRs early, especially if you want early feedback on a feature.
  - Try to avoid "grand unveilling" of large pieces of work.
  - Try to keep focused on "one feature" or "one issue". Aim to minimise unrelated code changes per PR.
- Convert to "Ready for review" when there are no changes expected from the requester side and add, minimally, one reviewer.
- New features should be covered by new test(s), unless it's obviously already tested.
- All tests should pass and all linting should pass. Test coverage should, at least, stay the same.
- If a reviewer has no comments or questions then the approving reviewer is encouraged to merge the PR themselves to keep things tidy.
- Merge via squash merge (so we have one commit per PR).

## Code style and linters

- We aim for PEP8-compliant Python code with one exception:
  > Capital letters for matrices (following conventional math notation).
  - The linter will flag up the use of i.e. `A` as a variable name. Silence this with `# noqa: N803`.
- Aim to document all public-facing API code with [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google) docstrings.
- Use [type hints](https://peps.python.org/pep-0484/) throughout and check with `mypy` or `pre-commit`.
  - No need to put type hints in docstrings.

### To run the linters locally

Install [pre-commit](https://pre-commit.com/) with:

```sh
python -m pip install pre-commit # first time only
pre-commit install
```

Run from the root of the repository:

```sh
pre-commit run
```