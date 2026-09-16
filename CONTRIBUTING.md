# Contributing

Install Python 3.11+ and run:

```bash
python -m pip install '.[dev]'
python -m pytest -q
python -m ruff check src tests
python -m build
```

Include a small deterministic regression case for behavioral changes. Keep examples synthetic, document score units and timing assumptions, and do not commit downloaded market datasets or credentials. Changes to financial formulas should describe the old and new behavior and distinguish statistical validation from implementation tests.

After changing source, reinstall with `python -m pip install '.[dev]'` before testing, or use an editable install (`python -m pip install -e '.[dev]'`) in a standard development interpreter.

Automatic GitHub Actions are not enabled in the initial release because the publishing credential lacks `workflow` scope. A maintainer with the appropriate permission can copy `docs/github-actions-tests.yml` to `.github/workflows/tests.yml` to enable the supplied Python-version matrix.
