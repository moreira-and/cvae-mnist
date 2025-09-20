# Contributing to C-VAE MNIST

Thanks for your interest in improving this project! Contributions of all kinds are welcome — from bug reports and docs fixes to features and refactors.

## How to Contribute

- Report bugs and request features via GitHub Issues.
- For code changes, open a Pull Request (PR) against `main`.
- Keep PRs focused: smaller changes are faster to review.

## Development Setup

```bash
poetry install
poetry shell  # optional
```

Run quality checks before submitting a PR:

```bash
poetry run black src
poetry run isort src
poetry run flake8 src
poetry run pytest -q
```

## Commit & PR Guidelines

- Use clear, descriptive titles.
- Reference related issues when relevant (e.g., "Fixes #123").
- Include tests for new logic when applicable.
- Update documentation (README or docs) to reflect changes.

## Coding Style

- Python formatting via `black` and `isort`.
- Keep functions small and focused.
- Prefer explicit names over abbreviations.

## Security

- Don’t include secrets in code or configs.
- If you discover a security issue, please report it privately.

Thanks for contributing!
