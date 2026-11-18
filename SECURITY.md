# Security policy

## Reporting a vulnerability

Report security problems privately through GitHub: open the repo's [Security tab](https://github.com/mlvpatel/papers-that-built-ai/security), then choose **Report a vulnerability**. Please do not open a public issue for them.

Include the file or workflow affected, what an attacker could do, and the steps to reproduce it.

## Scope

Only the `main` branch is supported. In scope are the demos under `papers/`, the tests, the GitHub Actions workflows under `.github/workflows/`, and the dependency pins in `requirements.txt` and `constraints.txt`, plus the files that decide what CI installs: `requirements-ci.txt`, `ci-tools.txt` and `cpu.txt`.

The demos download public data and models: CIFAR-10 through torchvision and three open models from the Hugging Face Hub, each pinned to one revision. A problem in those upstream projects belongs with them, but a way this repo fetches or loads them unsafely is in scope.

## How the repo is protected

- Every GitHub Action is pinned to a full commit SHA, and every workflow starts with no token permissions and grants only what each job needs.
- CI installs from `requirements-ci.txt`, which pins every package to one version and lists the sha256 hash of each allowed file, so a changed or swapped package fails the install. It was generated with:

  ```bash
  uv pip compile requirements.txt ci-tools.txt -c constraints.txt --override cpu.txt --generate-hashes --python-version 3.12 --python-platform x86_64-manylinux_2_28 --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --no-header --emit-index-url
  ```

  where `ci-tools.txt` (repo root) pins pytest, pytest-cov, ruff and their test dependencies, and `cpu.txt` pins the `+cpu` builds of torch and torchvision.
- `main` accepts changes only through pull requests whose CI and CodeQL checks pass. Force pushes and branch deletion are blocked.
- CodeQL scans the Python code and the workflows on every push, every pull request and once a week.
- Each release, built by the `release` workflow from a tag on `main`, attaches the source archive and its build provenance, signed through Sigstore. Check an archive with:

  ```bash
  gh attestation verify papers-that-built-ai-v1.0.0.tar.gz --repo mlvpatel/papers-that-built-ai
  ```

- Secret scanning with push protection, Dependabot alerts and Dependabot security updates are on.
