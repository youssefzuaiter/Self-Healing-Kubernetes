# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| Latest commit on `main` | Yes |
| Older commits, including the `stage-1b-frozen` tag | No |

## Reporting a vulnerability

Please don't report security problems in a public issue. Report them privately through
GitHub instead: open the repository's **Security** tab and choose **Report a vulnerability**,
or go directly to
[the new advisory form](https://github.com/youssefzuaiter/Self-Healing-Kubernetes/security/advisories/new).

Include the affected file and commit, the steps to reproduce the problem, and what an
attacker could do with it. The maintainer replies in the private advisory thread.

## Scope

This is a research simulator. It runs no network service, stores no credentials and handles
no user data. Reports are welcome about:

- code in this repository that can run unintended commands, or write files outside the
  directory it was given;
- the GitHub Actions workflow in `.github/workflows/tests.yml` and its permissions;
- known vulnerabilities in the dependencies pinned in `requirements.txt`.

## Known limits

These are documented design limits, not vulnerabilities:

- The safety shield's `ApprovedAction` token stops actions that bypass the shield through
  the public API. It is not unforgeable against code running in the same Python process,
  which can reach private state (`sim/shield.py`). It is a correctness guard, not a
  security boundary.
- Planned Stage 2 checkpoints are to be loaded with `weights_only=True`
  ([SPEC_03 Section 6.4](docs/SPEC_03_RL_TRAINING.md)). Loading checkpoints from
  untrusted sources is not supported.
