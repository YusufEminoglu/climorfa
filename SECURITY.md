# Security policy

## Reporting a vulnerability

If you discover a security issue — for example a residual credential, an API
key accidentally committed, or a path that exposes private infrastructure —
please **do not open a public issue**.

Instead, contact the corresponding author directly:

- **Yusuf Eminoğlu** — yusuf.eminoglu@deu.edu.tr

We will acknowledge your report within 72 hours and aim to resolve confirmed
issues within one week.

## Credential review

This repository's CI checks for common credential patterns on every push.
If a credential is inadvertently committed, it is revoked and the commit
history is cleaned before the repository is made public. The compiled
`paper/manuscript/src/main.pdf` is excluded from version control via
`.gitignore` to prevent accidental inclusion of publisher-formatted
material before publication terms are confirmed.
