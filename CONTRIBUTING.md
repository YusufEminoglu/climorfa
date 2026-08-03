# Contributing

Thank you for your interest. This repository accompanies a manuscript currently
under review. We welcome corrections, clarifications, and improvements.

## Before you start

1. Check existing [issues](../../issues) to avoid duplicates.
2. For bugs, use the Bug Report template.
3. For methodological questions, use the Method Question template.
4. For security issues, do **not** open a public issue — see
   [SECURITY.md](SECURITY.md).

## Local checks before submitting

```bash
# Verify Python sources compile
python -m compileall -q src scripts

# Run citation audit (if data files are available)
python scripts/audit_manuscript_citations.py

# Verify no credentials are staged
grep -rE "(api_key|password|token|secret)" --include="*.py" src/ scripts/ || true
```

## Pull request checklist

- [ ] No restricted data files (rasters, GeoPackages, shapefiles) are included
- [ ] No manuscript PDFs or LaTeX auxiliary files are included
- [ ] No credentials, API keys, or local file paths are present
- [ ] All Python sources compile without errors
- [ ] Changes are described clearly in the PR body

## Code of conduct

Be respectful. This is a small research repository; assume good faith and
communicate directly.
