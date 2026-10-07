# Anonymous release checklist

- Keep the review link pointed only to the anonymous repository.
- Do not include author names, affiliations, email addresses, personal URLs,
  acknowledgements, local absolute paths, or tracking links.
- Exclude Git history copied from a development repository.
- Exclude model weights, generated datasets, logs, cached credentials, and
  experiment dashboards unless they are explicitly intended for release.
- Inspect notebook metadata and image/PDF metadata before adding such files.
- Run `python scripts/audit_release.py` before preparing the anonymous archive.
- Create the final commit and push from the author's own Git configuration.

