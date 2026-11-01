# Security

Please don't open a public issue for security problems. Report them privately through GitHub:
**Security → Report a vulnerability** on this repository. You'll get a reply within a week.

In scope: the API (`apps/api`), the worker (`apps/worker`), the deploy setup (`deploy/`), and the apps'
handling of tokens, uploads and downloaded files.

Never include real secrets in a report. Secrets are only read from environment variables (see
`.env.example` and `deploy/README.md`); if you find one committed anywhere, that's worth reporting too.
