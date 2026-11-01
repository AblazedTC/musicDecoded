# Security

Please don't open a public issue for security problems. Report them privately through GitHub:
**Security → Report a vulnerability** on this repository. You'll get a reply within a week.

In scope: the API (`apps/api`), the worker (`apps/worker`), the deploy setup (`deploy/`), and the apps'
handling of tokens, uploads and downloaded files.

Never include real secrets in a report. Secrets are never committed: the backend reads them from environment
variables (`.env.example`, `deploy/README.md`), iOS keys go in the gitignored `apps/ios/Config/Secrets.xcconfig`,
and CI uses GitHub Actions secrets (`apps/ios/README.md`). If you find one committed anywhere, report it.
