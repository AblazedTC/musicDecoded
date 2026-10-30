# Contributing to MusicDecoded

Thanks for helping. All work is tracked in [TASKS.md](TASKS.md), and every PR closes exactly one task from it.
Read [PLAN.md](PLAN.md) for the architecture and decisions, and [DESIGN.md](DESIGN.md) before touching UI.

## Setup by lane

Full instructions are in [docs/running-locally.md](docs/running-locally.md). Short version:

| Lane | Folder | You need | Build / test |
|---|---|---|---|
| `contracts` | `contracts/` | Python + `jsonschema`, Node | `python contracts/validate.py` · `npx @redocly/cli@1 lint contracts/openapi.yaml` |
| `api` | `apps/api` | [uv](https://docs.astral.sh/uv/); Docker to run it against Postgres | `uv run pytest` |
| `worker` | `apps/worker` | Python 3.14; FFmpeg and the model submodules (`git submodule update --init`) for real analyses | `pip install -r requirements-dev.txt && pytest` (unit tests need no models) |
| `android` | `apps/android` | Android Studio, JDK 21 | `./gradlew assembleDebug testDebugUnitTest` |
| `ios` | `apps/ios` | macOS, Xcode 16+, XcodeGen | `xcodegen generate`, then `xcodebuild test` (see `apps/ios/README.md`) |
| `infra` | `docker-compose.yml`, `.github/`, `scripts/`, docs | Docker | `docker compose up --build -d`, `scripts/e2e.sh` |
| `design` | `DESIGN.md` | | Review against the existing screens |

The whole stack (API + worker + Postgres + Redis + MinIO) runs with `docker compose up --build -d`.

## Picking and claiming a task

1. Open [TASKS.md](TASKS.md) and pick the lowest-numbered `todo` task in your lane whose `Deps` are all `done`.
2. Claim it: set `Status: in-progress` and `Owner` (your GitHub handle), commit that change on its own,
   and branch as `<lane>/T-###-slug` (for example `android/T-502a-soundcloud-link`).
   If you can't push to the repo, comment on a draft PR with the claim instead so nobody else picks it up.
3. Stay inside the task's acceptance criteria. Found something else? Add it as a new task (template at the top
   of TASKS.md) instead of fixing it on the side.
4. Prove every criterion (a test, command output, or a screenshot), tick the boxes, set `Status: done` with
   a one-line `Notes:`, and open a PR titled `T-###: <title>`.
5. Stuck, or need a product decision? Set `Status: blocked`, explain why in `Notes:`, and stop.

App tasks come in twins: `a` = Android, `i` = iOS. They share acceptance criteria and can be done in
parallel by different people.

### Good first tasks

Small, self-contained, and with no open dependencies:

| Task | Lane | What |
|---|---|---|
| T-213a / T-213i: seek bar | android / ios | Style the player seek bar per DESIGN.md (thin track, round thumb) instead of the stock slider |
| T-213a / T-213i: song-row art | android / ios | Song rows show the initial-letter art (as in the mini player) when there's no thumbnail |
| T-213a / T-213i: import sheet | android / ios | Present the import sheet over Home instead of on an empty screen |
| T-502a | android | SoundCloud link source: paste or share a SoundCloud URL to start a `soundcloud` job |
| T-502i | ios | The same SoundCloud link source on iOS |

The T-213 items are independent, so one item per PR is fine: tick only that box and leave the task `todo`
until all items are done.

## Lanes

`contracts`, `api`, `worker`, `android`, `ios`, `infra`, `design`. Edit only your lane's folder, plus your
task's entry in TASKS.md.

**Contracts first.** `contracts/` (`openapi.yaml`, `analysis.schema.json`, `chord-voicings.json`, fixtures) is
the interface between every lane. A change to it lands first, in its own `contracts` task and PR, before
any lane implements against it. A PR touching `contracts/**` runs every CI workflow.

## Pull requests

- **One task per PR**, titled `T-###: <title>`. Keep unrelated refactors out.
- **CI green.** Each lane's workflow runs on the paths it owns (`.github/workflows/`). Add or update tests
  for the behavior you change.
- **Contracts first** (above). Never change an API or analysis shape only in code.
- **UI changes need screenshots** (light and dark) from an emulator or simulator, checked against DESIGN.md.
- Fill in the PR template checklist.

### Commits

Short, one-line subjects in the imperative, prefixed with the task id when there is one:

```
T-502a: SoundCloud link source
contracts: add soundcloudUrl to JobCreate
```

No body needed for most commits. Squash noisy fixup commits before asking for review.

## Licensing and the CLA

| Part | License |
|---|---|
| iOS app, Android app, `contracts/`, docs | [MIT](LICENSE) |
| `apps/api`, `apps/worker` | [AGPL-3.0](apps/api/LICENSE) |

Contributions are accepted under the license of the folder they touch.

PRs that change **`apps/api` or `apps/worker`** also need a signed **Contributor License Agreement**. The
project owner runs the hosted service and its Pro plan from this same code and may need to relicense the
backend later (PLAN.md decision D-9); the CLA keeps that possible. You keep the copyright to your work.
CLA Assistant comments on your PR with a link the first time; sign it once
with your GitHub account and the check turns green. PRs that touch only the apps, contracts or docs don't
need it.

"MusicDecoded" and its logo are not covered by either license. Forks must use a different name.

## Reporting bugs and ideas

Use the issue templates (bug report, feature request). Feature ideas are discussed in an issue first; once
agreed, they become a task in TASKS.md.

## Maintainer setup

One-time steps for the repo owner. They aren't automated because they need owner access:

- Set up the CLA. The hosted [CLA Assistant](https://github.com/apps/cla-assistant) app checks every PR and
  has no per-path option, so to ask only on `apps/api/**` / `apps/worker/**` changes use the
  [CLA Assistant GitHub Action](https://github.com/contributor-assistant/github-action) in a workflow with
  those `paths` filters, plus a CLA document (for example based on the Apache Individual CLA) and a branch
  to store signatures. Either way the bot posts the signing link on the PR.
- Protect `main`: require PRs and the CI checks.
- Create the labels `good first issue`, `lane:contracts`, `lane:api`, `lane:worker`, `lane:android`,
  `lane:ios`, `lane:infra`, `lane:design`, and open issues for the good first tasks above.
