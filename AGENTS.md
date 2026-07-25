# AGENTS.md

Perception Model Safety & Regression Validation Pipeline. CV model CI/testing system.

## Stack
YOLOv8n, ONNX, OpenVINO (CPU), pandas, SQLite, GitHub Actions, Docker, Streamlit.

## Branches
- `main`: releases only, protected, empty until first release.
- `dev`: integration branch, default.
- `feature/<issue#>-<short-desc>`, `bugfix/<issue#>-<short-desc>`.
- Never commit directly to `main` or `dev`.

## Workflow
1. Open GitHub issue for the task.
2. Branch from `dev`.
3. Commit sequentially — small, single-purpose, detailed messages (what+why).
4. PR into `dev`, link issue (`Closes #n`).
5. After `dev` is green, PR `dev`→`main`, tag release via CI.
6. CD deploys from `main` release.

## Commits
`<type>: <summary>` + body. One logical change per commit. No bulk commits.

## Code hygiene
Small functions, no duplication, no dead code, no secrets. Check for a linter/complexity extension each session.

## Docs
`/docs` is gitignored — never commit it.

## Testing
Unit test per new component. CI must pass before any merge.

## Ask before
Changing branch/release strategy. Adding heavy new dependencies.
