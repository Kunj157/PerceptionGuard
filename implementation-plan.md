# Implementation Plan — Perception Model Safety & Regression Validation Pipeline

Maps the 7 components to concrete phases, where each runs (local vs Colab), and how each phase moves through the git workflow (issue → branch → PR → merge → release).

---

## Phase 0 — Repo & CI Scaffolding
**Where:** Local
**Do:**
- Init repo, `main` (empty) + `dev` branches.
- `.gitignore`: `/docs`, model weights, datasets, `.venv`.
- Base folder structure: `/src`, `/tests`, `/ci`, `/docs` (local-only), `/data` (gitignored).
- Add `AGENTS.md`, `README.md` stub.
- CI skeleton: GitHub Actions workflow that runs lint + tests on PR to `dev`.
**Git flow:** Issue "Repo scaffolding" → `feature/1-repo-scaffolding` → PR → `dev`.

---

## Phase 1 — Data Acquisition & Scenario Tagging (Component 2.1)
**Where:** Local
**Do:**
- Download filtered BDD100K subset (a few GB, fits your 49GB free disk).
- Parse existing weather/scene/time-of-day attributes into a metadata table (pandas → CSV/SQLite).
- Unit tests: tagging output schema, missing-attribute handling.
**Git flow:** Issue "Scenario tagging pipeline" → `feature/2-scenario-tagging` → PR → `dev` (link issue).

---

## Phase 2 — Coverage Analysis (Component 2.2)
**Where:** Local
**Do:**
- Groupby scenario combinations, compute sample counts, flag `adequate`/`low`/`critical` by threshold.
- Output: coverage table + a first draft coverage report (markdown/CSV).
- Unit tests: threshold logic, edge case of zero-sample bucket.
**Git flow:** Issue "Coverage analysis" → `feature/3-coverage-analysis` → PR → `dev`.

---

## Phase 3 — Frozen Regression Test Suite (Component 2.3)
**Where:** Local
**Do:**
- Curate ~500-image balanced subset across scenario buckets (deliberately include low-coverage ones).
- Freeze as `test_suite_v1/` + manifest (JSON) checked into repo.
- Unit test: manifest integrity (all listed files exist, no duplicates).
**Git flow:** Issue "Frozen test suite v1" → `feature/4-test-suite` → PR → `dev`.

---

## Phase 4 — Model Training (Component: input to 2.4)
**Where:** **Colab (GPU)**
**Do:**
- Fine-tune YOLOv8n on 2-3 different training subsets to produce checkpoints v1/v2/v3 (deliberately vary data so a real regression exists to detect).
- Export each checkpoint to ONNX in Colab before downloading.
- Download ONNX files locally; do **not** commit raw weights — store as GitHub Release assets or gitignored `/models`.
**Git flow:** Issue "Train baseline + regression checkpoints" → `feature/5-model-checkpoints` → PR → `dev` (PR contains export scripts + docs, not the weight files themselves).

---

## Phase 5 — Scenario-Tagged Eval Harness (Component 2.4)
**Where:** Local (CPU, OpenVINO)
**Do:**
- Load ONNX model via OpenVINO runtime.
- Run inference over `test_suite_v1`, compute mAP per scenario bucket.
- Unit tests: metric calc correctness on a tiny synthetic example.
**Git flow:** Issue "Eval harness" → `feature/6-eval-harness` → PR → `dev`.

---

## Phase 6 — Regression Comparison + Metrics DB (Components 2.5, 2.6)
**Where:** Local
**Do:**
- SQLite schema: `runs(commit_sha, timestamp, scenario_combo, mAP, verdict)`.
- Comparison logic against last baseline, configurable per-bucket thresholds (YAML).
- Unit tests: regression detection at exact threshold boundary, no-baseline-yet case.
**Git flow:** Issue "Regression comparison + history DB" → `feature/7-regression-check` → PR → `dev`.

---

## Phase 7 — CI Gate + Dashboard (Component 2.7)
**Where:** Local (dev) + GitHub Actions (runtime)
**Do:**
- CI workflow: on PR touching `models/`, run Phases 5-6, post pass/fail as a check + PR comment.
- Dashboard (Streamlit): reads SQLite DB, shows accuracy-by-scenario over time + coverage heatmap.
**Git flow:** Issue "CI gate workflow" → `feature/8-ci-gate` → PR → `dev`. Issue "Dashboard" → `feature/9-dashboard` → PR → `dev`.

---

## Phase 8 — Packaging & First Release
**Where:** Local
**Do:**
- Dockerfile wrapping eval harness + dashboard.
- Full run-through on `dev`: all components pass CI green.
- PR `dev` → `main`.
- On merge to `main`: release workflow (`release.yml`) tags a version, builds Docker image, attaches artifacts.
- CD workflow (`deploy.yml`) deploys dashboard (e.g. to a free host — Render/Fly.io) from the tagged release.
**Git flow:** Issue "v1.0 release" → PR `dev`→`main` → CI creates tag `v1.0.0` → CD deploys.

---

## Post-v1: Ongoing Feature/Bugfix Cycle
For every new feature or fix after v1:
1. Open issue describing it.
2. Branch from `dev`: `feature/<n>-<desc>` or `bugfix/<n>-<desc>`.
3. Sequential, small commits with detailed messages.
4. PR into `dev`, linked to issue, CI must pass.
5. Once `dev` is stable, PR `dev`→`main`, CI tags a new release, CD redeploys.

---

## Summary Table

| Phase | Location | Output |
|---|---|---|
| 0 | Local | Repo, CI skeleton |
| 1 | Local | Scenario-tagged dataset |
| 2 | Local | Coverage report |
| 3 | Local | Frozen test suite v1 |
| 4 | **Colab (GPU)** | 2-3 ONNX checkpoints |
| 5 | Local (CPU/OpenVINO) | Per-scenario eval harness |
| 6 | Local | Regression logic + metrics DB |
| 7 | Local + GH Actions | CI gate + dashboard |
| 8 | Local + GH Actions | Docker image, release, CD deploy |
