# PerceptionGuard

CI-gated regression testing for perception models. Flags safety-relevant scenario coverage gaps and blocks deployment on category-level accuracy drops.

## Overview

PerceptionGuard validates perception model reliability before deployment by correlating prediction accuracy with environmental conditions. It uses scenario-aware evaluation to detect when a model degrades in specific real-world contexts — rain, night, intersections — rather than relying on aggregate metrics alone.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI Pipeline                              │
│                                                                 │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐  │
│  │ Scenario  │──▶│  Coverage  │──▶│  Eval    │──▶│ Regression│  │
│  │  Tagger   │   │  Analyzer  │   │  Harness │   │  Checker  │  │
│  └──────────┘   └───────────┘   └──────────┘   └───────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐  │
│  │ BDD100K  │   │ Coverage  │   │  Per-    │   │  SQLite   │  │
│  │ Metadata │   │   Report  │   │Scenario  │   │  Metrics  │  │
│  └──────────┘   └───────────┘   │  mAP     │   │    DB     │  │
│                                  └──────────┘   └───────────┘  │
│                                        │                        │
│                                        ▼                        │
│                               ┌──────────────┐                  │
│                               │  Dashboard   │                  │
│                               │ (Streamlit)  │                  │
│                               └──────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Components

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Scenario Tagger** | Parses BDD100K weather/scene/time-of-day attributes into a structured metadata table. |
| 2 | **Coverage Analyzer** | Computes per-scenario sample counts and flags critical gaps in dataset coverage. |
| 3 | **Test Suite Curator** | Freezes a balanced ~500-image regression test suite across scenario buckets with JSON manifest. |
| 4 | **Eval Harness** | Runs ONNX models via OpenVINO and computes per-scenario mAP metrics. |
| 5 | **Regression Checker** | Compares per-category mAP against baseline thresholds and stores results in SQLite. |
| 6 | **CI Gate** | GitHub Actions workflow that runs the full eval pipeline on PR and blocks merge on regressions. |
| 7 | **Dashboard** | Streamlit app for accuracy-by-scenario trends and coverage heatmaps. |

## Tech Stack

- **Models:** YOLOv8n (exported to ONNX)
- **Inference:** OpenVINO (CPU)
- **Data:** BDD100K, pandas, SQLite
- **CI/CD:** GitHub Actions, Docker
- **Dashboard:** Streamlit

## Project Structure

```
PerceptionGuard/
├── src/                    # Application source code
├── tests/                  # Unit tests
├── ci/                     # CI workflow configs
├── data/                   # Datasets (gitignored)
├── models/                 # Model weights (gitignored)
├── docs/                   # Local-only documentation (gitignored)
├── AGENTS.md               # Coding conventions & workflow
├── implementation-plan.md  # Phased development plan
├── LICENSE                 # MIT
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_install_install_bibles/Installer.html)
- GitHub account (for CI/CD)

### Installation

```bash
git clone git@github.com:Kunj157/PerceptionGuard.git
cd PerceptionGuard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Locally

```bash
# Run scenario tagging pipeline
python -m src.scenario_tagger

# Run coverage analysis
python -m src.coverage_analyzer

# Run evaluation harness
python -m src.eval_harness

# Launch dashboard
streamlit run src/dashboard.py
```

## Development Workflow

1. Open a GitHub issue for the task.
2. Branch from `dev`: `feature/<issue#>-<short-desc>` or `bugfix/<issue#>-<short-desc>`.
3. Commit sequentially with small, single-purpose messages.
4. Open a PR into `dev`, linking the issue (`Closes #n`).
5. After `dev` is green, PR `dev` → `main`. CI tags the release.
6. CD deploys from the `main` release.

### Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Releases only. Protected. |
| `dev` | Integration branch. Default. |
| `feature/*` | Feature development. |
| `bugfix/*` | Bug fixes. |

**Never commit directly to `main` or `dev`.**

## Roadmap

| Phase | Deliverable | Status |
|-------|------------|--------|
| 0 | Repo scaffolding & CI skeleton | `main` |
| 1 | Scenario tagging pipeline | Pending |
| 2 | Coverage analysis | Pending |
| 3 | Frozen test suite v1 | Pending |
| 4 | Model checkpoints (ONNX) | Pending |
| 5 | Per-scenario eval harness | Pending |
| 6 | Regression logic + metrics DB | Pending |
| 7 | CI gate + dashboard | Pending |
| 8 | Docker image & first release | Pending |

## License

MIT License. See [LICENSE](LICENSE) for details.
