# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-07-25

### Added
- Scenario tagger for BDD100K attribute parsing (weather, scene, time-of-day).
- Coverage analyzer with per-bucket adequate/low/critical flagging.
- Frozen regression test suite with balanced sampling and JSON manifest.
- Eval harness with IoU, AP, and per-scenario mAP computation.
- Regression checker with SQLite metrics DB, baseline comparison, and verdicts.
- CI gate GitHub Actions workflow with PR comments and merge blocking.
- Streamlit dashboard for accuracy trends and coverage heatmap.
- Release workflow triggered on dev → main PR merge.
