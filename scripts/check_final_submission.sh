#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

git diff --check
.venv/bin/python scripts/check_submission_readiness.py
.venv/bin/python -m pytest -s -q
