"""Generate the unified NPQR evidence manifest used by API, website, MCP, and reports."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import DEFAULT_EVIDENCE_PATH, write_npqr_evidence_manifest

DEFAULT_DOCS_COPY = PROJECT_ROOT / "docs" / "assets" / "npqr_evidence_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--docs-copy", type=Path, default=DEFAULT_DOCS_COPY)
    parser.add_argument("--no-docs-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_npqr_evidence_manifest(args.output)
    if not args.no_docs_copy:
        args.docs_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.docs_copy)
    json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
