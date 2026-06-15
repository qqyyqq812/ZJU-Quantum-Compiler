# AI collaboration disclosure

This project used AI tools as engineering assistants during development. The
tools supported code review, debugging, documentation editing, and interface
polishing. The project repository keeps the runnable source code, tests,
examples, and technical report as the source of truth.

## How AI tools were used

AI assistance was used in bounded review and implementation tasks:

- reviewing frontend layout and interaction details;
- checking REST, MCP, and command-line consistency;
- drafting and revising README and user-facing documentation;
- identifying stale project-process language in public files;
- proposing test updates for public release contracts.

## Human-controlled decisions

Project scope, algorithm claims, evidence boundaries, and final repository
content were selected by the project author. AI-generated suggestions were
accepted only when they matched the implemented code, tests, and measured
results.

## Verification

The public repository is validated through automated tests, readiness checks,
route-trace replay, and manual browser review. AI assistance does not replace
these checks. Public claims are based on checked-in code, checked-in examples,
and the evidence manifest returned by `GET /api/npqr/evidence`.
