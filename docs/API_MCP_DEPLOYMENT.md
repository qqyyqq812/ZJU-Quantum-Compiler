# Public API and MCP Deployment

This project has two server surfaces. They are intentionally separate.

## Browser route

Normal visitors use the GitHub Pages page in `docs/index.html`.

```text
visitor browser
  -> GitHub Pages static HTML
  -> FastAPI REST backend
  -> src.server.app:app
  -> POST /api/compile
```

The browser does not call MCP. The page uses `PUBLIC_API_BASE` by default and
supports `?api=https://your-api.example` only for maintainers.

Deploy the REST API with:

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port $PORT
```

The checked-in Render blueprint is `render.yaml`, and its lightweight
dependency file is `requirements-api.txt`.

After deployment, verify:

```bash
curl https://your-api.example/api/status
curl https://your-api.example/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

The default `/api/compile` route is NPQR. A successful response should include
`"backend":"npqr"`, `algorithm`, `components`, `compiled_qasm`, and a SABRE
`baseline` object. Explicit SABRE comparison remains available with
`{"backend":"sabre","heuristic":"lookahead"}`.

Then set `PUBLIC_API_BASE` in `docs/index.html` to the deployed API origin if
the domain differs from the current value.

## MCP helper route

MCP is for AI clients, review workflows, and tool-style access. It is not the
normal browser Run path.

```text
MCP client
  -> remote MCP endpoint
  -> src.server.mcp_app:app
  -> POST /mcp
```

Deploy the optional MCP helper with:

```bash
uvicorn src.server.mcp_app:app --host 0.0.0.0 --port $PORT
```

The optional Render blueprint is `render-mcp.yaml`, and its dependency file is
`requirements-mcp.txt`. Verify the helper with:

```bash
curl https://your-mcp.example/health
```

## What still needs a human

- Create the Render services or connect the GitHub repository in the Render UI.
- Confirm the generated REST API domain.
- Update `PUBLIC_API_BASE` when Render gives a different domain.
- Keep secrets out of the repository; the current services do not need secrets.

## NPQR note

The current public REST runtime defaults to `backend="npqr"`. NPQR combines
neural beam inference, mapping selection, and bounded suffix repair. SABRE is
returned only as a comparison baseline and is not counted as NPQR fallback.
