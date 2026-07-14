# Agent Learning Project

## Goal

Build a production-minded Agent application incrementally while helping the user understand and hand-write the core logic. Start from one HTTP endpoint and one LLM call, then expand only after the current module is understood and verified.

## Collaboration Style

- Communicate in Chinese unless the user requests otherwise.
- Work on exactly one module at a time. Do not start the next module until the user confirms.
- Keep each learning task small enough for roughly 30–60 minutes.
- Before coding, explain the module's purpose, inputs, outputs, boundaries, and acceptance criteria.
- By default, let the user hand-write core structure and logic. Provide interfaces, pseudocode, and focused hints first.
- Use progressive help when the user is stuck:
  1. Point to the relevant idea or file.
  2. Provide pseudocode or function signatures.
  3. Provide a complete implementation only when explicitly requested.
- If the user explicitly asks Codex to implement a change, implement and verify it normally.
- End each completed module with: what was learned, remaining technical debt, verification results, and a suggested commit message.

## Engineering Rules

- Prefer the smallest working vertical slice; avoid speculative abstractions, empty future directories, and premature infrastructure.
- Use Python, FastAPI, Pydantic, pytest, Ruff, MyPy, and `uv` unless the project later records a different decision.
- Start with the official model SDK and a single provider behind a small `LLMClient` boundary.
- Do not introduce LangChain, LangGraph, an Agents SDK, a vector database, Redis, or a message queue before the corresponding learning stage is approved.
- Keep API, application logic, model access, tools, context, memory, storage, and observability conceptually separate, but create modules only when they are needed.
- Prefer typed request, response, state, tool, and event models.
- Preserve existing user changes. Inspect relevant files before editing and avoid unrelated rewrites.
- Never commit secrets. Read credentials from environment variables and maintain a safe `.env.example` when configuration is introduced.

## Testing and Cost Control

- Add or update tests with every behavior change.
- Automated tests must use fake or mocked model clients by default.
- Keep paid/live API smoke tests separate and opt-in; do not run them without explicit user approval.
- For model calls, capture useful usage, latency, and error information without logging secrets or sensitive full prompts.
- Run the narrowest relevant tests first, then formatting, linting, typing, and broader tests when available.
- Do not rely on exact text matching for nondeterministic model behavior.

## Module Gate

A module is complete only when:

- its stated acceptance criteria are met;
- relevant tests pass;
- errors and important edge cases are handled;
- no unapproved future-stage functionality was added;
- the user can explain the main data flow and design choice.

After reporting completion, stop and wait for the user's instruction before moving on.

## Async Engineering

- Introduce asynchronous programming incrementally and explain each new mechanism before using it.
- Prefer async at I/O boundaries such as HTTP, model APIs, databases, and tools; keep pure computation synchronous.
- Never call blocking I/O directly inside the event loop.
- Bound concurrency with timeouts, task limits, and semaphores.
- Handle cancellation and resource cleanup explicitly.
- Test async behavior, timeouts, cancellation, partial failures, and concurrency limits.
- Give the user a minimal isolated exercise before introducing a new async pattern into the project.
## Initial Roadmap

Follow this order unless the user explicitly changes it:

1. Minimal FastAPI endpoint and one LLM call
2. Session and context management
3. Tool calling and the tool loop
4. Agent runtime/harness and checkpoints
5. Persistent memory and retrieval policies
6. Document retrieval and grounded answers
7. Planning, review, and guardrails
8. Multi-agent orchestration
9. Tracing, evaluations, security, and production hardening
10. Framework reimplementation and comparison
