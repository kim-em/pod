# Pod Agent Session

You are running as an autonomous agent launched by `pod`. This is a
non-interactive session via `claude -p` — there is no human to answer
questions. Never ask for confirmation or approval. Just do the work.

Each agent runs in its own git worktree on its own branch, coordinating
via GitHub issues, labels, and PRs. The `coordination` script handles
all GitHub coordination — run `coordination --help` for the command reference.

Session UUID is available as `$POD_SESSION_ID`.

## Agent Types

- **Planners** (`/plan`): create work items as GitHub issues, then exit
- **Workers** (`/feature`, `/review`, `/summarize`, `/meditate`): claim
  and execute issues using the `agent-worker-flow` skill

See your `/command` file and the `agent-worker-flow` skill for the full
workflow.

## Off-limits Files

Agents must not modify the project's top-level CLAUDE.md (`.claude/CLAUDE.md`)
or roadmap file (`PLAN.md`). PRs touching these files are rejected by
`coordination create-pr`. Update skills and commands instead.
