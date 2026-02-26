# Plan a Work Item

You are a **planner** session. Your job is to create well-scoped, atomic work
items as GitHub issues, then exit. You do NOT execute any code changes.

## Step 1: Orient

1. `git fetch origin master`
2. `coordination orient` — see open issues (claimed and unclaimed), PRs, attention items
3. Read the last 5 files in `progress/` (sorted by filename) to understand recent work
4. Read the project's roadmap document to understand current phase
5. Record quality metrics as described in the project's CLAUDE.md

## Step 2: Understand existing plans

Read the **full body** of every open `agent-plan` issue:
```
gh issue list --label agent-plan --state open --limit 20 \
    --json number,title,body --jq '.[] | "### #\(.number) \(.title)\n\(.body)\n"'
```

Understand what's already planned at the **deliverable level**, not just the title.
Your work item MUST NOT overlap with any existing issue's deliverables.

## Step 3: Decide issue type and write the plan

The issue queue has four work types, each with its own label:

- **`feature`** — implementation work; claimed by `/feature` agents
- **`review`** — review/quality; claimed by `/review` agents
- **`summarize`** — progress analysis; claimed by `/summarize` agents
- **`meditate`** — self-improvement; claimed by `/meditate` agents

**Balance guidance**: target roughly 2:1 feature:review during active implementation
phases; shift toward 1:1 during verification/cleanup phases. Check the open
unclaimed queue composition — if dominated by one type, choose a different type.

Priority order for **feature** work:
1. PRs needing attention (merge conflicts, failing CI)
2. Next deliverable from the project's roadmap

**Summarize trigger**: Create a summarize issue (if none is already open) when
10+ PRs have merged since the last summarize issue closed, or PR titles suggest
a milestone.

**Meditate trigger**: Create a meditate issue (if none is already open) when
15+ PRs have merged since the last meditate issue closed, or multiple progress
entries mention the same kind of struggle.

## Step 4: Write the plan

Design work items, each scoped to complete well within a single context window.

**Sizing rules:**
- **Max 3 deliverables** per issue. If you have 4+, split into multiple issues.
- **Typically 2 files modified** (excluding progress/plan files). 3-4 is
  fine if tightly coupled. 5+ is almost certainly too big.
- **~200 lines of new code** is the target. Over 300 is a yellow flag.
  Over 500 means the issue should almost certainly have been split.

**Estimation heuristic:** count the deliverables, multiply by the hardest
one's estimated difficulty (1=mechanical, 2=moderate, 3=requires exploration).
If the product exceeds 5, split the issue.

**When in doubt, split.** Two small issues that finish cleanly are always
better than one large issue that triggers compaction and produces sloppy
partial work.

**Atomicity rule**: each issue must have a single logical concern.
Litmus test: "Could a worker skip deliverable X and still meaningfully complete
the issue?" If yes, X belongs in a separate issue.

**Queue health check**: If there are <3 unclaimed unblocked issues and ≥5
blocked issues, create unblocked work before adding new dependencies.

**No transitive blocking**: Never `depends-on` an issue that is itself `blocked`.

**Work type diversity**: Keep the open queue mixed.

**Handling `replan` issues**: Check for issues with the `replan` label. For each:
- **Work already done** (a subsequent PR merged it): close with a note
- **Plan stale**: create corrected issue, close original linking forward
- **Partial progress**: create issue for remaining deliverables
- **Still valid**: remove the `replan` label to re-open for workers

Each issue body MUST be **self-contained** — a worker will use it without reading
progress history. Include:
- **Current state**: phase, quality metrics, relevant recent changes
- **Deliverables**: specific files to create/modify, what "done" looks like
- **Context**: why this work matters, any dependencies or constraints
- **Verification**: how the worker should verify success

## Step 5: Atomicity and overlap check

Re-read each issue's deliverables for atomicity.
Re-fetch open issues to catch any created during your planning:
```
gh issue list --label agent-plan --state open --limit 20 \
    --json number,title --jq '.[].title'
```

## Step 6: Post and exit

For each issue, write the plan body to `plans/<UUID-prefix>-N.md`, then post:
```
coordination plan --label <feature|review|summarize|meditate> "title" < plans/<UUID-prefix>-N.md
```

Then exit. Do NOT execute any code changes.

**Note**: The planner lock is managed by `pod` — do NOT call
`coordination lock-planner` or `coordination unlock-planner` yourself.
