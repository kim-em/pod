"""pod — Multi-agent manager for Claude Code.

Manages concurrent autonomous Claude sessions with a TUI and CLI interface.
Each agent runs as an independent background process; the TUI is just a viewer.

Usage:
    pod                  # Interactive TUI
    pod init             # Bootstrap .pod/ in current git repo
    pod update           # Re-populate agent config from package
    pod add [N]          # Launch N new agents (default 1); updates target
    pod target N         # Set target agent count (auto-spawn/cap enforced)
    pod list             # Show running agents
    pod finish [ID|all]  # Signal agent(s) to finish after current work
    pod kill [ID|all]    # Kill agent(s) immediately (unclaims issues)
    pod status           # Queue depth, agent count, total cost
    pod cleanup          # Remove stale worktrees not owned by any agent
    pod log [ID]         # Tail agent's session stdout
    pod config           # Print current config
    pod coordination ... # Run bundled coordination script
"""

from __future__ import annotations

import argparse
import contextlib
import curses
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import random
import re
import signal
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

from pod import accounts
from pod import github as gh


def _gh_cli(*argv: str, timeout: float = 60,
            input: bytes | str | None = None,
            text: bool = True,
            check: bool = False) -> subprocess.CompletedProcess:
    """Run a `gh` porcelain subprocess through the layer.

    Returns a `subprocess.CompletedProcess` so call sites keep the
    `r.returncode` / `r.stdout` / `r.stderr` shape. Direct GitHub API
    reads should use `gh.get_client().get(...)` instead, to take
    advantage of ETag conditional requests."""
    return gh.get_client().gh_cli(*argv, timeout=timeout, input=input,
                                   text=text, check=check)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _find_project_dir() -> Path:
    """Walk up from cwd to find a directory containing .pod/."""
    d = Path.cwd().resolve()
    while True:
        if (d / ".pod").is_dir():
            return d
        if d.parent == d:
            break
        d = d.parent
    # Fallback: git root
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return Path(r.stdout.strip())
    except Exception:
        pass
    return Path.cwd()


def _data_dir() -> Path:
    """Locate bundled data files from the installed package."""
    return Path(__file__).parent / "data"


PROJECT_DIR = _find_project_dir()
POD_DIR = PROJECT_DIR / ".pod"
AGENTS_DIR = POD_DIR / "agents"
CONFIG_PATH = POD_DIR / "config.toml"
LOG_PATH = POD_DIR / "pod.log"
CLAIM_HISTORY_PATH = POD_DIR / "claim-history.json"
PR_CLAIM_HISTORY_PATH = POD_DIR / "pr-claim-history.json"
HOUSEKEEPING_LOCK_PATH = POD_DIR / "housekeeping.lock"
HOUSEKEEPING_STAMP_PATH = POD_DIR / "housekeeping.stamp"
SHARED_ROTATE_LOCK_PATH = POD_DIR / "shared-account-rotate.lock"
SHARED_ROTATE_STAMP_PATH = POD_DIR / "shared-account-rotate.stamp"
ISOLATED_CONFIG_DIR = POD_DIR / "claude-config"
TARGET_FILE = POD_DIR / "target"  # Target agent count (int, one per line)
PLANNER_TARGET_FILE = POD_DIR / "planner-target"  # Planner-recommended target agent count
PLANNER_MIN_QUEUE_FILE = POD_DIR / "planner-min-queue"  # Planner-recommended min_queue
FORCE_QUOTA_FILE = POD_DIR / "force-quota"  # If exists, skip quota checks globally
_AUTH_EXPIRY_SKEW = 60  # Seconds: treat a token expiring within this window as already expired for dispatch

# ---------------------------------------------------------------------------
# Default configuration (written on first run)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = """\
# pod configuration — all values have sensible defaults.
# Edit this file to customise agent behaviour.

[project]
worktree_base = "worktrees"        # Where git worktrees are created
session_dir = "sessions"           # Session stdout capture directory
build_cache_dir = ".lake"          # Build cache seeded into worktrees
build_cache_symlink_subdirs = ["packages"]  # Subdirs of build_cache_dir to symlink
                                            # instead of copy (immutable shared deps).
                                            # Set to [] to copy everything.
protected_files = ["PLAN.md"]      # Files agents may not modify in PRs
min_free_disk_gb = 2               # Pause dispatch when free disk drops below this

[merge]
# Status check contexts that must pass before a PR can auto-merge.
# These are the `name:` field of jobs in .github/workflows/*.yml.
# Leave empty to skip branch-protection setup.
#
# Why this matters: `coordination create-pr` runs `gh pr merge --auto` on
# every PR it opens, but GitHub only honours --auto when the default branch
# has branch protection with at least one required status check. Without
# that, PRs go green but never merge.
#
# Example: required_checks = ["build-and-test"]
required_checks = []

[agent]
backend = "claude"                 # "claude", "codex", or "auto"
# In "auto" mode, each iteration picks whichever backend has available
# quota. Settings under [agent.auto] apply only to auto mode.
quota_retry_seconds = 60           # Auto-mode sleep when no backend has quota

[agent.auto]
prefer = "codex"                   # Tie-break when both backends have quota

[agent.claude]
model = "opus"                     # Claude model to use (back-compat default)
# Ordered list of acceptable model tiers; first satisfiable per account
# wins. Set to ["opus"] to require Opus, ["opus", "sonnet"] to fall
# through to Sonnet when no account has Opus quota. Absent → [model].
accepted_models = ["opus"]
quota_check = "~/.claude/skills/claude-usage/claude-available-model"
quota_check_required = true        # Hard-fail if quota unavailable
quota_retry_seconds = 60           # Sleep duration when quota unavailable
# When isolated_config = true, each agent gets its own
# CLAUDE_CONFIG_DIR under .pod/claude-config/<short_id>/ with its own
# keychain entry, so different agents can simultaneously run on
# different Claude accounts.
isolated_config = true             # Use per-agent CLAUDE_CONFIG_DIR for agents

[agent.codex]
model = "gpt-5.4"                  # Codex model to use
quota_check = "~/.claude/skills/claude-usage/codex-available-model"
quota_check_required = true        # Hard-fail if quota unavailable
quota_retry_seconds = 60           # Sleep duration when quota unavailable
isolated_config = true             # Use strict pod-managed CODEX_HOME (no ~/.codex state except auth.json)

[agent.bubble]
# Run each agent inside a `bubble` container instead of directly on the
# host. The host's ~/.claude/CLAUDE.md, skills/, commands/, and
# settings.json are NOT exposed to agents; only .credentials.json is
# mounted (read-only) so subscription auth still works. Requires the
# `bubble` CLI on PATH and a working Incus runtime. When enabled,
# `[agent.claude] isolated_config` is ignored: transcripts land in
# ~/.bubble/ai-projects/<bubble-name>/ and are read from there.
enabled = false

[quota]
# When an external account manager owns account selection — signalled by
# ~/.claude/.current-account, which pins pod to one account (see
# accounts.select_for_dispatch) — pod drives that manager itself rather than
# relying on its (typically systemd-user-timer) schedule, which can wedge on
# an unprivileged host across a `nixos-rebuild switch`. On the housekeeping
# cadence, and before sleeping on `waiting_quota`, pod runs `<cmd> best` to
# re-pick the best-quota account and refresh tokens, and un-wedges a stuck
# `systemd --user` rotation timer via `daemon-reexec`. Set to "" to disable;
# a no-op when ~/.claude/.current-account is absent or the command is missing.
account_manager_cmd = "~/.claude/swap-account"
# Max seconds a resuming agent will hold its claim/worktree while waiting for
# quota to return before giving up and releasing them.
max_pause_secs = 5400

# Dollars per million tokens.
# Resolution order when pricing an agent (see `_pricing_for`):
#   [pricing.<backend>.<model>]
#   → [pricing.<backend>.default]
#   → legacy flat [pricing]
#   → baked-in Claude Opus constants
# Legacy flat [pricing] remains honoured for backward compatibility.

[pricing.claude.opus]
input = 5.00
output = 25.00
cache_read = 0.50
cache_create = 6.25

[pricing.claude.default]
input = 5.00
output = 25.00
cache_read = 0.50
cache_create = 6.25

[pricing.codex."gpt-5.4"]
input = 2.50
output = 15.00
cache_read = 0.25
cache_create = 0.0

[pricing.codex.default]
input = 2.50
output = 15.00
cache_read = 0.25
cache_create = 0.0

[dispatch]
# Built-in strategies: "queue_balance", "round_robin"
# Or a path to a custom script (receives env vars, prints worker type name)
strategy = "queue_balance"
min_queue = 3                      # queue_balance: below this → planner-type

[repair]
# A PR is considered "stuck" if a required check has been IN_PROGRESS
# longer than this many minutes. Used by `coordination list-pr-repair`.
stuck_ci_minutes = 120

[security]
# Pod ingests issue/comment text into agent prompts. On a public repo,
# anyone can post that text unless GitHub interaction limits are set.
# Pod refuses to dispatch agents on a public repo whose interaction
# limit is missing, weaker than `minimum_interaction_limit`, or expiring
# in less than `minimum_expiry_days`.
#
# To set/renew the limit:
#   gh api -X PUT repos/<owner>/<repo>/interaction-limits \
#     -f limit=collaborators_only -f expiry=six_months
enforce_interaction_limits = true
minimum_interaction_limit = "collaborators_only"  # or "contributors_only", "existing_users"
minimum_expiry_days = 7

# Per-message author provenance gate. On top of the repo-wide
# interaction-limit check, refuse to surface issue bodies / comments
# whose authors don't have a trusted association with the repo.
# See README for the threat model and limits, including the org-
# membership caveat for `MEMBER`.
trust_only_collaborators = true
trusted_author_associations = ["OWNER", "MEMBER", "COLLABORATOR"]
trusted_users = []                 # bot allowlist
                                   # e.g. ["dependabot[bot]", "github-actions[bot]"]
provenance_cache_seconds = 60      # applies to list/orient; claim/read are always fresh

[monitor]
poll_interval = 2                  # Seconds between status updates
jsonl_stale_warning = 300          # Warn if JSONL unchanged for this many seconds
jsonl_missing_warning = 60         # Warn if JSONL not created after this many seconds
max_claim_restarts = 1             # Max times to auto-restart a dead session before releasing claim
show_costs = true                  # Show estimated API costs in TUI and status
stuck_initial_timeout = 3600       # Seconds since last assistant output before first stuck check
stuck_confirm_timeout = 1200       # Seconds to wait before confirming kill (after first detection)
stuck_check_interval = 30          # Seconds between process health checks during stuck detection

# --- Worker Types ---
# Each [worker_types.<name>] defines a type of agent session.
# The dispatch strategy chooses among these.

[worker_types.plan]
prompt = "/plan"
lock = "planner"                   # Acquire this lock before running
copy_build_cache = false

[worker_types.work]
prompt = "/work"
copy_build_cache = true

# Repair agent: handles unhealthy PRs (merge conflicts, failed CI, stuck CI).
# Only selected when `coordination list-pr-repair` reports candidates (see
# `dispatch_queue_balance`). `copy_build_cache` is enabled so repair agents
# can run the project's verification before pushing.
[worker_types.repair]
prompt = "/repair"
copy_build_cache = true

# Replan agent: triages issues labelled `replan` (from PRs closed
# unsalvageable, `coordination skip --replan`, or `create-pr --partial`).
# Shares the planner lock with `/plan` so the two cannot race on issue
# state. Dispatched by the replan short-circuit in `dispatch_queue_balance`,
# ordered after `repair` and before `plan`.
[worker_types.replan]
prompt = "/replan"
lock = "planner"
copy_build_cache = false

# Example additional worker type:
# [worker_types.review]
# prompt = "/review"
# copy_build_cache = true
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agent_config_sync_check():
    """Compare pod template commands/skills with the project .claude/ and report/update.

    Only runs for Claude backend — Codex config is installed per-session
    into CODEX_HOME, not into the project tree.

    Uses .claude/.pod-checksums to track what pod last installed, enabling
    three-way detection:
      - pod updated, project unchanged  → auto-overwrite
      - project customised, pod unchanged → print note (user customisation)
      - both changed independently       → warn, don't overwrite
      - no checksums file yet            → bootstrap: record current hashes,
                                           warn on any differences
    """
    # Only meaningful for Claude backend — Codex installs per-session
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                cfg = tomllib.load(f)
            if cfg.get("agent", {}).get("backend", "claude") != "claude":
                return
        except (OSError, tomllib.TOMLDecodeError):
            pass
    data_config = _data_dir() / "agent-config" / "claude"
    proj_claude = PROJECT_DIR / ".claude"
    checksums_file = proj_claude / ".pod-checksums"

    # Collect all template files under commands/ and skills/
    template_files: list[Path] = []
    for subdir in ("commands", "skills"):
        src = data_config / subdir
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    template_files.append(f.relative_to(data_config))

    if not template_files:
        return

    stored: dict[str, str] = {}
    parse_ok = False
    if checksums_file.exists():
        try:
            stored = json.loads(checksums_file.read_text())
            parse_ok = True
        except Exception:
            stored = {}

    first_run = not checksums_file.exists() or not parse_ok
    updated: list[str] = []
    custom: list[str] = []
    conflicts: list[str] = []

    for rel in template_files:
        key = str(rel)
        pkg_file = data_config / rel
        proj_file = proj_claude / rel

        pkg_hash = _sha256(pkg_file)
        proj_hash = _sha256(proj_file) if proj_file.exists() else None
        inst_hash = stored.get(key)

        if pkg_hash == proj_hash:
            continue  # identical, nothing to do

        if first_run or inst_hash is None:
            # No tracking history yet — can't distinguish customisation from old version
            if proj_file.exists():
                conflicts.append(key)  # report as unknown difference
            else:
                updated.append(key)    # new file in template, install it
        elif proj_hash == inst_hash:
            # Project unchanged since install, template has been updated
            updated.append(key)
        elif pkg_hash == inst_hash:
            # Template unchanged, project has been customised
            custom.append(key)
        else:
            # Both changed independently
            conflicts.append(key)

    # Apply auto-updates
    for key in updated:
        src = data_config / key
        dst = proj_claude / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    # Write checksums: record the pod-template hash for every file we installed or
    # that is already in sync.  Do NOT record first-run conflicts — leave their
    # inst_hash absent so they continue to be reported as unknown on future runs.
    skip_keys = set(conflicts) if first_run else set()
    new_checksums = dict(stored)
    for rel in template_files:
        key = str(rel)
        if key in skip_keys:
            continue
        pkg_file = data_config / rel
        # After any copy the project file matches the template; record pkg hash
        # so future runs can distinguish "user edited" from "pod updated".
        new_checksums[key] = _sha256(pkg_file)
    checksums_file.write_text(json.dumps(new_checksums, sort_keys=True, indent=2) + "\n")

    if updated:
        print(f"pod: updated {len(updated)} file(s) from new pod version: "
              f"{', '.join(updated)}", file=sys.stderr)
    if custom:
        print(f"pod: {len(custom)} project-customised file(s) differ from pod template: "
              f"{', '.join(custom)}", file=sys.stderr)
    if conflicts and not first_run:
        print(f"pod: WARNING: {len(conflicts)} file(s) modified in both pod template and "
              f"project .claude/ — not auto-updating: {', '.join(conflicts)}", file=sys.stderr)
    elif conflicts and first_run:
        print(f"pod: {len(conflicts)} file(s) differ from pod template "
              f"(origin unknown, tracking from now): {', '.join(conflicts)}", file=sys.stderr)


def ensure_config() -> dict:
    """Load config, requiring pod init to have been run."""
    if not CONFIG_PATH.exists():
        print(f"No .pod/config.toml found. Run 'pod init' first.", file=sys.stderr)
        sys.exit(1)
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    _agent_config_sync_check()
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    cfg = _migrate_legacy_config(cfg)
    validate_security_config(cfg)
    return cfg


def _claude_isolation_enabled(config: dict) -> bool:
    """True iff Claude agents should use a per-agent CLAUDE_CONFIG_DIR.

    Returns False when the configured backend is Codex-only or when
    ``isolated_config`` is explicitly off for Claude.
    """
    if _backend(config) == "codex":
        return False
    return bool(_backend_cfg(
        config, "isolated_config", backend="claude", default=False))


def get_isolated_config_dir(config: dict,
                              short_id: str | None = None) -> Path | None:
    """Return the CLAUDE_CONFIG_DIR for ``short_id``, or None if disabled.

    With per-agent isolation enabled, each agent gets its own
    ``.pod/claude-config/<short_id>/`` directory. Passing
    ``short_id=None`` returns the root claude-config dir (callers that
    don't yet know the agent id, e.g. legacy code paths).
    """
    if not _claude_isolation_enabled(config):
        return None
    if short_id is None:
        return ISOLATED_CONFIG_DIR
    return accounts.agent_claude_config_dir(POD_DIR, short_id)


def ensure_isolated_config(config: dict,
                            short_id: str | None = None) -> Path | None:
    """Materialise the agent's CLAUDE_CONFIG_DIR. Returns path or None if disabled.

    Creates the per-agent dir with ``settings.json`` and ``projects/``.
    The ``.credentials.json`` file is *not* written here — it lands
    when ``accounts.mirror_canonical_to_isolated`` runs at lease
    acquisition (which selects the specific account to use).
    """
    if not _claude_isolation_enabled(config):
        return None
    if short_id is None:
        # Legacy callers (very rare): fall back to the shared dir.
        ISOLATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return ISOLATED_CONFIG_DIR
    return accounts.ensure_agent_claude_config_dir(POD_DIR, short_id)


def _instruction_file_warning(git_root: Path, backend: str) -> str | None:
    """Return a warning if instruction files are inconsistent for the backend.

    `auto` mode requires both AGENTS.md and .claude/CLAUDE.md (since either
    backend may run in any iteration).
    """
    agents = git_root / "AGENTS.md"
    claude = git_root / ".claude" / "CLAUDE.md"

    agents_exists = agents.exists() or agents.is_symlink()
    claude_exists = claude.exists() or claude.is_symlink()

    needs_claude = backend in ("claude", "auto")
    needs_agents = backend in ("codex", "auto")

    if not agents_exists and not claude_exists:
        return None
    if agents_exists and not claude_exists:
        if needs_claude:
            return ("warning: repo has AGENTS.md but no .claude/CLAUDE.md, "
                    f"while pod is configured for {backend}\n"
                    "    → Prefer AGENTS.md -> .claude/CLAUDE.md for cross-backend parity.")
        return None
    if claude_exists and not agents_exists:
        if needs_agents:
            return ("warning: repo has .claude/CLAUDE.md but no AGENTS.md, "
                    f"while pod is configured for {backend}\n"
                    "    → Prefer AGENTS.md -> .claude/CLAUDE.md for cross-backend parity.")
        return None

    try:
        if agents.is_symlink() and agents.resolve() == claude.resolve():
            return None
    except OSError:
        pass
    try:
        if claude.is_symlink() and claude.resolve() == agents.resolve():
            return None
    except OSError:
        pass

    return ("warning: AGENTS.md and .claude/CLAUDE.md both exist but are not symlinked together\n"
            "    → Prefer AGENTS.md -> .claude/CLAUDE.md for cross-backend parity.")


def _claude_projects_dir(claude_config_dir: Path | None = None) -> Path:
    """Return the directory containing JSONL project subdirs."""
    if claude_config_dir is not None:
        return claude_config_dir / "projects"
    return Path.home() / ".claude" / "projects"


def _use_bubble(config: dict) -> bool:
    """Whether agents should run inside `bubble` containers."""
    return bool(cfg_get(config, "agent", "bubble", "enabled", default=False))


def _bubble_name(session_uuid: str) -> str:
    """Deterministic per-session bubble container name."""
    return f"pod-{session_uuid[:8]}"


def _bubble_in_container_repo(wt_dir: str) -> str:
    """Path inside the bubble where the worktree's repo is cloned.

    Bubble clones into /home/user/<repo-name>; pod's worktrees are named
    after their branch but the in-container clone uses the repo name.
    """
    # Bubble names the in-container clone after the upstream repo, not
    # the worktree directory. Read it from the worktree's git config.
    try:
        url = subprocess.check_output(
            ["git", "-C", wt_dir, "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # e.g. git@github.com:owner/name.git -> name
        name = url.rsplit("/", 1)[-1].removesuffix(".git")
        return f"/home/user/{name}"
    except (subprocess.CalledProcessError, OSError):
        return f"/home/user/{Path(wt_dir).name}"


def _bubble_jsonl_dir(bubble_name: str, wt_in_bubble: str) -> Path:
    """Host-visible path where Claude (running inside the bubble) writes
    its JSONL transcript. Bubble mounts ~/.bubble/ai-projects/<name>/
    at /home/user/.claude/projects/.
    """
    encoded = wt_in_bubble.replace("/", "-")
    return Path.home() / ".bubble" / "ai-projects" / bubble_name / encoded


def cfg_get(config: dict, *keys, default=None):
    """Nested dict lookup with default."""
    d = config
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _backend(config: dict) -> str:
    """Return the agent backend name.

    Returns 'claude', 'codex', or 'auto'. Callers that need a concrete
    backend (not 'auto') should use `_select_backend()` instead, or pass
    the per-iteration choice explicitly to `_backend_cfg(..., backend=)`.
    """
    return cfg_get(config, "agent", "backend", default="claude")


def _backend_cfg(config: dict, *keys, backend: str | None = None, default=None):
    """Read a backend-specific config value.

    e.g. `_backend_cfg(config, 'model')` reads `agent.<backend>.model`.

    Pass `backend=` to read settings for a specific backend (required in
    `auto` mode, where the global `agent.backend` value isn't a concrete
    backend name).
    """
    b = backend if backend is not None else _backend(config)
    return cfg_get(config, "agent", b, *keys, default=default)


def _migrate_legacy_config(cfg: dict) -> dict:
    """Migrate legacy top-level [claude] section to [agent.claude]."""
    if "claude" not in cfg:
        return cfg
    agent = cfg.get("agent", {})
    if "claude" in agent:
        raise SystemExit(
            "Config error: both [claude] and [agent.claude] exist — "
            "remove the legacy [claude] section"
        )
    log("Deprecation: [claude] section is now [agent.claude]; migrating in memory")
    cfg.setdefault("agent", {}).setdefault("backend", "claude")
    cfg["agent"]["claude"] = cfg.pop("claude")
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def log(msg: str):
    """Append timestamped message to pod.log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as f:
                f.write(line)
        except OSError:
            pass


def say(msg: str):
    """Print to stderr and log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    log(msg)


# Route pod.accounts diagnostics through pod's logger so everything
# lands in .pod/pod.log under the same timestamp format.
accounts.set_logger(log)


# ---------------------------------------------------------------------------
# Claim history — persists which issues our local sessions have claimed
# ---------------------------------------------------------------------------

CLAIM_HISTORY_LOCK_PATH = POD_DIR / "claim-history.lock"


@contextlib.contextmanager
def _claim_history_filelock():
    """Cross-process exclusive lock for claim-history.json via fcntl.flock."""
    CLAIM_HISTORY_LOCK_PATH.touch()
    fd = open(CLAIM_HISTORY_LOCK_PATH)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


@contextlib.contextmanager
def _housekeeping_filelock():
    """Non-blocking exclusive lock for the housekeeping sweep.

    Yields True if the caller acquired the lock and owns this housekeeping
    cycle, False if another agent already owns it (caller should skip).
    The OS releases the lock automatically if the holder is killed mid-sweep.
    """
    POD_DIR.mkdir(parents=True, exist_ok=True)
    HOUSEKEEPING_LOCK_PATH.touch()
    fd = open(HOUSEKEEPING_LOCK_PATH)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError):
            yield False
            return
        yield True
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _housekeeping_due(interval: float) -> bool:
    """True if no housekeeping has completed within the last `interval` seconds.

    Reads the stamp file written at the end of the previous successful sweep.
    Treats a missing or unparseable stamp as "due now" so a fresh project
    or one with a corrupted stamp doesn't get stuck."""
    try:
        last = float(HOUSEKEEPING_STAMP_PATH.read_text().strip())
    except (OSError, ValueError):
        return True
    return (time.time() - last) > interval


def _housekeeping_mark_done():
    """Record the timestamp of a successful housekeeping sweep. Written at the
    *end* of the sweep so a crash mid-sweep doesn't suppress the next one."""
    try:
        HOUSEKEEPING_STAMP_PATH.write_text(f"{time.time()}\n")
    except OSError as e:
        log(f"Failed to update housekeeping stamp: {e}")


# --- Shared-mode credential rotation ----------------------------------------
#
# When pod defers account choice to an external manager (a `swap-account`-style
# script, signalled by ``~/.claude/.current-account`` — see
# ``accounts.select_for_dispatch``), that manager is normally driven by a
# periodic ``systemd --user`` timer. On an unprivileged host that timer is
# fragile: a ``nixos-rebuild switch`` reexecs the user manager and can wedge
# its timer clock (next-elapse computed to ``infinity``, no future trigger),
# and the only durable fix — ``loginctl enable-linger`` — needs root. When that
# happens the pinned account goes stale: every agent sits in ``waiting_quota``
# even though a sibling account still has headroom.
#
# pod is itself an always-on unprivileged process, so it *drives* the manager
# instead of depending on the timer: it re-runs ``<manager> best`` (which
# re-picks the best-quota account and refreshes tokens) on the housekeeping
# cadence and whenever an agent would otherwise sleep on ``waiting_quota`` —
# and, since ``systemctl --user daemon-reexec`` needs no privilege, it
# un-wedges the timer when it detects it stuck. Every path is rate-limited to
# one run per interval across the whole fleet, and the whole feature is a no-op
# unless pod is in shared mode with the manager command present.

_USER_ROTATE_TIMERS = (
    "claude-credential-swap.timer",
    "claude-keep-alive.timer",
)


@contextlib.contextmanager
def _shared_rotate_filelock():
    """Non-blocking lock so only one agent rotates per window (others skip)."""
    POD_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_ROTATE_LOCK_PATH.touch()
    fd = open(SHARED_ROTATE_LOCK_PATH)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError):
            yield False
            return
        yield True
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _shared_account_manager_cmd(config: dict) -> str | None:
    """Path to the external account manager to drive, or None when disabled.

    Enabled only when pod is deferring account choice to an external manager
    (``~/.claude/.current-account`` present) and the manager command exists on
    disk. The command defaults to ``~/.claude/swap-account`` and can be
    overridden — or disabled with ``""`` — via ``[quota] account_manager_cmd``.
    """
    if accounts.current_account() is None:
        return None
    cmd = cfg_get(config, "quota", "account_manager_cmd",
                  default="~/.claude/swap-account")
    if not cmd:
        return None
    path = os.path.expanduser(cmd)
    return path if os.path.exists(path) else None


def _user_rotate_timers_wedged() -> bool:
    """True if a ``systemd --user`` rotation timer is active but has no future
    trigger — the post-reexec wedge a ``daemon-reexec`` clears. Fail-safe to
    False on any error or non-systemd host."""
    for timer in _USER_ROTATE_TIMERS:
        try:
            r = subprocess.run(
                ["systemctl", "--user", "show", timer,
                 "-p", "ActiveState",
                 "-p", "NextElapseUSecRealtime",
                 "-p", "NextElapseUSecMonotonic"],
                capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if r.returncode != 0:
            continue
        props = dict(
            ln.split("=", 1) for ln in r.stdout.splitlines() if "=" in ln)
        if props.get("ActiveState") != "active":
            continue
        # A healthy active timer has a finite next elapse on at least one clock.
        real = props.get("NextElapseUSecRealtime", "").strip()
        mono = props.get("NextElapseUSecMonotonic", "").strip()
        if not real and mono in ("", "infinity"):
            return True
    return False


def _heal_wedged_user_timers() -> None:
    """Un-wedge ``systemd --user`` rotation timers via ``daemon-reexec``.

    No-op unless ``_user_rotate_timers_wedged()``. ``daemon-reexec`` needs no
    privilege and preserves running units — pod agents are plain subprocesses,
    not user units, so they are unaffected."""
    if not _user_rotate_timers_wedged():
        return
    log("shared-account: user rotation timer wedged — running "
        "`systemctl --user daemon-reexec` to re-arm it")
    for argv in (["systemctl", "--user", "daemon-reexec"],
                 ["systemctl", "--user", "restart", *_USER_ROTATE_TIMERS]):
        try:
            subprocess.run(argv, capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _maybe_rotate_shared_account(config: dict, *, min_interval: float) -> None:
    """Drive the external account manager to re-pick the best-quota account and
    refresh tokens, at most once per ``min_interval`` seconds across all agents.
    No-op outside shared mode. See the section comment above."""
    cmd = _shared_account_manager_cmd(config)
    if cmd is None:
        return

    def _recent() -> bool:
        try:
            last = float(SHARED_ROTATE_STAMP_PATH.read_text().strip())
        except (OSError, ValueError):
            return False
        return (time.time() - last) < min_interval

    if _recent():
        return
    with _shared_rotate_filelock() as owns:
        if not owns or _recent():
            return
        try:
            subprocess.run([cmd, "best"], capture_output=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as e:
            log(f"shared-account: `{cmd} best` failed: {e}")
        else:
            _heal_wedged_user_timers()
        try:
            SHARED_ROTATE_STAMP_PATH.write_text(f"{time.time()}\n")
        except OSError:
            pass


def load_claim_history() -> dict:
    """Load {issue_num_str -> {session_uuid, short_id, restart_count}}."""
    try:
        return json.loads(CLAIM_HISTORY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_claim_history(history: dict):
    """Atomically write claim history."""
    tmp = CLAIM_HISTORY_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(history, indent=2) + "\n")
        tmp.rename(CLAIM_HISTORY_PATH)
    except OSError as e:
        log(f"Failed to save claim history: {e}")


def record_claim(issue: int, session_uuid: str, short_id: str):
    """Record that our session claimed this issue. Preserves restart_count.

    If a *different* session re-claims an issue that was previously released,
    the new claim overwrites the old released entry (with restart_count=0).
    If the *same* session tries to re-add its own released claim, we skip
    to avoid a feedback loop where a resumed session triggers infinite
    restart spawning.
    """
    with _claim_history_filelock():
        history = load_claim_history()
        key = str(issue)
        existing = history.get(key, {})
        if existing.get("released"):
            if existing.get("session_uuid") == session_uuid:
                return  # Don't re-add our own released claim (prevents restart loops)
            # Different session re-claimed this issue — allow tracking the new claim
        history[key] = {
            "session_uuid": session_uuid,
            "short_id": short_id,
            "restart_count": existing.get("restart_count", 0) if not existing.get("released") else 0,
        }
        _save_claim_history(history)


def clear_claim(issue: int, session_uuid: str | None = None):
    """Mark issue as released in claim history, but only if the entry belongs
    to session_uuid (compare-and-swap).  If session_uuid is None, releases
    unconditionally (legacy callers / PR-created path)."""
    with _claim_history_filelock():
        history = load_claim_history()
        key = str(issue)
        entry = history.get(key)
        if not entry:
            return
        if session_uuid and entry.get("session_uuid") != session_uuid:
            return  # Entry belongs to a different session — don't clobber
        entry["released"] = True
        _save_claim_history(history)


# ---------------------------------------------------------------------------
# PR claim history (for repair agents)
# ---------------------------------------------------------------------------
# Mirrors issue-claim-history but keyed by PR number. A repair agent claims
# a PR via `coordination claim-pr-repair`, which records an entry here. The
# housekeeping loop (check_dead_pr_claimed_prs) releases stale claims whose
# owning session has died, to keep the `repair-claimed` label from leaking.

def load_pr_claim_history() -> dict:
    """Load {pr_num_str -> {session_uuid, short_id, claimed_at}}."""
    try:
        return json.loads(PR_CLAIM_HISTORY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_pr_claim_history(history: dict):
    tmp = PR_CLAIM_HISTORY_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(history, indent=2) + "\n")
        tmp.rename(PR_CLAIM_HISTORY_PATH)
    except OSError as e:
        log(f"Failed to save PR claim history: {e}")


def record_pr_claim(pr: int, session_uuid: str, short_id: str):
    """Record that our session is repairing PR #pr. Reuses the issue-claim
    filelock since writes are rare and concurrency is low."""
    with _claim_history_filelock():
        history = load_pr_claim_history()
        history[str(pr)] = {
            "session_uuid": session_uuid,
            "short_id": short_id,
            "claimed_at": time.time(),
        }
        _save_pr_claim_history(history)


def clear_pr_claim(pr: int, session_uuid: str | None = None):
    """Remove the PR-claim entry, compare-and-swap on session_uuid if given."""
    with _claim_history_filelock():
        history = load_pr_claim_history()
        key = str(pr)
        entry = history.get(key)
        if not entry:
            return
        if session_uuid and entry.get("session_uuid") != session_uuid:
            return
        history.pop(key, None)
        _save_pr_claim_history(history)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Mutable state for one agent, serialised to .pod/agents/<id>.json."""
    short_id: str = ""
    uuid: str = ""
    backend_session_id: str = ""  # Backend-native id (Codex thread_id, or same as uuid for Claude)
    pid: int = 0
    pid_start_time: float = 0.0    # /proc start time — detects PID reuse
    worker_type: str = ""          # e.g. "work", "plan"
    status: str = "starting"       # starting, running, waiting_quota, finishing, stopped
    session_start: float = 0.0
    claimed_issue: int = 0
    pr_number: int = 0
    repair_pr: int = 0             # PR currently claimed by this agent for repair
    git_start: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_create: int = 0
    last_text: str = ""
    last_activity: float = 0.0
    finishing: bool = False
    force_quota: bool = False      # Skip quota check for this agent
    lock_held: str = ""            # Name of lock held (e.g. "planner"), or ""
    loop_iteration: int = 0
    worktree: str = ""
    branch: str = ""
    resume_session_uuid: str = ""  # If set, first iteration uses this UUID (to resume conversation)
    backend: str = "claude"        # "claude" or "codex" (for per-backend pricing lookup)
    model: str = ""                # e.g. "opus", "gpt-5.4" (for per-model pricing lookup)
    # Per-agent account lease (Claude only). `account_label` is the
    # currently-held lease; "" when waiting or running Codex.
    # `lease_acquired_at` is a wall-clock timestamp used for stale-lease
    # debugging and orphan eviction. `claude_config_dir` is the per-agent
    # CLAUDE_CONFIG_DIR so the TUI and crash-recovery can locate the
    # isolated config without recomputing the short_id mapping.
    account_label: str = ""
    lease_acquired_at: float = 0.0
    claude_config_dir: str = ""
    # One-shot mode: agent claims and works on `target_issue` once with the
    # given `target_type` prompt, then exits. Set by `pod once`. Bypasses
    # dispatch_queue_balance, the planner / replan / repair lock dance, and
    # the dispatch quota cap. `force_quota` is set in parallel so the agent
    # doesn't park on `select_available_backend`.
    target_issue: int = 0
    target_type: str = ""
    # Quota pause/resume. When an agent hits a usage limit, instead of aborting
    # (releasing the claim and deleting the worktree) it pauses — keeping its
    # claim, worktree and session — waits for quota to return (swap-account
    # rotates the canonical to a fresh account, or the 5h window resets), then
    # `--resume`s and continues on whatever account is now available.
    # `resuming_after_pause` makes the next loop iteration take the resume path
    # (skip dispatch/reset, reuse the worktree, `--resume`). `quota_paused_at`
    # is the wall-clock of the first pause in a run; it bounds the total pause.
    resuming_after_pause: bool = False
    quota_paused_at: float = 0.0

    def cost(self, config: dict) -> float:
        """Calculate cost in dollars using backend/model-aware pricing."""
        pricing = _pricing_for(config, self.backend, self.model)
        return (
            self.tokens_in * pricing.get("input", 5.0) / 1e6
            + self.cache_create * pricing.get("cache_create", 6.25) / 1e6
            + self.cache_read * pricing.get("cache_read", 0.50) / 1e6
            + self.tokens_out * pricing.get("output", 25.0) / 1e6
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AgentState:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def write(self):
        """Atomically write state to .pod/agents/<id>.json."""
        path = AGENTS_DIR / f"{self.short_id}.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
            tmp.rename(path)
        except OSError as e:
            log(f"Failed to write state for {self.short_id}: {e}")

    def remove_file(self):
        """Remove the state file."""
        path = AGENTS_DIR / f"{self.short_id}.json"
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_regular_agent(a: AgentState) -> bool:
    """True iff `a` counts against the regular `target` pool.

    Regular agents are spawned by `pod target N` / auto-spawn and live
    inside the dispatch quota cap. One-shot agents (set by `pod once`)
    explicitly bypass quota and live *outside* the pool — every
    accountant that consults `target` (auto-spawn `running` count,
    `cmd_add` setting `new_target`, `cmd_finish` / `cmd_kill`
    decrementing target on signal/kill) must filter them out via this
    predicate, otherwise launching a single one-shot worker silently
    reduces the regular pool's effective size for the duration of the
    priority job.
    """
    return a.status not in ("stopped", "dead") and not a.target_issue


def _abort_one_shot_iteration(state: AgentState, reason: str) -> None:
    """Mark a one-shot agent as `finishing` so its iteration loop
    exits on the next check instead of cycling.

    `pod once` semantics promise a single iteration; failure-path
    `continue`s in `agent_process_main` would otherwise retry the
    dispatch (re-creating worktrees, re-launching the model, possibly
    looping on a permanently broken environment). Call this from each
    failure branch before the loop's `continue`/`break`. No-op for
    regular agents.
    """
    if state.target_issue:
        log(f"Agent {state.short_id}: one-shot mode aborting "
            f"({reason}, target_issue=#{state.target_issue})")
        state.finishing = True


def read_all_agents() -> list[AgentState]:
    """Read all agent state files, filtering out stale (dead process) ones."""
    agents = []
    if not AGENTS_DIR.exists():
        return agents
    for p in sorted(AGENTS_DIR.glob("*.json")):
        if p.suffix != ".json" or p.name.endswith(".tmp"):
            continue
        try:
            d = json.loads(p.read_text())
            agent = AgentState.from_dict(d)
            # Check if process is still alive (and not a reused PID)
            if agent.pid > 0:
                if not _pid_is_valid(agent.pid, agent.pid_start_time):
                    agent.status = "dead"
            agents.append(agent)
        except (json.JSONDecodeError, OSError):
            continue
    return agents


def _read_commented_int(path: Path) -> int | None:
    """Read the first non-comment integer from a small state file."""
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return int(stripped)
    except (OSError, ValueError):
        return None
    return None


def _write_commented_int(path: Path, value: int, comments: list[str]):
    """Write a small state file with comment headers plus a numeric payload."""
    header = "\n".join(f"# {comment}" for comment in comments)
    path.write_text(f"{header}\n{max(0, value)}\n")


def read_target() -> int | None:
    """Read target agent count from .pod/target, or None if not set."""
    return _read_commented_int(TARGET_FILE)


def write_target(n: int):
    """Write target agent count to .pod/target."""
    _write_commented_int(TARGET_FILE, n, [
        "User target agent count.",
        "pod tries to keep this many agents running.",
    ])


def read_planner_target() -> int | None:
    """Read planner-recommended target from .pod/planner-target, or None if not set."""
    return _read_commented_int(PLANNER_TARGET_FILE)


def read_planner_min_queue() -> int | None:
    """Read planner-recommended min_queue from .pod/planner-min-queue, or None if not set."""
    return _read_commented_int(PLANNER_MIN_QUEUE_FILE)


def get_effective_target() -> int | None:
    """Effective target = min(user_target, planner_target). User target is the ceiling."""
    user_target = read_target()
    planner_target = read_planner_target()
    if user_target is not None and planner_target is not None:
        return min(user_target, planner_target)
    return user_target  # planner_target alone doesn't create a target; user must set one


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_pid_start_time(pid: int) -> float:
    """Get process start time from /proc (Linux). Returns 0 if unavailable."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            # Field 22 (1-indexed) is starttime; split past last ')' for comm
            parts = f.read().split(")")[-1].split()
            return float(parts[19])  # starttime = field 22 - 2 (pid,comm) = index 19
    except (OSError, IndexError, ValueError):
        return 0.0


def _pid_is_valid(pid: int, expected_start: float) -> bool:
    """Check if PID is alive AND belongs to the expected process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # Exists but we can't signal it — still valid
    if expected_start > 0:
        actual = _get_pid_start_time(pid)
        if actual > 0 and actual != expected_start:
            return False  # PID reused by a different process
    return True


def _check_process_stuck(pid: int) -> tuple[bool, str]:
    """Check if a claude subprocess (process group leader) appears stuck.

    Checks two hard signals for the entire process group:
      1. CPU usage: sum of %CPU across all group members must be < 0.5%
      2. No network connections: no TCP/UDP sockets open by any group member

    Child process state is logged for diagnostics but not used as a gate.

    Returns (is_stuck, detail) where detail is a human-readable string.
    Both hard signals must indicate stuck for is_stuck=True.
    If any check fails, conservatively returns (False, ...).
    """
    details = []

    # --- Get all processes in the process group ---
    try:
        result = subprocess.run(
            ["ps", "ax", "-o", "pid=,pgid=,%cpu=,state="],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return (False, f"ps failed: {e}")

    group_procs: list[tuple[int, float, str]] = []  # (pid, cpu, state)
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            p_pid = int(parts[0])
            p_pgid = int(parts[1])
            p_cpu = float(parts[2])
            p_state = parts[3]
        except (ValueError, IndexError):
            continue
        if p_pgid == pid:
            group_procs.append((p_pid, p_cpu, p_state))

    if not group_procs:
        return (True, "process group empty (all processes gone)")

    # --- CPU check (hard signal) ---
    total_cpu = sum(cpu for _, cpu, _ in group_procs)
    if total_cpu >= 0.5:
        return (False, f"CPU active: {total_cpu:.1f}% across {len(group_procs)} processes")
    details.append(f"cpu={total_cpu:.1f}%")

    # --- Child state (diagnostic only, not a gate) ---
    children = [(p, s) for p, _, s in group_procs if p != pid]
    zombie_children = [(p, s) for p, s in children if s.startswith("Z")]
    active_children = [(p, s) for p, s in children if not s.startswith("Z")]
    if children:
        if active_children:
            details.append(f"children={len(children)} ({len(active_children)} active, "
                           f"{len(zombie_children)} zombie)")
        else:
            details.append(f"children={len(children)} all zombie")
    else:
        details.append("no children")

    # --- Network connections (hard signal) ---
    all_pids = [str(p) for p, _, _ in group_procs]
    pid_list_str = ",".join(all_pids)
    try:
        result = subprocess.run(
            ["lsof", "-i", "-a", "-p", pid_list_str],
            capture_output=True, text=True, timeout=10,
        )
        # lsof returns exit 1 when no matches found (normal for "no connections")
        net_lines = [l for l in result.stdout.strip().splitlines()
                     if l and not l.startswith("COMMAND")]
        if net_lines:
            return (False, f"network connections: {len(net_lines)} open sockets")
        details.append("no network")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return (False, f"lsof failed: {e}")

    # --- Both hard signals indicate stuck ---
    return (True, "; ".join(details))


def _kill_stuck_subprocess(proc: subprocess.Popen) -> None:
    """Kill a stuck claude subprocess and its entire process group.

    Uses SIGTERM then SIGKILL escalation (same as _sigterm_handler).
    Does NOT exit the agent process or do cleanup — the caller's monitor
    loop detects proc.poll() != None and proceeds through normal
    session-end logic (unclaiming, worktree cleanup, etc.).
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return  # Already dead

    # SIGTERM first
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return

    # Wait briefly for graceful shutdown
    try:
        proc.wait(timeout=5)
        return  # Exited cleanly
    except subprocess.TimeoutExpired:
        pass

    # SIGKILL — force kill
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass

    # Final wait to reap
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        log(f"WARNING: failed to reap stuck subprocess PID {proc.pid} after SIGKILL")


def human_size(n: int) -> str:
    if n >= 1048576:
        return f"{n / 1048576:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n}B"


def human_duration(secs: int | float) -> str:
    secs = int(secs)
    if secs >= 3600:
        return f"{secs // 3600}h{secs % 3600 // 60:02d}m"
    if secs >= 60:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs}s"


def timeago(iso_ts: str) -> str:
    """Convert ISO 8601 timestamp to relative time string like '2h ago'."""
    if not iso_ts:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        delta = datetime.datetime.now(datetime.timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 0:
            return ""
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except (ValueError, TypeError):
        return ""


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)


_BAKED_IN_PRICING = {
    "input": 5.00,
    "output": 25.00,
    "cache_read": 0.50,
    "cache_create": 6.25,
}


def _pricing_for(config: dict, backend: str, model: str) -> dict:
    """Resolve the pricing dict for a (backend, model) pair.

    Fallback order:
      pricing.<backend>.<model>
        → pricing.<backend>.default
        → legacy flat [pricing]
        → baked-in Claude Opus constants
    """
    pricing_root = (config or {}).get("pricing") or {}
    backend_table = pricing_root.get(backend) if isinstance(pricing_root, dict) else None
    if isinstance(backend_table, dict):
        specific = backend_table.get(model) if model else None
        if isinstance(specific, dict):
            return specific
        default = backend_table.get("default")
        if isinstance(default, dict):
            return default
    # Legacy flat [pricing] = {"input": ..., "output": ..., ...}
    if isinstance(pricing_root, dict) and all(
        isinstance(v, (int, float)) for v in pricing_root.values()
    ) and pricing_root:
        return pricing_root
    return _BAKED_IN_PRICING


def token_summary(state: AgentState, config: dict, show_costs: bool = True) -> str:
    total_in = state.tokens_in + state.cache_read + state.cache_create
    if total_in == 0 and state.tokens_out == 0:
        return ""
    if show_costs:
        cost = state.cost(config)
        return f"{fmt_tokens(total_in)}/{fmt_tokens(state.tokens_out)}~${cost:.2f}"
    return f"{fmt_tokens(total_in)}/{fmt_tokens(state.tokens_out)}"


def _price_tokens(pricing: dict, tokens_in: int, tokens_out: int,
                  cache_read: int, cache_create: int) -> float:
    return (
        tokens_in * pricing.get("input", 5.0) / 1e6
        + cache_create * pricing.get("cache_create", 6.25) / 1e6
        + cache_read * pricing.get("cache_read", 0.50) / 1e6
        + tokens_out * pricing.get("output", 25.0) / 1e6
    )


def _claude_historical_cost(config: dict,
                             claude_config_dir: Path | None) -> float:
    """Scan Claude JSONL rollouts for this project and return total $."""
    # Compute the Claude projects-dir prefix for this project.
    # Claude Code munges paths: replace / with -, strip leading dots from components.
    def _claude_dir_name(p: Path) -> str:
        parts = str(p).split("/")
        cleaned = [part.lstrip(".") for part in parts]
        return "-".join(cleaned)

    project_prefix = _claude_dir_name(PROJECT_DIR)

    # Scan both ~/.claude/projects (historical) and isolated config (new sessions)
    projects_dirs = []
    real_projects = Path.home() / ".claude" / "projects"
    if real_projects.is_dir():
        projects_dirs.append(real_projects)
    if claude_config_dir is not None:
        iso_projects = claude_config_dir / "projects"
        if iso_projects.is_dir():
            projects_dirs.append(iso_projects)

    total_in = 0
    total_out = 0
    total_cache_read = 0
    total_cache_create = 0

    for projects_dir in projects_dirs:
        for d in projects_dir.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            # Match this project's dirs (main + worktrees)
            if not (name == project_prefix or name.startswith(project_prefix + "-")):
                continue
            for f in d.glob("*.jsonl"):
                try:
                    with open(f) as fh:
                        for line in fh:
                            if '"usage"' not in line or '"assistant"' not in line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if rec.get("type") != "assistant":
                                continue
                            usage = rec.get("message", {}).get("usage", {})
                            total_in += usage.get("input_tokens", 0)
                            total_out += usage.get("output_tokens", 0)
                            total_cache_read += usage.get("cache_read_input_tokens", 0)
                            total_cache_create += usage.get("cache_creation_input_tokens", 0)
                except OSError:
                    continue

    # Historical Claude runs predate the per-session backend/model manifest,
    # so price them at the Claude default tier.
    pricing = _pricing_for(config, "claude", "")
    return _price_tokens(pricing, total_in, total_out, total_cache_read, total_cache_create)


def _read_codex_manifest(session_uuid: str) -> dict | None:
    path = _codex_manifest_dir() / f"{session_uuid}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _session_uuid_from_rollout(rollout_path: Path) -> str | None:
    """Extract the session UUID from a Codex rollout filename.

    Codex names rollouts `rollout-<ISO-timestamp>-<session-uuid>.jsonl`.
    When pod is the launcher, the session-uuid matches the pod session_uuid
    used for the captured stdout file and the manifest sidecar.
    """
    name = rollout_path.name
    # Expected form: rollout-2026-04-22T02-30-18-<uuid>.jsonl
    if not name.startswith("rollout-") or not name.endswith(".jsonl"):
        return None
    stem = name[len("rollout-"):-len(".jsonl")]
    # The UUID is the last 5 dash-separated groups (8-4-4-4-12).
    parts = stem.rsplit("-", 5)
    if len(parts) < 6:
        return None
    return "-".join(parts[1:])


def _rollout_final_token_count(path: Path) -> tuple[int, int, int]:
    """Return (input_tokens, cached_input_tokens, output_tokens) from the
    last token_count event in a Codex rollout.

    `total_token_usage` is cumulative, so the last value is the full-session
    total. Returns (0, 0, 0) if the file has no token_count events.
    """
    last = (0, 0, 0)
    try:
        with open(path) as fh:
            for line in fh:
                if '"token_count"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = rec.get("payload", {})
                if payload.get("type") != "token_count":
                    continue
                info = payload.get("info") or {}
                total = info.get("total_token_usage") or {}
                last = (
                    total.get("input_tokens", 0),
                    total.get("cached_input_tokens", 0),
                    total.get("output_tokens", 0),
                )
    except OSError:
        return (0, 0, 0)
    return last


def _codex_historical_cost(config: dict) -> float:
    """Scan project-local Codex rollouts and stdout captures.

    Rollouts are authoritative; for any session_uuid with both a rollout
    and a .stdout, the .stdout is skipped to avoid double counting. This
    covers pre-relocation sessions (where only .stdout exists) and
    post-relocation sessions (where the rollout exists).
    """
    # 1. Rollouts (post-relocation, plus any legacy files that got moved in).
    rollout_root = _codex_rollouts_dir()
    seen_uuids: set[str] = set()
    total = 0.0
    if rollout_root.is_dir():
        for rollout in rollout_root.rglob("rollout-*.jsonl"):
            sid = _session_uuid_from_rollout(rollout)
            if sid:
                seen_uuids.add(sid)
            tokens_in, cached, tokens_out = _rollout_final_token_count(rollout)
            if tokens_in == 0 and tokens_out == 0:
                continue
            manifest = _read_codex_manifest(sid) if sid else None
            model = (manifest or {}).get("model", "")
            pricing = _pricing_for(config, "codex", model)
            # Codex's `input_tokens` is total input; `cached_input_tokens`
            # is a subset that is also counted under input. Price the
            # cached portion at the cache_read rate and the rest at the
            # full input rate.
            non_cached_in = max(0, tokens_in - cached)
            total += _price_tokens(pricing, non_cached_in, tokens_out, cached, 0)

    # 2. Pre-relocation `.stdout` captures: sum per-turn deltas, skip any
    # uuid already covered by a rollout.
    stdout_dir = PROJECT_DIR / cfg_get(config, "project", "session_dir", default="sessions")
    if stdout_dir.is_dir():
        for stdout_path in stdout_dir.glob("*.stdout"):
            sid = stdout_path.stem
            if sid in seen_uuids:
                continue
            s_in = s_cached = s_out = 0
            try:
                with open(stdout_path) as fh:
                    for line in fh:
                        if '"turn.completed"' not in line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("type") != "turn.completed":
                            continue
                        usage = rec.get("usage", {})
                        s_in += usage.get("input_tokens", 0)
                        s_cached += usage.get("cached_input_tokens", 0)
                        s_out += usage.get("output_tokens", 0)
            except OSError:
                continue
            if s_in == 0 and s_out == 0:
                continue
            manifest = _read_codex_manifest(sid)
            model = (manifest or {}).get("model", "")
            pricing = _pricing_for(config, "codex", model)
            non_cached_in = max(0, s_in - s_cached)
            total += _price_tokens(pricing, non_cached_in, s_out, s_cached, 0)

    return total


def compute_historical_cost(config: dict,
                            claude_config_dir: Path | None = None) -> float:
    """Scan project rollouts (Claude + Codex) and return total estimated cost.

    Uses the same counting methodology as the JSONL monitor (sum all
    assistant records for Claude; sum final token_count for Codex rollouts
    or per-turn `turn.completed` for legacy stdout captures).
    """
    return (
        _claude_historical_cost(config, claude_config_dir)
        + _codex_historical_cost(config)
    )


def _reload_config_value(*keys, default=None):
    """Re-read a single value from config.toml on disk (hot-reload safe)."""
    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        # Support legacy [claude] config
        if "claude" in cfg and "agent" not in cfg:
            cfg.setdefault("agent", {})["claude"] = cfg.pop("claude")
            cfg["agent"].setdefault("backend", "claude")
        for k in keys:
            cfg = cfg[k]
        return cfg
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return default


_MODEL_TIER: dict[str, dict[str, int]] = {
    "claude": {"opus": 2, "sonnet": 1, "haiku": 0},
    "codex": {"gpt-5.4": 2, "gpt-5.3-codex": 2, "gpt-5.4-mini": 1, "gpt-5": 2},
}


def _model_tier(backend: str, model: str) -> int:
    """Return the capability tier (higher = more capable) for a model."""
    return _MODEL_TIER.get(backend, {}).get(model, 0)


# Keychain service name + credential resolution moved to `pod.accounts`
# so the lease layer and cli.py share one source of truth. Keep
# back-compat aliases at module scope for any callers that still expect
# the cli-level names.
_claude_keychain_service = accounts.claude_keychain_service
resolve_claude_credential = accounts.resolve_claude_credential


def _backend_available(config: dict, backend: str,
                        claude_config_dir: Path | None = None) -> bool:
    """Legacy single-account quota check. True iff ``backend``'s quota
    helper reports ≥ the configured model tier.

    Kept for back-compat with ``select_available_backend`` (and a couple
    of tests). The agent loop calls ``acquire_backend`` instead, which
    enumerates every Claude account and atomically leases the best one.
    Does *not* mutate the keychain — credential mirroring is now part
    of lease acquisition in ``acquire_backend``.
    """
    cmd = os.path.expanduser(_backend_cfg(config, "quota_check", backend=backend, default=""))
    if not cmd:
        return True
    quota_required = _backend_cfg(config, "quota_check_required", backend=backend, default=True)
    args = [cmd]
    if backend == "claude":
        # Use the module-level alias so existing tests can patch
        # `cli.resolve_claude_credential` and intercept this call.
        resolved = resolve_claude_credential(claude_config_dir)
        label = resolved.get("accountLabel", "")
        if label and label != "?":
            args += ["--account", label]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 2 and not quota_required:
            # Exit 2 = cache missing/stale; proceed if quota check is optional
            log(f"quota_check ({backend}) returned 2 (unknown); proceeding because quota_check_required=false")
            return True
        if result.returncode != 0:
            return False
        available = result.stdout.strip()
        # Re-read model from disk so config.toml edits take effect without restart.
        required = _reload_config_value("agent", backend, "model", default="opus")
        # Model tier: higher-tier availability satisfies lower-tier requirements.
        return _model_tier(backend, available) >= _model_tier(backend, required)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _auto_backend_order(config: dict) -> list[str]:
    """Return the backends to try in `auto` mode, preferred first."""
    prefer = cfg_get(config, "agent", "auto", "prefer", default="codex")
    if prefer not in ("claude", "codex"):
        prefer = "codex"
    other = "claude" if prefer == "codex" else "codex"
    return [prefer, other]


def select_available_backend(config: dict, force: bool = False,
                              claude_config_dir: Path | None = None) -> str | None:
    """Legacy single-account backend picker, returning just a backend name.

    Wraps ``_backend_available`` over the configured backend(s) and
    returns the first one whose quota check passes. New code should
    use ``acquire_backend``, which enumerates every Claude account and
    holds a host-wide lease so two agents can't race onto the same one.
    """
    raw = cfg_get(config, "agent", "backend", default="claude")
    if raw == "auto":
        order = _auto_backend_order(config)
        if force or FORCE_QUOTA_FILE.exists():
            return order[0]
        for b in order:
            if _backend_available(config, b, claude_config_dir):
                return b
        return None
    # Fixed-backend mode
    if force or FORCE_QUOTA_FILE.exists():
        return raw
    if _backend_available(config, raw, claude_config_dir):
        return raw
    return None


def check_quota(config: dict, force: bool = False,
                claude_config_dir: Path | None = None) -> bool:
    """Back-compat shim: True iff some backend has available quota now.

    New code should call ``acquire_backend()`` directly so it knows
    *which* backend, account, and model to launch.
    """
    return select_available_backend(config, force=force,
                                    claude_config_dir=claude_config_dir) is not None


# --- Per-agent acquire / release ------------------------------------------


def _claude_accepted_models(config: dict | None = None) -> list[str]:
    """Return the ordered list of accepted Claude model tiers.

    Reads ``[agent.claude].accepted_models`` from config on disk (so
    edits take effect without restarting pod). Falls back to
    ``[agent.claude].model`` for back-compat with older configs.
    """
    raw = _reload_config_value("agent", "claude", "accepted_models", default=None)
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    fallback = _reload_config_value("agent", "claude", "model", default="opus")
    return [fallback]


def _release_account_lease(state: AgentState,
                             claude_config_dir: Path | None) -> None:
    """Release the agent's Claude account lease and harvest any refreshed
    OAuth token back to canonical ``~/.claude/credentials<N>.json``.

    Idempotent: a no-op when ``state.account_label`` is empty. Clears
    the lease bookkeeping on state. Safe to call from cleanup paths
    (normal iteration end, quota-exhaust, rapid-failure backoff,
    SIGTERM, exception handlers).
    """
    label = state.account_label
    if not label:
        return
    # Shared-credential mode (external account manager owns selection): no
    # lease was taken and the credential is a symlink to the canonical file,
    # so there is nothing to harvest or release.
    if state.lease_acquired_at == 0.0:
        state.account_label = ""
        return
    if claude_config_dir is not None:
        for a in accounts.list_claude_accounts():
            if a.label == label:
                try:
                    accounts.harvest_isolated_to_canonical(
                        label, a.number, claude_config_dir)
                except OSError as e:
                    log(f"lease: harvest failed for {label}: {e}")
                break
    try:
        accounts.release_lease(label, state.short_id)
    except Exception as e:
        log(f"lease: release failed for {label}: {e}")
    state.account_label = ""
    state.lease_acquired_at = 0.0


def acquire_backend(
    config: dict,
    *,
    state: AgentState,
    claude_config_dir: Path | None,
    force: bool = False,
    pin_label: str | None = None,
    pin_backend: str | None = None,
) -> accounts.Candidate | None:
    """Pick and (for Claude) atomically lease the best (backend, account,
    model) triple for this iteration.

    Resolution order, all of it under the lease meta-lock for the
    Claude path:

    1. If we already hold a lease that still satisfies (the helper still
       reports a model tier ≥ one of ``accepted_models``), keep it.
       Avoids release/re-acquire churn between iterations.
    2. Otherwise enumerate every Claude account (and Codex if allowed),
       filter to accepted tiers, order by ``[agent.auto].prefer``, then
       for each candidate try the lease — on success re-probe with
       ``--force`` to confirm quota didn't drop between bulk probe and
       acquisition; if it did, release and continue.
    3. On commit, mirror the canonical ``credentials<N>.json`` into the
       agent's isolated keychain entry (skipping the write if the
       isolated entry is already fresher, e.g. mid-session refresh).

    Returns ``None`` when every candidate is exhausted or leased by
    other agents — the caller should treat that as ``waiting_quota``.

    ``pin_backend``/``pin_label`` are set during resume to keep the
    session on the backend (and Claude account) it originally ran on.
    """
    short_id = state.short_id
    raw = pin_backend or cfg_get(config, "agent", "backend", default="claude")
    force = force or FORCE_QUOTA_FILE.exists()

    accepted_models = _claude_accepted_models(config)
    codex_model = _reload_config_value("agent", "codex", "model", default="gpt-5.4")

    claude_quota_cmd = _backend_cfg(
        config, "quota_check", backend="claude", default="")
    codex_quota_cmd = _backend_cfg(
        config, "quota_check", backend="codex", default="")

    if raw == "codex":
        backends_allowed = {"codex"}
    elif raw == "claude":
        backends_allowed = {"claude"}
    else:
        backends_allowed = {"claude", "codex"}
    prefer = cfg_get(config, "agent", "auto", "prefer", default="codex")
    if prefer not in ("claude", "codex"):
        prefer = "codex"

    # Step 1: existing lease still good? Skipped in shared mode (external
    # manager owns selection): there is no lease, and we must re-link the
    # live canonical and re-check the marker every dispatch.
    if (state.account_label
            and not force
            and "claude" in backends_allowed
            and accounts.current_account() is None
            and (pin_label is None or pin_label == state.account_label)):
        fresh = accounts.probe_account(
            claude_quota_cmd, state.account_label, force=False)
        if fresh:
            avail_tier = accounts.model_tier("claude", fresh)
            for tier in accepted_models:
                if avail_tier >= accounts.model_tier("claude", tier):
                    acct = next(
                        (a for a in accounts.list_claude_accounts()
                         if a.label == state.account_label),
                        None)
                    if acct is not None:
                        return accounts.Candidate(
                            backend="claude",
                            label=state.account_label,
                            model=tier,
                            account_num=acct.number,
                        )
                    break
        # Stale or no longer good — release before re-picking.
        _release_account_lease(state, claude_config_dir)

    # Step 2: bulk-probe all accounts + codex once per pass.
    # Narrow to the account an external manager (e.g. a `swap-account` script,
    # via ~/.claude/.current-account) marked active — pod must use its choice
    # for new dispatches instead of ordering the whole pool by its own
    # preference. No-op when the marker is absent. Selection only: the raw
    # `list_claude_accounts()` enumeration is left intact for lease
    # release/harvest and `pod accounts list`.
    claude_accts = (
        accounts.select_for_dispatch(accounts.list_claude_accounts())
        if "claude" in backends_allowed else [])
    available_by_label: dict[str, str | None] = {}
    if "claude" in backends_allowed and not force:
        for a in claude_accts:
            available_by_label[a.label] = accounts.probe_account(
                claude_quota_cmd, a.label)
    elif "claude" in backends_allowed and force:
        available_by_label = {a.label: accepted_models[0]
                              for a in claude_accts}

    codex_avail = (
        "codex" in backends_allowed
        and (force or accounts.probe_codex(codex_quota_cmd)))

    candidates = accounts.enumerate_candidates(
        claude_accounts=claude_accts,
        available_by_label=available_by_label,
        accepted_models=accepted_models,
        codex_available=codex_avail,
        codex_model=codex_model,
        prefer=prefer,
        pin_label=pin_label,
    )
    if not candidates:
        # No usable candidate. If we got here we already released any
        # stale lease in step 1, so just signal the caller to wait.
        return None

    # Step 3: lease + revalidate under host-wide meta-lock.
    chosen: accounts.Candidate | None = None
    with accounts.lease_critical_section():
        # Best-effort orphan eviction so a dead agent's lease doesn't
        # block live ones forever. Cheap: walks .pod/agents/ and the
        # lease dir.
        try:
            live = {a.short_id for a in read_all_agents()}
            live.add(short_id)
            accounts.evict_orphan_leases(live)
        except Exception:
            pass
        held_by_others = {
            l.label for l in accounts.list_leases()
            if l.short_id != short_id
        }
        # When an external account manager owns selection
        # (~/.claude/.current-account present), pod defers fully: no
        # exclusive lease and no keychain. Every agent shares the live
        # canonical ~/.claude/.credentials.json via a symlink, so any number
        # run concurrently and a swap of the canonical is picked up by the
        # next launch. Otherwise keep pod's own per-agent account leasing.
        # Only enter shared mode when the marker resolves to exactly the
        # account select_for_dispatch pinned; an unresolvable marker falls
        # back to the full pool, where leasing must still apply.
        _cur = accounts.current_account()
        shared = (_cur is not None and len(claude_accts) == 1
                  and claude_accts[0].number == _cur)
        for cand in candidates:
            if cand.backend == "codex":
                chosen = cand
                break
            if not shared:
                if cand.label in held_by_others:
                    continue
                if not accounts.try_acquire_lease(
                        cand.label, short_id,
                        project_dir=str(PROJECT_DIR)):
                    continue
            if not force:
                fresh = accounts.probe_account(
                    claude_quota_cmd, cand.label, force=True)
                if (not fresh
                        or accounts.model_tier("claude", fresh)
                        < accounts.model_tier("claude", cand.model)):
                    if not shared:
                        accounts.release_lease(cand.label, short_id)
                    continue
            # Place the credential the agent will use. Shared mode symlinks
            # the live canonical (no stale copy, no keychain); otherwise
            # validate + mirror the leased account as one credential-locked
            # op. Quota can read fine on a logged-out account — its OAuth
            # token expired or was cleared — and every session launched on
            # such an account fails auth at startup (~20s, exit 1); the
            # status check below guards against re-dispatching it forever.
            if shared:
                try:
                    status = accounts.place_shared_credential(
                        claude_config_dir, now=time.time(),
                        skew=_AUTH_EXPIRY_SKEW)
                except OSError as e:
                    log(f"acquire_backend: shared credential link failed: {e}")
                    continue
            else:
                try:
                    status = accounts.preflight_and_mirror(
                        cand.label, cand.account_num, claude_config_dir,
                        now=time.time(), skew=_AUTH_EXPIRY_SKEW)
                except (OSError, accounts.CredentialMirrorError) as e:
                    log(f"acquire_backend: mirror failed for "
                        f"{cand.label}: {e}")
                    accounts.release_lease(cand.label, short_id)
                    continue
            if status != "ok":
                reason = ("no credential (logged out)" if status == "missing"
                          else "token expired")
                log(f"acquire_backend: skipping {cand.label} — {reason}")
                if not shared:
                    accounts.release_lease(cand.label, short_id)
                continue
            chosen = cand
            break

    if chosen is None:
        return None
    if chosen.backend == "claude":
        state.account_label = chosen.label
        # Shared mode took no lease; lease_acquired_at == 0.0 marks that so
        # cleanup skips harvest/release (there is nothing to release).
        state.lease_acquired_at = 0.0 if shared else time.time()
    else:
        # Picked codex; release any outstanding claude lease.
        if state.account_label:
            _release_account_lease(state, claude_config_dir)
    return chosen


def coordination(config: dict, *args, env_extra: dict | None = None,
                 stdin_data: str | None = None) -> subprocess.CompletedProcess:
    """Run a coordination subcommand."""
    script = str(_data_dir() / "coordination")
    env = dict(os.environ)
    # Pass protected-files list so coordination can enforce it.
    pf = cfg_get(config, "project", "protected_files", default=["PLAN.md"])
    if isinstance(pf, list):
        pf = list(pf)
    else:
        pf = pf.split(":")
    # Auto-protect files installed by pod
    for rel in _pod_installed_files():
        if rel not in pf:
            pf.append(rel)
    env["POD_PROTECTED_FILES"] = ":".join(pf)
    # Pass repair-agent thresholds so the coordination script doesn't have to
    # re-parse config.toml. Treated as advisory defaults — script may override
    # for local testing by exporting these before calling.
    stuck_ci_minutes = cfg_get(config, "repair", "stuck_ci_minutes", default=120)
    env.setdefault("POD_STUCK_CI_MINUTES", str(stuck_ci_minutes))
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [script, *args],
        capture_output=True, text=True, timeout=60,
        cwd=str(PROJECT_DIR), env=env,
        input=stdin_data,
    )


def _gh_rate_limit_wait() -> int:
    """Seconds until GH GraphQL rate limit resets, or 0 if plenty remaining.

    Uses the REST API (separate bucket) to check the GraphQL limit.
    Returns 0 on error or if remaining >= 100.
    """
    try:
        resp = gh.get_client().get("/rate_limit")
        if not resp.ok():
            return 0
        body = resp.body() or {}
        graphql = (body.get("resources", {}) or {}).get("graphql", {}) or {}
        remaining = int(graphql.get("remaining", 0))
        if remaining >= 100:
            return 0
        reset = int(graphql.get("reset", 0))
        return max(0, reset - int(time.time()))
    except Exception:
        return 0


def _clear_gh_cache():
    """Clear the gh CLI HTTP cache to avoid stale rate-limit headers.

    After rate limit exhaustion, gh caches introspection query responses
    (with X-Gh-Cache-Ttl: 24h) that carry X-Ratelimit-Remaining: 0.
    Even after the rate limit resets, gh reads the cached headers and
    refuses to make requests. Clearing the cache forces fresh requests.
    """
    import pathlib
    cache_dir = pathlib.Path.home() / ".cache" / "gh"
    if not cache_dir.is_dir():
        return
    for f in cache_dir.rglob("*"):
        if f.is_file() and not f.name.endswith(".zip"):
            try:
                f.unlink()
            except OSError:
                pass


_queue_depth_cache: tuple[float, int] = (0.0, 0)  # (timestamp, depth)
_QUEUE_DEPTH_TTL = 30  # seconds — avoid redundant calls within same dispatch cycle


def get_queue_depth(config: dict) -> int:
    """Get number of unclaimed issues (cached for 30s to reduce API calls)."""
    global _queue_depth_cache
    now = time.time()
    if now - _queue_depth_cache[0] < _QUEUE_DEPTH_TTL:
        return _queue_depth_cache[1]
    try:
        r = coordination(config, "queue-depth")
        depth = int(r.stdout.strip())
        _queue_depth_cache = (now, depth)
        return depth
    except (ValueError, subprocess.TimeoutExpired):
        # On failure, return last known good value (don't cache failure as 0)
        if _queue_depth_cache[0] > 0:
            return _queue_depth_cache[1]
        return 0


def get_return_to_human(config: dict) -> bool:
    """Check whether a planner has signalled return-to-human on the sentinel issue."""
    try:
        r = coordination(config, "check-return-to-human")
        return r.stdout.strip() == "true"
    except subprocess.TimeoutExpired:
        return False


def clear_return_to_human(config: dict) -> None:
    """Remove the return-to-human signal from the sentinel issue."""
    try:
        coordination(config, "clear-return-to-human")
    except subprocess.TimeoutExpired:
        pass


@dataclasses.dataclass
class GHItem:
    """An issue or PR for TUI display."""
    kind: str           # "issue" or "pr"
    number: int
    title: str
    labels: list[str]
    ci_status: str      # "" (unknown/none), "pass", "fail"
    state: str          # "open", "closed", "merged"
    timestamp: str      # ISO 8601 timestamp for the current state


_TUI_REFRESH_QUERY = """
query TuiRefresh($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    openAgentPlan: issues(first: 100, states: OPEN,
                           labels: ["agent-plan"],
                           orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number title createdAt updatedAt
        labels(first: 20) { nodes { name } }
      }
    }
    openDirective: issues(first: 100, states: OPEN,
                           labels: ["directive"],
                           orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number title createdAt updatedAt
        labels(first: 20) { nodes { name } }
      }
    }
    closedAgentPlan: issues(first: 30, states: CLOSED,
                              labels: ["agent-plan"],
                              orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title createdAt updatedAt closedAt
        labels(first: 20) { nodes { name } }
      }
    }
    closedDirective: issues(first: 30, states: CLOSED,
                              labels: ["directive"],
                              orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title createdAt updatedAt closedAt
        labels(first: 20) { nodes { name } }
      }
    }
    blocked: issues(first: 100, states: OPEN, labels: ["blocked"],
                     orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number body }
    }
    hasPrIssues: issues(first: 100, states: OPEN, labels: ["has-pr"],
                         orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        closedByPullRequestsReferences(first: 20) {
          nodes { number state }
        }
      }
    }
    pullRequests(first: 15,
                  states: [OPEN, CLOSED, MERGED],
                  orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title state createdAt updatedAt closedAt mergedAt
        labels(first: 20) { nodes { name } }
        commits(last: 1) {
          nodes {
            commit {
              statusCheckRollup { state }
            }
          }
        }
      }
    }
  }
}
"""


_TUI_REFRESH_CACHE: tuple[float, dict | None] = (0.0, None)
_TUI_REFRESH_TTL = 4.0  # seconds; matches the TUI's fast-path tick cadence
_TUI_REFRESH_LOCK = threading.Lock()
# Bump when `_TUI_REFRESH_QUERY` changes shape so an on-disk snapshot from
# an older pod version is rejected instead of feeding the new code stale
# data with missing keys (`openAgentPlan`, `blocked`, etc.).
# Schema 2 (2026-05-12): removed `openIssueNumbers` (dep resolution moved
# to `fetch_issue_states`); added `orderBy: CREATED_AT DESC` to
# `openAgentPlan`/`openHumanOversight` and `orderBy: UPDATED_AT DESC` to
# `blocked`/`hasPrIssues`; bumped `openAgentPlan` cap to 200.
# Schema 3 (2026-05-15): renamed GraphQL aliases `openHumanOversight` /
# `closedHumanOversight` to `openDirective` / `closedDirective` to match
# the renamed `directive` label (was `human-oversight`).
_TUI_CACHE_SCHEMA = 3
# One-shot flag: try the disk fallback exactly once per process, on the
# first refresh tick after start-up. After that we either have live data
# or we're already serving the in-memory copy.
_TUI_DISK_CACHE_LOADED = False
# Shown in the header once the cache crosses these ages. Tuned so a
# single missed refresh tick doesn't flash the indicator on and off.
_TUI_STALE_AFTER_S = 60.0
_TUI_VERY_STALE_AFTER_S = 3600.0


def _tui_cache_path() -> Path:
    return POD_DIR / "tui-cache.json"


def _save_tui_cache(fetched_at: float, slug: str, repo_node: dict) -> None:
    """Atomic write of the latest successful GraphQL response so a
    later dry-quota startup can serve a stale snapshot rather than an
    empty items panel. chmod 600 — issue bodies and comments are in the
    payload."""
    payload = {
        "schema": _TUI_CACHE_SCHEMA,
        "repo": slug,
        "fetched_at": fetched_at,
        "body": repo_node,
    }
    p = _tui_cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload))
        os.chmod(tmp, 0o600)
        tmp.replace(p)
    except OSError:
        pass


def _load_tui_cache(slug: str) -> tuple[float, dict] | None:
    """Returns `(fetched_at, repo_node)` for a previously-saved snapshot
    of the same repo and schema, or `None` if missing, corrupt, or
    schema-mismatched. Never raises."""
    p = _tui_cache_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != _TUI_CACHE_SCHEMA:
        return None
    if data.get("repo") != slug:
        return None
    try:
        fetched_at = float(data.get("fetched_at", 0.0))
    except (TypeError, ValueError):
        return None
    body = data.get("body")
    if not isinstance(body, dict):
        return None
    # Reject snapshots whose top-level keys don't match the current query
    # shape (defence in depth on top of the schema-version check).
    required = ("openAgentPlan", "openDirective",
                "closedAgentPlan", "closedDirective",
                "blocked", "hasPrIssues", "pullRequests")
    if not all(k in body for k in required):
        return None
    return (fetched_at, body)


def _tui_refresh_batch() -> dict | None:
    """One GraphQL query feeding all three TUI fetch helpers.

    Burner-rewrite (B2): replaces 4–6 `gh issue list` / `gh pr list`
    calls per TUI refresh tick with a single GraphQL request. The
    `_TUI_REFRESH_LOCK` makes the check-then-fetch atomic, so three
    concurrent background threads share one round-trip (rather than
    each firing its own). The layer also ETag-caches the GraphQL
    response (since the cache key includes a hash of the request body),
    so cache-miss hits return 304s.

    Stale-render: if the live call fails (rate limit, 5xx, network),
    we return whatever's currently in the in-memory cache rather than
    `None`. That cache is seeded once per process from
    `.pod/tui-cache.json` on the first cold-start tick, so even a fresh
    pod startup with the GraphQL bucket dry shows the last successful
    snapshot (with an `items stale: …` indicator in the header) instead
    of an empty panel.
    """
    global _TUI_REFRESH_CACHE, _TUI_DISK_CACHE_LOADED
    with _TUI_REFRESH_LOCK:
        now = time.time()
        ts, cached = _TUI_REFRESH_CACHE
        if cached is not None and (now - ts) < _TUI_REFRESH_TTL:
            return cached
        slug = _get_repo()

        # One-shot disk load: only fires on the very first refresh
        # before any live data lands. Carries `fetched_at` from disk
        # forward so the staleness indicator reflects the original
        # snapshot age, not now.
        if cached is None and not _TUI_DISK_CACHE_LOADED:
            _TUI_DISK_CACHE_LOADED = True
            disk = _load_tui_cache(slug)
            if disk is not None:
                _TUI_REFRESH_CACHE = disk
                cached = disk[1]

        owner, _, name = slug.partition("/")
        if not (owner and name):
            return cached
        try:
            resp = gh.get_client().graphql(
                _TUI_REFRESH_QUERY,
                variables={"owner": owner, "name": name},
                cache="etag",
            )
        except Exception:
            return cached  # serve stale
        if not resp.ok():
            return cached
        body = resp.body() or {}
        repo_node = (body.get("data") or {}).get("repository") or {}
        if not repo_node:
            return cached
        _TUI_REFRESH_CACHE = (now, repo_node)
        _save_tui_cache(now, slug, repo_node)
        return repo_node


def _gql_rollup_to_ci(commits_node: dict) -> str:
    """Map a single PR's last-commit `statusCheckRollup` to "pass" / "fail" / ""."""
    nodes = (commits_node or {}).get("nodes") or []
    if not nodes:
        return ""
    rollup = (((nodes[0] or {}).get("commit") or {})
              .get("statusCheckRollup") or {})
    state = rollup.get("state")
    if state == "FAILURE":
        return "fail"
    if state in ("SUCCESS",):
        return "pass"
    return ""


def fetch_issues_and_prs() -> list[GHItem]:
    """Fetch issues (agent-plan or directive label, all states) and
    recent PRs from GitHub. Powered by the batched TUI GraphQL query
    so it costs one round-trip (or a 304) per refresh tick."""
    repo_node = _tui_refresh_batch()
    if repo_node is None:
        return []
    items: list[GHItem] = []
    seen_open: set[int] = set()
    for key in ("openAgentPlan", "openDirective"):
        nodes = ((repo_node.get(key) or {}).get("nodes")) or []
        for iss in nodes:
            num = iss.get("number")
            if not isinstance(num, int) or num in seen_open:
                continue
            seen_open.add(num)
            labels = [l.get("name") for l in
                      ((iss.get("labels") or {}).get("nodes") or [])]
            ts = iss.get("updatedAt") or iss.get("createdAt") or ""
            items.append(GHItem(
                kind="issue", number=num, title=iss.get("title", "") or "",
                labels=labels, ci_status="", state="open", timestamp=ts,
            ))
    seen_closed: set[int] = set()
    for key in ("closedAgentPlan", "closedDirective"):
        nodes = ((repo_node.get(key) or {}).get("nodes")) or []
        for iss in nodes:
            num = iss.get("number")
            if not isinstance(num, int):
                continue
            if num in seen_closed or num in seen_open:
                continue
            seen_closed.add(num)
            labels = [l.get("name") for l in
                      ((iss.get("labels") or {}).get("nodes") or [])]
            ts = (iss.get("closedAt") or iss.get("updatedAt")
                  or iss.get("createdAt") or "")
            items.append(GHItem(
                kind="issue", number=num, title=iss.get("title", "") or "",
                labels=labels, ci_status="", state="closed", timestamp=ts,
            ))
    pr_nodes = ((repo_node.get("pullRequests") or {}).get("nodes")) or []
    for pr in pr_nodes:
        num = pr.get("number")
        if not isinstance(num, int):
            continue
        labels = [l.get("name") for l in
                  ((pr.get("labels") or {}).get("nodes") or [])]
        ci = _gql_rollup_to_ci(pr.get("commits") or {})
        pr_state = (pr.get("state") or "OPEN").lower()
        if pr_state in ("merged", "closed"):
            ts = pr.get("mergedAt") or pr.get("closedAt") or ""
        else:
            ts = pr.get("updatedAt") or pr.get("createdAt") or ""
        items.append(GHItem(
            kind="pr", number=num, title=pr.get("title", "") or "",
            labels=labels, ci_status=ci, state=pr_state, timestamp=ts,
        ))
    items.sort(key=lambda x: (-x.number, x.kind))
    return items


def fetch_blocked_deps() -> dict[int, list[int]]:
    """For every blocked issue, return its currently-OPEN
    `depends-on: #N` deps. Closed (or otherwise non-OPEN) deps are
    filtered out; entries with no remaining open deps are dropped.

    Dep states are resolved via `fetch_issue_states`, which does an
    aliased-batch GraphQL lookup and TTL-caches the result on disk.
    This avoids the older `openIssueNumbers` enumeration which capped
    at 100 oldest open issues and silently dropped annotations on
    repos with more than 100 open issues.

    Fail-closed: deps whose state lookup fails are treated as not-open
    (the annotation is hidden, never wrongly shown).
    """
    import re as _re
    repo_node = _tui_refresh_batch()
    if repo_node is None:
        return {}
    blocked = ((repo_node.get("blocked") or {}).get("nodes")) or []
    raw: dict[int, list[int]] = {}
    all_deps: set[int] = set()
    for iss in blocked:
        num = iss.get("number")
        if not isinstance(num, int):
            continue
        body = iss.get("body") or ""
        deps = [int(d) for d in _re.findall(r"depends-on: #(\d+)", body)]
        if deps:
            raw[num] = deps
            all_deps.update(deps)
    if not all_deps:
        return {}
    states = fetch_issue_states(_get_repo(), sorted(all_deps))
    return {num: filtered for num, deps in raw.items()
            if (filtered := [d for d in deps if states.get(d) == "OPEN"])}


def fetch_has_pr_links() -> dict[int, list[int]]:
    """For every open `has-pr` issue, return the list of currently-OPEN
    closing PRs. Empty list means the label is orphan (all referenced
    PRs merged or closed) — caller still gets the entry. Reads from
    the batched TUI GraphQL response, so the previous N+1 fan-out via
    `gh pr view` becomes 0 extra calls."""
    repo_node = _tui_refresh_batch()
    if repo_node is None:
        return {}
    has_pr = ((repo_node.get("hasPrIssues") or {}).get("nodes")) or []
    out: dict[int, list[int]] = {}
    for iss in has_pr:
        num = iss.get("number")
        if not isinstance(num, int):
            continue
        prs = ((iss.get("closedByPullRequestsReferences") or {})
               .get("nodes") or [])
        open_prs = [p.get("number") for p in prs
                    if isinstance(p, dict) and p.get("state") == "OPEN"
                    and isinstance(p.get("number"), int)]
        out[num] = open_prs
    return out


# ---------------------------------------------------------------------------
# JSONL Monitor (runs as a thread inside agent process)
# ---------------------------------------------------------------------------

def jsonl_monitor(jsonl_path: str, state: AgentState, stop: threading.Event,
                  backend: str = "claude"):
    """Poll JSONL file and update agent state. Runs in a daemon thread."""
    pos = 0
    while True:
        try:
            if not os.path.exists(jsonl_path):
                if stop.is_set():
                    break
                stop.wait(1)
                continue
            with open(jsonl_path, "rb") as f:
                f.seek(pos)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        break  # Partial line — retry next poll
                    pos += len(line)
                    _parse_jsonl_line(line, state, backend)
        except OSError:
            pass
        if stop.is_set():
            break  # Final read done — exit
        stop.wait(1)


def _parse_jsonl_line(line: bytes, state: AgentState, backend: str = "claude"):
    """Dispatch JSONL parsing to the appropriate backend parser."""
    if backend == "codex":
        _parse_codex_jsonl_line(line, state)
    else:
        _parse_claude_jsonl_line(line, state)


def _parse_codex_jsonl_line(line: bytes, state: AgentState):
    """Parse one JSONL line from Codex --json stdout and update state.

    Codex stdout event types (verified from codex exec --json v0.104.0):
      thread.started   → capture backend_session_id for resume
      turn.completed   → token usage
      item.completed   → agent_message (text), command_execution (tool output)
    """
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return

    t = d.get("type")

    if t == "thread.started":
        # Capture Codex thread id for resume; do NOT overwrite state.uuid
        if not state.backend_session_id:
            state.backend_session_id = d.get("thread_id", "")
        state.last_activity = time.time()

    elif t == "turn.completed":
        usage = d.get("usage", {})
        # Codex reports `input_tokens` inclusive of the cached portion
        # (verified: input_tokens + output_tokens == total_tokens in
        # Codex rollouts). Normalize to the same convention as Claude:
        # `tokens_in` = non-cached input, `cache_read` = cached input.
        in_total = usage.get("input_tokens", 0)
        cached = usage.get("cached_input_tokens", 0)
        state.tokens_in += max(0, in_total - cached)
        state.cache_read += cached
        state.tokens_out += usage.get("output_tokens", 0)
        # Codex has no cache_creation_input_tokens equivalent
        state.last_activity = time.time()

    elif t == "item.completed":
        item = d.get("item", {})
        itype = item.get("type")
        if itype == "agent_message":
            text = item.get("text", "").strip()
            if text:
                state.last_text = text[:200]
                state.last_activity = time.time()
        elif itype == "command_execution":
            cmd = item.get("command", "")[:120]
            state.last_text = f"[exec] {cmd}"
            state.last_activity = time.time()
            # Detect issue claim from coordination script output.
            # Anchor the regex to start-of-line so historical mentions in
            # `git log`, `coordination orient`, `gh issue view`, etc. do not
            # match. Skip for `repair`/`replan` workers, which do not claim
            # issues — they claim PRs (repair) or edit issues directly
            # (replan); a stray match would leave stale state for cleanup.
            output = item.get("aggregated_output", "")
            if state.worker_type not in ("repair", "replan"):
                m = re.search(r"^Claimed issue #(\d+)\s*$",
                              output, re.MULTILINE)
                if m:
                    state.claimed_issue = int(m.group(1))
            # Detect PR repair claim from output (only on success — the command
            # emits "Claimed PR #N for repair" only after race-detect passes)
            m_pr_claim = re.search(r"^Claimed PR #(\d+) for repair\s*$",
                                   output, re.MULTILINE)
            if m_pr_claim:
                state.repair_pr = int(m_pr_claim.group(1))
            # Detect PR creation from coordination create-pr
            m2 = re.search(r"(?:coordination\s+create-pr\s+(?:--\S+\s+)*)(\d+)", cmd)
            if m2:
                state.pr_number = int(m2.group(1))
            # Detect repair-claim release (mark-pr-salvaged / close-pr-unsalvageable)
            m_rel = re.search(
                r"coordination\s+(?:mark-pr-salvaged|close-pr-unsalvageable)\s+(\d+)",
                cmd,
            )
            if m_rel:
                state.repair_pr = 0


def _parse_claude_jsonl_line(line: bytes, state: AgentState):
    """Parse one JSONL line and update state."""
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return

    msg_type = d.get("type")

    # Process tool results (type: "user") to detect successful claims
    if msg_type == "user":
        content = d.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_text = block.get("content", "")
                    if isinstance(result_text, str):
                        # Same anchoring + worker_type guard as the codex
                        # parser. See the codex path for the rationale.
                        if state.worker_type not in ("repair", "replan"):
                            m = re.search(r"^Claimed issue #(\d+)\s*$",
                                          result_text, re.MULTILINE)
                            if m:
                                state.claimed_issue = int(m.group(1))
                        m_pr = re.search(r"^Claimed PR #(\d+) for repair\s*$",
                                         result_text, re.MULTILINE)
                        if m_pr:
                            state.repair_pr = int(m_pr.group(1))
        return

    if msg_type != "assistant":
        return

    usage = d.get("message", {}).get("usage", {})
    state.tokens_in += usage.get("input_tokens", 0)
    state.cache_create += usage.get("cache_creation_input_tokens", 0)
    state.cache_read += usage.get("cache_read_input_tokens", 0)
    state.tokens_out += usage.get("output_tokens", 0)
    state.last_activity = time.time()

    for block in d.get("message", {}).get("content", []):
        btype = block.get("type")
        if btype == "text" and block.get("text", "").strip():
            state.last_text = block["text"].strip()
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = block.get("input", {})
            detail = _tool_detail(name, inp, state)
            state.last_text = f"[{name}] {detail}" if detail else f"[{name}]"


def _tool_detail(name: str, inp: dict, state: AgentState) -> str:
    """Extract a display-friendly detail string from a tool invocation."""
    if name == "Bash":
        desc = inp.get("description", "")
        cmd = inp.get("command", "")
        # Detect coordination create-pr (issue-claim is detected from tool results instead)
        m = re.search(r"(?:^|&&\s*|;\s*)(?:\./)?coordination\s+create-pr\s+(?:--\S+\s+)*(\d+)", cmd)
        if m:
            state.pr_number = int(m.group(1))
        # Repair-claim release: mark-pr-salvaged / close-pr-unsalvageable
        # clear the repair-claimed label, so we should clear our own tracking
        # as soon as we see the command fire.
        m_rel = re.search(
            r"(?:^|&&\s*|;\s*)(?:\./)?coordination\s+(?:mark-pr-salvaged|close-pr-unsalvageable)\s+(\d+)",
            cmd,
        )
        if m_rel:
            state.repair_pr = 0
        if desc and cmd:
            return f"{desc}: {cmd}"
        return desc or cmd
    elif name == "Edit":
        p = inp.get("file_path", "")
        return p.split("/")[-1] if p else ""
    elif name in ("Read", "Write"):
        p = inp.get("file_path", "")
        return p.split("/")[-1] if p else ""
    elif name in ("Grep", "Glob"):
        return inp.get("pattern", "")
    elif name == "TodoWrite":
        todos = inp.get("todos", [])
        active = [t for t in todos if t.get("status") == "in_progress"]
        return active[0].get("activeForm", "") if active else ""
    elif name == "Task":
        return inp.get("description", "")
    return name


# ---------------------------------------------------------------------------
# Dispatch Strategies
# ---------------------------------------------------------------------------

def _count_running_repair_agents(agents: list | None = None) -> int:
    """Count live agents whose worker_type is 'repair'."""
    if agents is None:
        agents = read_all_agents()
    return sum(1 for a in agents
               if a.worker_type == "repair"
               and a.status not in ("dead", "stopped", "killed"))


def _list_pr_repair_count(config: dict) -> int:
    """Return the number of PR-repair candidates, or 0 on error.

    Calls `coordination list-pr-repair` and counts output lines.
    """
    try:
        r = coordination(config, "list-pr-repair")
    except (subprocess.TimeoutExpired, OSError):
        return 0
    if r.returncode != 0:
        return 0
    return sum(1 for line in r.stdout.splitlines() if line.strip())


def _count_running_replan_agents(agents: list | None = None) -> int:
    """Count live agents whose worker_type is 'replan'."""
    if agents is None:
        agents = read_all_agents()
    return sum(1 for a in agents
               if a.worker_type == "replan"
               and a.status not in ("dead", "stopped", "killed"))


def _list_replan_count(config: dict) -> int:
    """Return the number of replan candidates, or 0 on error.

    Calls `coordination list-replan` and counts output lines.
    """
    try:
        r = coordination(config, "list-replan")
    except (subprocess.TimeoutExpired, OSError):
        return 0
    if r.returncode != 0:
        return 0
    return sum(1 for line in r.stdout.splitlines() if line.strip())


_gh_quota_cache: dict = {"graphql_remaining": None, "checked_at": 0.0}
_GH_QUOTA_CHECK_TTL = 30.0  # seconds between `gh api rate_limit` polls
_GH_QUOTA_PAUSE_THRESHOLD = 100  # pause dispatch below this (of 5000/hr)


class _GraphQLRateLimited(Exception):
    """Raised when a `gh` subprocess returns a GraphQL rate-limit error.

    Catch at sweep boundaries (`check_dead_claimed_issues`,
    `reconcile_untracked_github_claims`, `check_dead_pr_claimed_prs`) to
    abort the whole pass instead of attempting every remaining item — each
    failed attempt would burn one or more quota slots that already aren't
    there, and silent re-queueing of failures used to turn one rate-limit
    blip into a self-perpetuating retry loop."""
    pass


def _is_rate_limit_error(result) -> bool:
    """True if a `subprocess.run` result carries gh's rate-limit message.

    Matches both `text=True` and binary captures, both stderr and stdout
    (some gh failure paths route the error to stdout). Substring match on
    "rate limit" is intentionally broad — gh's wording has shifted over
    versions ("API rate limit exceeded", "GraphQL: API rate limit already
    exceeded", "secondary rate limit", ...)."""
    for stream_name in ("stderr", "stdout"):
        s = getattr(result, stream_name, "") or ""
        if isinstance(s, bytes):
            s = s.decode(errors="replace")
        if "rate limit" in s.lower():
            return True
    return False


def _gh_quota_ok() -> bool:
    """Return False when GitHub GraphQL quota is near exhaustion.

    Polls `gh api rate_limit` at most every `_GH_QUOTA_CHECK_TTL` seconds
    and caches the result. The `rate_limit` endpoint itself does not
    consume quota, so the check is effectively free.

    Returns True (allow dispatch) if:
      - quota has never been checked (first tick — assume OK)
      - quota remaining >= threshold
      - a poll failed (timeout / parse error — assume OK, previous reading
        if cached)
    """
    import time
    now = time.time()
    if now - _gh_quota_cache["checked_at"] < _GH_QUOTA_CHECK_TTL:
        rem = _gh_quota_cache["graphql_remaining"]
        return rem is None or rem >= _GH_QUOTA_PAUSE_THRESHOLD
    try:
        resp = gh.get_client().get("/rate_limit", timeout=10)
        if resp.ok():
            graphql = ((resp.body() or {}).get("resources", {}) or {}
                       ).get("graphql", {}) or {}
            remaining = graphql.get("remaining")
            if remaining is not None:
                _gh_quota_cache["graphql_remaining"] = int(remaining)
                _gh_quota_cache["checked_at"] = now
    except Exception:
        pass  # keep previous reading (if any); treat unknown as OK
    rem = _gh_quota_cache["graphql_remaining"]
    return rem is None or rem >= _GH_QUOTA_PAUSE_THRESHOLD


def _choose_draining(config: dict, draining: dict) -> str | None:
    """Pick a random draining worker type that has available work.

    For types with issue_label, check per-label queue depth via
    `coordination queue-depth <label>`. Shuffled so no type is
    systematically starved.
    For types without issue_label, always consider them available — except
    `repair`, which is excluded here; repair dispatch is handled by the
    short-circuit at the top of dispatch_queue_balance so that a generic
    draining pass does not pick `repair` when no PRs need it.
    Returns None if no draining type has work.
    """
    items = list(draining.items())
    random.shuffle(items)
    for name, wt in items:
        if name == "repair":
            log("_choose_draining: skipping repair (handled by top-level short-circuit)")
            continue
        issue_label = wt.get("issue_label", "")
        if issue_label:
            try:
                r = coordination(config, "queue-depth", issue_label)
                depth = int(r.stdout.strip())
            except (ValueError, subprocess.TimeoutExpired):
                depth = 0
            if depth > 0:
                log(f"_choose_draining: {name} (label={issue_label}) has depth={depth}")
                return name
            else:
                log(f"_choose_draining: {name} (label={issue_label}) has depth=0, skipping")
        else:
            # No label filter — treat as always having work
            log(f"_choose_draining: {name} (no label filter) → selected")
            return name
    log(f"_choose_draining: no type had work out of {[n for n,_ in items]}")
    return None


def _get_critical_path_depth(config: dict, label: str = "") -> int:
    """Check for unclaimed critical-path issues, optionally filtered by label."""
    try:
        args = ["critical-path-depth"]
        if label:
            args.append(label)
        r = coordination(config, *args)
        return int(r.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        return 0


def _choose_critical_path_draining(config: dict, draining: dict) -> str | None:
    """Pick a draining worker type that has unclaimed critical-path work.

    Like _choose_draining but only considers critical-path issues.
    For typed workers (with issue_label), checks per-label critical-path depth.
    For untyped workers, checks global critical-path depth.
    `repair` is excluded — it handles PRs, not critical-path issues.
    """
    items = list(draining.items())
    random.shuffle(items)
    for name, wt in items:
        if name == "repair":
            continue
        issue_label = wt.get("issue_label", "")
        if issue_label:
            if _get_critical_path_depth(config, issue_label) > 0:
                return name
        else:
            # No label filter — handles all issue types
            if _get_critical_path_depth(config) > 0:
                return name
    return None


def dispatch_queue_balance(config: dict, queue_depth: int,
                           worker_types: dict,
                           state: AgentState | None = None) -> str | None:
    """Low queue → locked types (queue-filling), high queue → unlocked types (queue-draining).

    **Repair preference**: before filling/draining, if the `repair` worker
    type is configured and there are unclaimed PR-repair candidates,
    dispatch `repair`. Repair is a regular worker type bounded by the
    same `target` as every other type — it does not live outside the
    target.
    """
    # Honour an effective target of 0.  `get_effective_target()` returns 0
    # whenever the user or the planner has set their respective target to 0
    # (e.g. via `coordination return-to-human`).  Without this check the
    # per-session dispatch loop would keep handing prompts to an existing
    # agent forever, because target=0 is only consulted by the auto-spawn
    # and restart paths.
    effective_target = get_effective_target()
    if effective_target == 0:
        log("dispatch: effective_target=0, no dispatch (return-to-human or user target=0)")
        return None

    # GitHub GraphQL quota guard. When the hourly quota is near exhaustion,
    # pause all dispatch — spawning agents whose first GH call will fail
    # just burns tokens and leaves `claimed` labels stuck. Running agents
    # already back off on their own via their `GH rate limit low, sleeping`
    # path.
    if not _gh_quota_ok():
        log(f"dispatch: GitHub GraphQL quota low "
            f"(remaining={_gh_quota_cache['graphql_remaining']}), "
            f"pausing dispatch until quota recovers")
        return None

    # Repair preference. If there are unclaimed PR-repair candidates,
    # prefer `repair` over other work. Each dispatch tick claims one
    # repair; the next tick will either pick another repair (if more
    # candidates remain and we have the target budget) or fall through
    # to normal work.
    if "repair" in worker_types:
        candidates = _list_pr_repair_count(config)
        running_repair = _count_running_repair_agents()
        if candidates > running_repair:
            log(f"dispatch: repair preferred (candidates={candidates}, "
                f"running_repair={running_repair})")
            return "repair"

    # Replan preference. After repair, before any planner-lock filling
    # or queue-draining work, prefer `replan` if there are unclaimed
    # replan-labelled issues. Unlike `repair` (lock-less), `replan`
    # shares the `planner` lock with `plan`, so we acquire the lock
    # here. If it fails, skip the rest of the filling path (which would
    # try the same lock again) and fall straight to draining — calling
    # `lock-planner` twice per tick is expensive (posts/sleeps/deletes
    # a comment).
    if "replan" in worker_types:
        replan_candidates = _list_replan_count(config)
        running_replan = _count_running_replan_agents()
        if replan_candidates > running_replan:
            replan_lock = worker_types["replan"].get("lock", "planner")
            r = coordination(config, f"lock-{replan_lock}")
            if r.returncode == 0:
                if state:
                    state.lock_held = replan_lock
                    state.write()
                log(f"dispatch: replan preferred "
                    f"(candidates={replan_candidates}, "
                    f"running_replan={running_replan})")
                return "replan"
            log(f"dispatch: replan preferred but {replan_lock} lock held; "
                f"skipping filling, trying draining")
            draining = {k: v for k, v in worker_types.items()
                        if not v.get("lock")}
            if queue_depth > 0 and draining:
                return _choose_draining(config, draining)
            return None

    config_min_queue = cfg_get(config, "dispatch", "min_queue", default=3)
    planner_min_queue = read_planner_min_queue()
    if planner_min_queue is not None:
        min_queue = max(0, min(config_min_queue, planner_min_queue))
    else:
        min_queue = max(0, config_min_queue)

    # Separate types into queue-filling (have locks) and queue-draining (no locks).
    # `repair` is draining in principle but is handled exclusively by the
    # short-circuit above, so _choose_draining skips it.
    filling = {k: v for k, v in worker_types.items() if v.get("lock")}
    draining = {k: v for k, v in worker_types.items() if not v.get("lock")}

    log(f"dispatch: queue_depth={queue_depth} min_queue={min_queue} "
        f"(config={config_min_queue} planner={planner_min_queue}) "
        f"filling={list(filling)} draining={list(draining)}")

    # Critical-path override: if there are unclaimed critical-path issues,
    # always dispatch a matching worker regardless of min_queue threshold.
    # This prevents the startup stall where the pipeline is bottlenecked on
    # a single issue but the queue is too small to trigger worker dispatch.
    if queue_depth > 0 and draining:
        chosen = _choose_critical_path_draining(config, draining)
        if chosen is not None:
            log(f"dispatch: critical-path override → {chosen}")
            return chosen

    if min_queue > 0 and queue_depth < min_queue and filling:
        log(f"dispatch: queue low ({queue_depth} < {min_queue}), trying filling types")
        # Try to acquire lock for a filling type
        for name, wt in filling.items():
            lock_name = wt["lock"]
            r = coordination(config, f"lock-{lock_name}")
            if r.returncode == 0:
                if state:
                    state.lock_held = lock_name
                    state.write()
                log(f"dispatch: acquired {lock_name} lock → {name}")
                return name
        # Lock held — fall back to draining if queue > 0
        if queue_depth > 0 and draining:
            chosen = _choose_draining(config, draining)
            log(f"dispatch: lock held, draining fallback → {chosen}")
            return chosen
        # Queue empty and lock held — wait
        log("dispatch: lock held, queue empty → None")
        return None
    elif draining:
        chosen = _choose_draining(config, draining)
        if chosen is not None:
            log(f"dispatch: draining → {chosen}")
            return chosen
        # No labeled work available despite nonzero global queue (e.g. unlabeled issues
        # from before the typed-worker migration). Fall back to planner to create
        # properly-typed issues rather than stalling indefinitely.
        log("dispatch: _choose_draining returned None despite draining types existing, "
            "falling back to filling")
        if filling:
            for name, wt in filling.items():
                lock_name = wt["lock"]
                r = coordination(config, f"lock-{lock_name}")
                if r.returncode == 0:
                    if state:
                        state.lock_held = lock_name
                        state.write()
                    log(f"dispatch: draining-fallback acquired {lock_name} lock → {name}")
                    return name
        log("dispatch: no work available → None")
        return None
    elif filling:
        log("dispatch: no draining types, trying filling only")
        # Only filling types exist — try them
        for name, wt in filling.items():
            lock_name = wt["lock"]
            r = coordination(config, f"lock-{lock_name}")
            if r.returncode == 0:
                if state:
                    state.lock_held = lock_name
                    state.write()
                log(f"dispatch: filling-only acquired {lock_name} lock → {name}")
                return name
        log("dispatch: filling-only, all locks held → None")
        return None
    log("dispatch: no worker types matched → None")
    return None


_round_robin_idx = 0


def dispatch_round_robin(config: dict, queue_depth: int,
                          worker_types: dict,
                          state: AgentState | None = None) -> str | None:
    """Cycle through worker types, skipping locked ones."""
    global _round_robin_idx
    # Honour effective_target=0 (issue #27) so the per-session dispatch loop
    # doesn't keep handing prompts to a running agent after return-to-human.
    if get_effective_target() == 0:
        log("dispatch: effective_target=0, no dispatch (round_robin)")
        return None
    names = list(worker_types.keys())
    if not names:
        return None
    for _ in range(len(names)):
        name = names[_round_robin_idx % len(names)]
        _round_robin_idx += 1
        wt = worker_types[name]
        lock_name = wt.get("lock")
        if lock_name:
            r = coordination(config, f"lock-{lock_name}")
            if r.returncode != 0:
                continue
            if state:
                state.lock_held = lock_name
                state.write()
        return name
    return None


def dispatch_custom(config: dict, queue_depth: int,
                     worker_types: dict,
                     state: AgentState | None = None) -> str | None:
    """Run a custom dispatch script."""
    # Honour effective_target=0 (issue #27).  We could pass it to the script
    # via env, but the safer default is to never invoke the script at all
    # when shutdown has been requested — there's nothing the script could
    # legitimately return that we'd want to act on.
    if get_effective_target() == 0:
        log("dispatch: effective_target=0, no dispatch (custom)")
        return None
    strategy = cfg_get(config, "dispatch", "strategy", default="")
    script = os.path.expanduser(strategy)
    env = dict(os.environ)
    env["POD_QUEUE_DEPTH"] = str(queue_depth)
    env["POD_AGENT_COUNT"] = str(len(read_all_agents()))
    env["POD_WORKER_TYPES"] = ",".join(worker_types.keys())
    try:
        r = subprocess.run(
            [script], capture_output=True, text=True, timeout=30, env=env,
            cwd=str(PROJECT_DIR),
        )
        if r.returncode != 0:
            return None
        name = r.stdout.strip()
        if name in worker_types:
            # If the chosen type has a lock, try to acquire it
            lock_name = worker_types[name].get("lock")
            if lock_name:
                lr = coordination(config, f"lock-{lock_name}")
                if lr.returncode != 0:
                    return None
                if state:
                    state.lock_held = lock_name
                    state.write()
            return name
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def dispatch(config: dict, state: AgentState | None = None) -> str | None:
    """Choose a worker type to run. Returns type name or None (wait).
    If state is provided, sets state.lock_held immediately upon lock acquisition."""
    strategy = cfg_get(config, "dispatch", "strategy", default="queue_balance")
    worker_types = cfg_get(config, "worker_types", default={})
    if not worker_types:
        say("No worker types configured")
        return None

    queue_depth = get_queue_depth(config)

    if strategy == "queue_balance":
        return dispatch_queue_balance(config, queue_depth, worker_types, state)
    elif strategy == "round_robin":
        return dispatch_round_robin(config, queue_depth, worker_types, state)
    elif os.path.exists(os.path.expanduser(strategy)):
        return dispatch_custom(config, queue_depth, worker_types, state)
    else:
        say(f"Unknown dispatch strategy: {strategy}, falling back to queue_balance")
        return dispatch_queue_balance(config, queue_depth, worker_types, state)


# ---------------------------------------------------------------------------
# Dead claim recovery
# ---------------------------------------------------------------------------

_cached_base_branch: str = ""
_cached_repo_name: str = ""


def _get_base_branch() -> str:
    """Auto-detect the default branch (e.g. 'main' or 'master'). Cached after first call."""
    global _cached_base_branch
    if _cached_base_branch:
        return _cached_base_branch
    # Direct REST via the access layer (ETag-cached after the first hit),
    # so this doesn't burn GraphQL quota and steady-state cost is a 304.
    try:
        resp = gh.get_client().get(f"/repos/{_get_repo()}", timeout=15)
        if resp.ok():
            db = ((resp.body() or {}).get("default_branch") or "").strip()
            if db:
                _cached_base_branch = db
                return _cached_base_branch
    except Exception:
        pass
    return "master"


def _get_repo() -> str:
    """Auto-detect GitHub repo (owner/name) from the current git remote. Cached after first call.

    Uses `git remote get-url origin` first — purely local, no API call. Falls
    back to `gh api repos/{slug}` only if the remote isn't a GitHub URL we can
    parse, which keeps startup off the GraphQL quota path.
    """
    global _cached_repo_name
    if _cached_repo_name:
        return _cached_repo_name
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10, cwd=str(PROJECT_DIR),
        )
        url = r.stdout.strip()
        # Handle SSH (git@github.com:owner/repo.git) and HTTPS
        import re as _re
        m = _re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
        if m:
            _cached_repo_name = m.group(1)
            return _cached_repo_name
    except Exception:
        pass
    # Fallback: ask gh porcelain. Only reached when the remote isn't a
    # parseable GitHub URL — e.g. a non-default remote name or a deploy-key
    # URL. Routed through the layer so it's still logged.
    try:
        r = _gh_cli("repo", "view", "--json", "nameWithOwner",
                    "-q", ".nameWithOwner", timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            _cached_repo_name = r.stdout.strip()
            return _cached_repo_name
    except Exception:
        pass
    return "unknown/unknown"


# Strictness ranking of GitHub interaction-limit values. Higher = more restrictive.
_INTERACTION_LIMIT_RANK = {
    "existing_users": 1,
    "contributors_only": 2,
    "collaborators_only": 3,
}

# Possible values of GitHub's `authorAssociation` field on issues and comments.
# See https://docs.github.com/en/graphql/reference/enums#commentauthorassociation
_GH_AUTHOR_ASSOCIATIONS = {
    "OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR",
    "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR", "MANNEQUIN", "NONE",
}

# Re-check the live interaction limit at most this often, to bound the cost
# of `gh api` calls when many spawns happen in quick succession but still
# detect limit removal/expiry inside long-running pod processes.
_SECURITY_CHECK_TTL_SECONDS = 300

# Disk-persisted version of the in-memory TTL: survives across `pod`
# invocations so the TUI startup doesn't burn a REST call every time.
# Bounded short enough that a private→public flip (or limit removal)
# is caught within an hour even if `pod` is invoked back-to-back.
_SECURITY_DISK_TTL_SECONDS = 3600

_security_last_ok: float = 0.0


def _security_cache_path() -> Path:
    return PROJECT_DIR / ".pod" / "security-cache.json"


def _load_security_cache(slug: str, minimum: str, min_days: float) -> bool:
    """Return True if a recent successful check for `slug` is on disk and
    still satisfies the current `[security]` policy. Reading the cache must
    not itself fail-closed: any error returns False so we fall through to
    the live check."""
    p = _security_cache_path()
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except Exception:
        return False
    if data.get("slug") != slug:
        return False
    try:
        age = time.time() - float(data.get("checked_at", 0))
    except (TypeError, ValueError):
        return False
    if age < 0 or age > _SECURITY_DISK_TTL_SECONDS:
        return False
    visibility = (data.get("visibility") or "").lower()
    if visibility != "public":
        # Non-public repos skip the interaction-limits check entirely;
        # caching the visibility verdict is sufficient.
        return True
    have = data.get("interaction_limit") or ""
    if _INTERACTION_LIMIT_RANK.get(have, 0) < _INTERACTION_LIMIT_RANK.get(minimum, 0):
        return False
    expires = data.get("expires_at")
    if expires:
        try:
            t = datetime.datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        except ValueError:
            return False
        delta = t - datetime.datetime.now(datetime.timezone.utc)
        if (delta.total_seconds() / 86400) < min_days:
            return False
    return True


def _save_security_cache(slug: str, *, visibility: str,
                         interaction_limit: str | None = None,
                         expires_at: str | None = None) -> None:
    """Persist a successful check verdict so subsequent invocations can
    skip the REST calls. Errors here are non-fatal: the cache is an
    optimisation, not a security boundary."""
    try:
        p = _security_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "slug": slug,
            "checked_at": time.time(),
            "visibility": visibility,
            "interaction_limit": interaction_limit,
            "expires_at": expires_at,
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)
    except Exception:
        pass


def validate_security_config(cfg: dict) -> None:
    """Reject typo'd values in `[security]` so they fail loudly rather than
    silently weakening the policy.
    """
    sec = cfg.get("security", {}) or {}
    minimum = sec.get("minimum_interaction_limit", "collaborators_only")
    if minimum not in _INTERACTION_LIMIT_RANK:
        valid = ", ".join(sorted(_INTERACTION_LIMIT_RANK))
        print(f"pod: invalid [security].minimum_interaction_limit = {minimum!r}; "
              f"must be one of: {valid}", file=sys.stderr)
        sys.exit(1)
    enforce = sec.get("enforce_interaction_limits", True)
    if not isinstance(enforce, bool):
        print(f"pod: invalid [security].enforce_interaction_limits = {enforce!r}; "
              f"must be true or false", file=sys.stderr)
        sys.exit(1)
    days = sec.get("minimum_expiry_days", 7)
    if not isinstance(days, (int, float)) or isinstance(days, bool) or days < 0:
        print(f"pod: invalid [security].minimum_expiry_days = {days!r}; "
              f"must be a non-negative number", file=sys.stderr)
        sys.exit(1)

    trust = sec.get("trust_only_collaborators", True)
    if not isinstance(trust, bool):
        print(f"pod: invalid [security].trust_only_collaborators = {trust!r}; "
              f"must be true or false", file=sys.stderr)
        sys.exit(1)
    assocs = sec.get("trusted_author_associations",
                     ["OWNER", "MEMBER", "COLLABORATOR"])
    if not isinstance(assocs, list) or not all(isinstance(a, str) for a in assocs):
        print(f"pod: invalid [security].trusted_author_associations = {assocs!r}; "
              f"must be a list of strings", file=sys.stderr)
        sys.exit(1)
    bad = [a for a in assocs if a not in _GH_AUTHOR_ASSOCIATIONS]
    if bad:
        valid = ", ".join(sorted(_GH_AUTHOR_ASSOCIATIONS))
        print(f"pod: invalid value(s) in [security].trusted_author_associations: "
              f"{bad!r}; must each be one of: {valid}", file=sys.stderr)
        sys.exit(1)
    users = sec.get("trusted_users", [])
    if not isinstance(users, list) or not all(isinstance(u, str) for u in users):
        print(f"pod: invalid [security].trusted_users = {users!r}; "
              f"must be a list of strings", file=sys.stderr)
        sys.exit(1)
    cache_s = sec.get("provenance_cache_seconds", 60)
    if (not isinstance(cache_s, (int, float)) or isinstance(cache_s, bool)
            or cache_s < 0):
        print(f"pod: invalid [security].provenance_cache_seconds = {cache_s!r}; "
              f"must be a non-negative number", file=sys.stderr)
        sys.exit(1)


def _security_fail(slug: str, msg: str, *, auth_hint: bool = False) -> None:
    print(f"pod: security check failed for {slug}: {msg}", file=sys.stderr)
    print("", file=sys.stderr)
    if auth_hint:
        print("If `gh` is unauthenticated, run `gh auth status` / `gh auth login`.",
              file=sys.stderr)
        print("", file=sys.stderr)
    print("Refusing to dispatch agents. Either:", file=sys.stderr)
    print("  • Set/renew the interaction limit:", file=sys.stderr)
    print(f"      gh api -X PUT repos/{slug}/interaction-limits \\", file=sys.stderr)
    print("        -f limit=collaborators_only -f expiry=six_months", file=sys.stderr)
    print("  • Or disable the check (not recommended for public repos):", file=sys.stderr)
    print("      under [security] in .pod/config.toml, set", file=sys.stderr)
    print("      enforce_interaction_limits = false", file=sys.stderr)
    sys.exit(1)


def _fetch_repo_security_meta(slug: str) -> tuple[dict | None, int | None]:
    """Fetch visibility + interaction-limit metadata for `slug`.

    Tries GraphQL first (one round-trip, the B3 burner shape), then falls
    back to REST (two calls: `GET /repos/{slug}` and `GET
    /repos/{slug}/interaction-limits`) when GraphQL is unusable. The two
    REST buckets are independent of the GraphQL bucket on GitHub, so this
    fallback is the difference between "pod refuses to start" and "pod
    starts" when GraphQL is rate-limited.

    Returns `(meta, last_status)`:
      meta — dict `{"visibility": str, "ability": dict | None}` on success,
             or `None` if both transports failed.
             `ability` is `None` when no interaction limit is set, else
             `{"limit": str (lowercased), "expiresAt": str | None}`.
      last_status — the HTTP status of the last failing transport, used
             by the caller to print an auth hint on 401/403. `None` when
             the failure was non-HTTP (exception, empty body).

    Raises `FileNotFoundError` if `gh` is missing — caller handles.
    """
    owner, _, name = slug.partition("/")
    client = gh.get_client()
    last_status: int | None = None

    def _normalize_gql(body: dict) -> dict | None:
        repo_node = ((body.get("data") or {}).get("repository") or {})
        visibility = (repo_node.get("visibility") or "").lower()
        if not visibility:
            return None
        ability_raw = repo_node.get("interactionAbility") or None
        if ability_raw:
            have_raw = ability_raw.get("limit") or ""
            ability = {
                "limit": have_raw.lower() if isinstance(have_raw, str) else "",
                "expiresAt": ability_raw.get("expiresAt"),
            }
        else:
            ability = None
        return {"visibility": visibility, "ability": ability}

    # Smart routing: GraphQL is one round-trip vs REST's two, so we prefer
    # it normally. But if its bucket is near-empty while REST has budget,
    # skip straight to REST — a guaranteed-to-fail GraphQL call burns the
    # bucket further and delays the inevitable fallback.
    skip_graphql = False
    try:
        rates = client.rate()
        g = rates.get("graphql")
        c = rates.get("core")
        if (g is not None and g.limit and g.remaining < 100
                and c is not None and c.limit and c.remaining > 200):
            skip_graphql = True
    except Exception:
        pass

    # --- GraphQL attempt ---
    if not skip_graphql:
        try:
            resp = client.graphql(
                "query($owner: String!, $name: String!) {"
                "  repository(owner: $owner, name: $name) {"
                "    visibility"
                "    interactionAbility { limit expiresAt origin }"
                "  }"
                "}",
                variables={"owner": owner, "name": name},
            )
        except FileNotFoundError:
            raise
        except Exception:
            resp = None
        if resp is not None:
            if resp.ok():
                meta = _normalize_gql(resp.body() or {})
                if meta is not None:
                    return (meta, None)
                # Status 200 but empty data (GraphQL `errors` block,
                # typically RATE_LIMITED) — fall through to REST.
            else:
                last_status = resp.status

    # --- REST fallback (independent rate-limit bucket) ---
    try:
        repo_resp = client.get(f"/repos/{slug}")
    except FileNotFoundError:
        raise
    except Exception:
        return (None, last_status)
    if not repo_resp.ok():
        return (None, repo_resp.status)
    repo_body = repo_resp.body() or {}
    visibility = (repo_body.get("visibility") or "").lower()
    if not visibility:
        # Older REST shape lacks `visibility`; derive from the `private`
        # boolean, which has always been present.
        priv = repo_body.get("private")
        if priv is True:
            visibility = "private"
        elif priv is False:
            visibility = "public"
        else:
            return (None, repo_resp.status)
    if visibility != "public":
        return ({"visibility": visibility, "ability": None}, None)

    try:
        lim_resp = client.get(f"/repos/{slug}/interaction-limits")
    except FileNotFoundError:
        raise
    except Exception:
        return (None, None)
    # The endpoint returns 204 No Content, or 200 with an empty / no-limit
    # body, when no limit is set.
    if lim_resp.status == 204:
        return ({"visibility": visibility, "ability": None}, None)
    if not lim_resp.ok():
        return (None, lim_resp.status)
    lim_body = lim_resp.body() or {}
    have_raw = lim_body.get("limit") if isinstance(lim_body, dict) else ""
    if not have_raw:
        return ({"visibility": visibility, "ability": None}, None)
    ability = {
        "limit": have_raw.lower() if isinstance(have_raw, str) else "",
        "expiresAt": lim_body.get("expires_at"),
    }
    return ({"visibility": visibility, "ability": ability}, None)


def check_repo_security(config: dict) -> None:
    """Refuse to dispatch agents if the repo's GitHub interaction limits are
    missing, too lax, or expiring soon.

    Pod feeds issue bodies and comments on labeled issues into agent prompts.
    On a public repo with no interaction limit, anyone with a GitHub account
    can inject text into that prompt stream. GitHub's interaction-limits
    feature gates who can post; this check enforces it before any spawn.

    Note: interaction limits are forward-only — content posted *before* the
    limit was enabled (issue bodies, comments on issues labeled later) is
    not protected. This check guards against new injection, not provenance
    of historical content.

    Re-runs at most every `_SECURITY_CHECK_TTL_SECONDS`. Called from
    `spawn_agent` so every dispatch path is gated, including TUI auto-spawn
    and dead-session restart from `check_dead_claimed_issues`.
    """
    global _security_last_ok
    now = time.time()
    if _security_last_ok and (now - _security_last_ok) < _SECURITY_CHECK_TTL_SECONDS:
        return

    sec = config.get("security", {}) or {}
    if not sec.get("enforce_interaction_limits", True):
        _security_last_ok = now
        return

    minimum = sec.get("minimum_interaction_limit", "collaborators_only")
    min_days = sec.get("minimum_expiry_days", 7)

    slug = _get_repo()

    # Disk cache: skip the API round-trip if a successful check for this
    # repo is on file and still satisfies the current policy. Bounded by
    # _SECURITY_DISK_TTL_SECONDS so transitions are caught.
    if _load_security_cache(slug, minimum, min_days):
        _security_last_ok = now
        return

    try:
        meta, last_status = _fetch_repo_security_meta(slug)
    except FileNotFoundError:
        print("pod: security: `gh` CLI not found; cannot verify interaction limits.",
              file=sys.stderr)
        sys.exit(1)
    if meta is None:
        if last_status is not None:
            print(f"pod: security: cannot determine repo visibility: HTTP {last_status}",
                  file=sys.stderr)
            if last_status in (401, 403):
                print("    → run `gh auth status` / `gh auth login` and retry.",
                      file=sys.stderr)
        else:
            print("pod: security: cannot determine repo visibility: "
                  "GraphQL and REST both unavailable", file=sys.stderr)
            print("    → run `gh api rate_limit` to inspect; "
                  "if both buckets are dry, wait for reset.", file=sys.stderr)
        sys.exit(1)

    visibility = meta["visibility"]
    if visibility != "public":
        _save_security_cache(slug, visibility=visibility)
        _security_last_ok = now
        return

    ability = meta["ability"]
    if not ability:
        _security_fail(slug,
            "no interaction limit set on this public repo. "
            "Random GitHub users can post issues/comments whose contents "
            "feed into pod agent prompts.")

    have = ability.get("limit") or ""
    if _INTERACTION_LIMIT_RANK.get(have, 0) < _INTERACTION_LIMIT_RANK.get(minimum, 0):
        _security_fail(slug,
            f"interaction limit is `{have or 'unknown'}`, "
            f"but pod requires `{minimum}` or stricter.")

    expires = ability.get("expiresAt")
    if expires:
        try:
            t = datetime.datetime.fromisoformat(expires.replace("Z", "+00:00"))
        except ValueError:
            _security_fail(slug, f"unparseable expiresAt: {expires!r}")
        delta = t - datetime.datetime.now(datetime.timezone.utc)
        days = delta.total_seconds() / 86400
        if days < min_days:
            _security_fail(slug,
                f"interaction limit `{have}` expires in {days:.1f} days "
                f"(< {min_days}-day threshold).")

    _save_security_cache(slug, visibility=visibility,
                         interaction_limit=have, expires_at=expires)
    _security_last_ok = now


# ----------------------------------------------------------------------
# Per-message author provenance gate.
#
# A defense-in-depth layer on top of the repo-wide interaction-limit
# check. We refuse to surface issue bodies / comments authored by
# accounts that lack a trusted association with the repo. Trust is
# decided by GitHub's `authorAssociation` field on issues and comments
# (no admin scope needed). See README "Public-repo safety" for the
# threat model and limits — including the `MEMBER` ≠ repo-write caveat
# for org repos.
# ----------------------------------------------------------------------


@dataclass
class _ProvenanceComment:
    comment_id: int
    login: str
    association: str


@dataclass
class _IssueProvenance:
    repo: str
    issue_num: int
    author_login: str
    author_association: str
    comments: list[_ProvenanceComment]
    fetched_at: float


# In-process cache: {(repo, issue_num): _IssueProvenance}. Kept for the
# rare case of repeated checks inside one subprocess (e.g. CLI helpers
# that read several issues). The cross-process tier below is what
# actually keeps the GraphQL bucket from draining.
_provenance_cache: dict[tuple[str, int], _IssueProvenance] = {}


def _provenance_ttl(config: dict) -> float:
    sec = config.get("security", {}) or {}
    return float(sec.get("provenance_cache_seconds", 60))


class _ProvenanceDiskCache:
    """Cross-process TTL'd cache for issue provenance.

    `cmd_filter_trusted_issues` runs as a fresh subprocess per
    `coordination orient` / `list-unclaimed` tick, so an in-process
    cache can't help across invocations. Each agent worktree ticks
    independently, so with N agents and M open agent-plan issues the
    naive cost is N*M GraphQL POSTs per tick. GitHub does not honor
    `If-None-Match` on `/graphql` POSTs (field data: zero 304s out of
    ~14k POSTs), so the layer's ETag store can't help either.

    This is a plain on-disk value cache: one JSON file per
    (repo, issue_num), keyed by sha256 of those fields so we don't
    collide across repos. Atomic writes via tmp+rename. TTL is read
    from `security.provenance_cache_seconds` (default 60s).
    """

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _key(repo: str, issue_num: int) -> str:
        return hashlib.sha256(f"{repo}#{issue_num}".encode()).hexdigest()

    def _path_for(self, repo: str, issue_num: int) -> Path:
        return self.root / f"{self._key(repo, issue_num)}.json"

    def get(self, repo: str, issue_num: int,
            ttl: float) -> _IssueProvenance | None:
        p = self._path_for(repo, issue_num)
        try:
            raw = p.read_text()
        except (OSError, FileNotFoundError):
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        fetched_at = float(data.get("fetched_at", 0.0))
        if (time.time() - fetched_at) >= ttl:
            return None
        try:
            comments = [
                _ProvenanceComment(
                    comment_id=int(c["comment_id"]),
                    login=str(c.get("login", "")),
                    association=str(c.get("association", "NONE")),
                )
                for c in (data.get("comments") or [])
            ]
            return _IssueProvenance(
                repo=str(data["repo"]),
                issue_num=int(data["issue_num"]),
                author_login=str(data.get("author_login", "")),
                author_association=str(
                    data.get("author_association", "NONE")),
                comments=comments,
                fetched_at=fetched_at,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, prov: _IssueProvenance) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        payload = {
            "repo": prov.repo,
            "issue_num": prov.issue_num,
            "author_login": prov.author_login,
            "author_association": prov.author_association,
            "comments": [
                {"comment_id": c.comment_id, "login": c.login,
                 "association": c.association}
                for c in prov.comments
            ],
            "fetched_at": prov.fetched_at,
        }
        p = self._path_for(prov.repo, prov.issue_num)
        tmp = p.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, separators=(",", ":")))
            os.chmod(tmp, 0o600)
            tmp.replace(p)
        except OSError:
            pass

    def invalidate(self, repo: str, issue_num: int) -> None:
        p = self._path_for(repo, issue_num)
        try:
            p.unlink()
        except (OSError, FileNotFoundError):
            pass


_PROVENANCE_DISK_CACHE: _ProvenanceDiskCache | None = None


def _provenance_disk_cache() -> _ProvenanceDiskCache:
    global _PROVENANCE_DISK_CACHE
    if _PROVENANCE_DISK_CACHE is None:
        _PROVENANCE_DISK_CACHE = _ProvenanceDiskCache(
            POD_DIR / "provenance-cache")
    return _PROVENANCE_DISK_CACHE


# Per-process memoisation. The current repo can't change inside one
# subprocess, so caching the visibility verdict for the lifetime of the
# process turns ~3 redundant calls per tick into 1. We still re-check
# in fresh subprocesses, so a maintainer flipping the repo to private
# is picked up on the next tick (~seconds, not minutes).
_REPO_PUBLIC_MEMO: dict[str, bool] = {}


def _is_repo_public(config: dict) -> bool:
    """Return True iff the current repo is public. Routed through the
    layer (ETag-cached on the REST side); fails closed by treating
    ambiguous results as public so the security gate still runs.

    Memoised per process: three call sites used to fire on every tick;
    they now share one round-trip.
    """
    slug = _get_repo()
    if slug in _REPO_PUBLIC_MEMO:
        return _REPO_PUBLIC_MEMO[slug]
    try:
        resp = gh.get_client().get(f"/repos/{slug}", timeout=15)
    except Exception:
        return True  # fail-closed: treat as public so we still check
    if not resp.ok():
        return True
    visibility = ((resp.body() or {}).get("visibility") or "").lower()
    if not visibility:
        return True
    result = visibility == "public"
    _REPO_PUBLIC_MEMO[slug] = result
    return result


def _parse_provenance_node(repo: str, issue_num: int,
                            issue_node: dict) -> tuple[
                                _IssueProvenance, bool]:
    """Pull `_IssueProvenance` out of one GraphQL `issue` node.

    Returns `(prov, has_next_page)`. The caller is responsible for
    paginating remaining comments via REST when `has_next_page` is True.
    """
    author = (issue_node.get("author") or {}).get("login") or ""
    author_assoc = issue_node.get("authorAssociation") or "NONE"

    comments_node = issue_node.get("comments") or {}
    nodes = comments_node.get("nodes") or []
    page_info = comments_node.get("pageInfo") or {}
    has_next = bool(page_info.get("hasNextPage"))

    comments: list[_ProvenanceComment] = []
    for c in nodes:
        if not isinstance(c, dict):
            continue
        cid = c.get("databaseId") or 0
        login = (c.get("author") or {}).get("login") or ""
        assoc = c.get("authorAssociation") or "NONE"
        comments.append(_ProvenanceComment(
            comment_id=int(cid), login=login, association=assoc))

    return _IssueProvenance(
        repo=repo,
        issue_num=issue_num,
        author_login=author,
        author_association=author_assoc,
        comments=comments,
        fetched_at=time.time(),
    ), has_next


def _paginate_comments_into(prov: _IssueProvenance) -> None:
    """Append comments past the first 100 to `prov` via REST. Used as
    the rare-case tail for issues with >100 comments. Mutates `prov`."""
    client = gh.get_client()
    for page in client.paginate(
            f"/repos/{prov.repo}/issues/{prov.issue_num}/comments",
            max_pages=20):
        if not page.ok():
            raise RuntimeError(
                f"failed to page comments for "
                f"{prov.repo}#{prov.issue_num}: HTTP {page.status}")
        page_body = page.body() or []
        if not isinstance(page_body, list):
            continue
        for rc in page_body:
            if not isinstance(rc, dict):
                continue
            rid = int(rc.get("id") or 0)
            if not rid or any(c.comment_id == rid for c in prov.comments):
                continue
            prov.comments.append(_ProvenanceComment(
                comment_id=rid,
                login=(rc.get("user") or {}).get("login", "") or "",
                association=rc.get("author_association", "NONE") or "NONE",
            ))


# Cap on aliases per batched GraphQL query. GitHub's documented
# limits allow much more (node-count caps in the thousands), but
# response size grows with comment counts, so we keep batches modest.
_PROVENANCE_BATCH_SIZE = 25


def _build_provenance_batch_query(count: int) -> str:
    """Build a single GraphQL query with `count` aliased `issue(...)`
    selections. Each alias `i{k}` uses `$num{k}: Int!`."""
    aliases = "\n".join(
        f"    i{k}: issue(number: $num{k}) {{\n"
        f"      author {{ login }}\n"
        f"      authorAssociation\n"
        f"      comments(first: 100) {{\n"
        f"        nodes {{\n"
        f"          databaseId\n"
        f"          author {{ login }}\n"
        f"          authorAssociation\n"
        f"        }}\n"
        f"        pageInfo {{ hasNextPage endCursor }}\n"
        f"      }}\n"
        f"    }}"
        for k in range(count)
    )
    var_decls = ", ".join(f"$num{k}: Int!" for k in range(count))
    return (
        f"query ProvenanceBatch($owner: String!, $name: String!, "
        f"{var_decls}) {{\n"
        f"  repository(owner: $owner, name: $name) {{\n"
        f"{aliases}\n"
        f"  }}\n"
        f"}}\n"
    )


def fetch_issue_provenances(repo: str,
                             issue_nums: list[int]
                             ) -> dict[int, _IssueProvenance]:
    """Fetch provenance for multiple issues with a single batched
    GraphQL request per chunk of `_PROVENANCE_BATCH_SIZE` issues.

    Returns a `{issue_num: _IssueProvenance}` dict. Missing or
    inaccessible issues are simply absent from the result; callers
    treat absence as a fetch failure (fails-closed via
    `check_issue_provenance`).

    Does not consult the disk cache itself — callers (i.e.
    `_cached_provenance`) are expected to filter cached issues out
    before calling here.
    """
    client = gh.get_client()
    owner, _, name = repo.partition("/")
    if not (owner and name):
        raise RuntimeError(f"unparseable repo slug: {repo!r}")

    out: dict[int, _IssueProvenance] = {}
    nums = list(dict.fromkeys(int(n) for n in issue_nums))  # dedup, ordered
    for start in range(0, len(nums), _PROVENANCE_BATCH_SIZE):
        chunk = nums[start:start + _PROVENANCE_BATCH_SIZE]
        query = _build_provenance_batch_query(len(chunk))
        variables: dict = {"owner": owner, "name": name}
        for k, n in enumerate(chunk):
            variables[f"num{k}"] = n
        resp = client.graphql(query, variables=variables, cache="none")
        if not resp.ok():
            raise RuntimeError(
                f"failed to fetch batched provenance for {repo}: "
                f"HTTP {resp.status}")
        body = resp.body() or {}
        repo_node = (body.get("data") or {}).get("repository") or {}
        for k, n in enumerate(chunk):
            issue_node = repo_node.get(f"i{k}")
            if not issue_node:
                continue
            prov, has_next = _parse_provenance_node(repo, n, issue_node)
            if has_next:
                _paginate_comments_into(prov)
            out[n] = prov
    return out


def fetch_issue_provenance(repo: str, issue_num: int) -> _IssueProvenance:
    """Single-issue fetch. Thin wrapper around the batched fetcher so
    the parsing + REST tail-paginate code is shared.

    Raises `RuntimeError` on layer failure or missing issue node.
    """
    got = fetch_issue_provenances(repo, [issue_num])
    if issue_num not in got:
        raise RuntimeError(
            f"failed to fetch provenance for {repo}#{issue_num}: "
            "GraphQL returned no issue node")
    return got[issue_num]


# ---------------------------------------------------------------------------
# Issue-state batched lookup (used by `fetch_blocked_deps` to resolve
# `depends-on: #N` references without enumerating every open issue).
# ---------------------------------------------------------------------------

_ISSUE_STATE_BATCH_SIZE = 25  # aliases per GraphQL POST, matches provenance
_ISSUE_STATE_TTL_SECONDS = 60.0  # how long a state lookup is reused


def _build_issue_state_batch_query(count: int) -> str:
    """Build a GraphQL query with `count` aliased single-issue lookups
    returning just `number` and `state`. Each alias `i{k}` uses
    `$num{k}: Int!`."""
    aliases = "\n".join(
        f"    i{k}: issue(number: $num{k}) {{ number state }}"
        for k in range(count)
    )
    var_decls = ", ".join(f"$num{k}: Int!" for k in range(count))
    return (
        f"query IssueStateBatch($owner: String!, $name: String!, "
        f"{var_decls}) {{\n"
        f"  repository(owner: $owner, name: $name) {{\n"
        f"{aliases}\n"
        f"  }}\n"
        f"}}\n"
    )


_ISSUE_STATE_CACHE_FILE = "issue-state-cache.json"


def _issue_state_cache_path() -> Path:
    return POD_DIR / _ISSUE_STATE_CACHE_FILE


def _load_issue_state_cache() -> dict[str, dict]:
    p = _issue_state_cache_path()
    try:
        return json.loads(p.read_text())
    except (OSError, FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def _save_issue_state_cache(cache: dict[str, dict]) -> None:
    p = _issue_state_cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, separators=(",", ":"), sort_keys=True))
        tmp.replace(p)
    except OSError:
        pass


def _issue_state_cache_key(repo: str, num: int) -> str:
    return f"{repo}#{num}"


def fetch_issue_states(repo: str, nums: list[int]) -> dict[int, str]:
    """Return `{issue_number: state}` for each number in `nums`.

    State is `"OPEN"`, `"CLOSED"`, or `""` (lookup failed / not found).
    Cached on disk at `.pod/issue-state-cache.json` with TTL
    `_ISSUE_STATE_TTL_SECONDS`; only cache misses (or expired entries)
    trigger a batched GraphQL POST. On transport failure for a chunk,
    the affected numbers are omitted from the result (callers treat
    absence as "not open" — fail-closed; the UI hides annotations
    rather than ever showing wrong ones).
    """
    owner, _, name = repo.partition("/")
    if not (owner and name) or not nums:
        return {}

    cache = _load_issue_state_cache()
    now = time.time()
    out: dict[int, str] = {}
    misses: list[int] = []
    # Dedupe while preserving order so test fixtures stay deterministic.
    deduped = list(dict.fromkeys(int(n) for n in nums))
    for n in deduped:
        entry = cache.get(_issue_state_cache_key(repo, n))
        if (isinstance(entry, dict)
                and isinstance(entry.get("fetched_at"), (int, float))
                and now - float(entry["fetched_at"]) < _ISSUE_STATE_TTL_SECONDS
                and isinstance(entry.get("state"), str)):
            out[n] = entry["state"]
        else:
            misses.append(n)

    if not misses:
        return out

    client = gh.get_client()
    cache_dirty = False
    for start in range(0, len(misses), _ISSUE_STATE_BATCH_SIZE):
        chunk = misses[start:start + _ISSUE_STATE_BATCH_SIZE]
        query = _build_issue_state_batch_query(len(chunk))
        variables: dict = {"owner": owner, "name": name}
        for k, n in enumerate(chunk):
            variables[f"num{k}"] = n
        try:
            resp = client.graphql(query, variables=variables, cache="none")
        except Exception:
            # Transport-level failure: skip this chunk, leave its entries
            # missing from `out`. Callers fail-closed.
            continue
        if not resp.ok():
            continue
        body = resp.body() or {}
        repo_node = (body.get("data") or {}).get("repository") or {}
        for k, n in enumerate(chunk):
            node = repo_node.get(f"i{k}")
            state = ""
            if isinstance(node, dict):
                s = node.get("state")
                if isinstance(s, str):
                    state = s
            out[n] = state
            cache[_issue_state_cache_key(repo, n)] = {
                "state": state, "fetched_at": now,
            }
            cache_dirty = True

    if cache_dirty:
        _save_issue_state_cache(cache)
    return out


def is_trusted(provenance: _IssueProvenance, config: dict) -> tuple[bool, str]:
    """Pure policy: decide whether `provenance` passes the trust rule.

    Returns `(True, "")` if trusted, else `(False, reason)`. Reason is a
    grep-friendly single line naming the offending login and (for
    comments) the comment id.
    """
    sec = config.get("security", {}) or {}
    if not sec.get("trust_only_collaborators", True):
        return True, ""
    trusted_assocs = set(sec.get("trusted_author_associations",
                                  ["OWNER", "MEMBER", "COLLABORATOR"]))
    trusted_users = set(sec.get("trusted_users", []) or [])

    def author_ok(login: str, assoc: str) -> bool:
        return assoc in trusted_assocs or login in trusted_users

    if not author_ok(provenance.author_login, provenance.author_association):
        return False, (f"untrusted issue author @{provenance.author_login} "
                       f"({provenance.author_association})")
    for c in provenance.comments:
        if not author_ok(c.login, c.association):
            return False, (f"untrusted comment c#{c.comment_id} "
                           f"@{c.login} ({c.association})")
    return True, ""


def _cached_provenance(repo: str, issue_num: int, config: dict,
                        *, fresh: bool) -> _IssueProvenance:
    """Shared cache entry point. Consults in-process first, then the
    cross-process disk cache. `fresh=True` invalidates both tiers and
    forces a refetch."""
    key = (repo, issue_num)
    ttl = _provenance_ttl(config)
    disk = _provenance_disk_cache()
    if fresh:
        _provenance_cache.pop(key, None)
        disk.invalidate(repo, issue_num)
    else:
        cached = _provenance_cache.get(key)
        if cached and (time.time() - cached.fetched_at) < ttl:
            return cached
        on_disk = disk.get(repo, issue_num, ttl)
        if on_disk is not None:
            _provenance_cache[key] = on_disk
            return on_disk
    prov = fetch_issue_provenance(repo, issue_num)
    _provenance_cache[key] = prov
    disk.put(prov)
    return prov


def _cached_provenances(repo: str, issue_nums: list[int], config: dict,
                         *, fresh: bool) -> dict[int, _IssueProvenance]:
    """Batched cache entry point. For each requested issue, return a
    cached entry if available within TTL; otherwise group the misses
    into one batched GraphQL fetch.

    Returns `{issue_num: _IssueProvenance}`. Issues that couldn't be
    fetched (e.g. deleted, transferred) are absent from the result, so
    callers must treat absence as a fetch failure.
    """
    ttl = _provenance_ttl(config)
    disk = _provenance_disk_cache()
    out: dict[int, _IssueProvenance] = {}
    misses: list[int] = []
    for n in issue_nums:
        key = (repo, n)
        if fresh:
            _provenance_cache.pop(key, None)
            disk.invalidate(repo, n)
            misses.append(n)
            continue
        cached = _provenance_cache.get(key)
        if cached and (time.time() - cached.fetched_at) < ttl:
            out[n] = cached
            continue
        on_disk = disk.get(repo, n, ttl)
        if on_disk is not None:
            _provenance_cache[key] = on_disk
            out[n] = on_disk
            continue
        misses.append(n)
    if misses:
        fetched = fetch_issue_provenances(repo, misses)
        for n, prov in fetched.items():
            _provenance_cache[(repo, n)] = prov
            disk.put(prov)
            out[n] = prov
    return out


def check_issue_provenance(repo: str, issue_num: int, config: dict,
                            *, fresh: bool = False) -> tuple[bool, str]:
    """High-level gate: short-circuits on private/disabled, fetches (cached
    or fresh per `fresh`), runs `is_trusted`, returns `(ok, reason)`."""
    sec = config.get("security", {}) or {}
    if not sec.get("trust_only_collaborators", True):
        return True, ""
    if not _is_repo_public(config):
        return True, ""
    try:
        prov = _cached_provenance(repo, issue_num, config, fresh=fresh)
    except RuntimeError as e:
        return False, f"failed to fetch provenance for #{issue_num}: {e}"
    except (json.JSONDecodeError, ValueError) as e:
        return False, f"unparseable provenance response for #{issue_num}: {e}"
    return is_trusted(prov, config)


def cmd_check_provenance(config: dict, args) -> None:
    """`pod _check-provenance N [--fresh]` — single-issue gate.

    Exit 0 if trusted, exit 1 with reason on stderr if not.
    Used by `coordination claim` and `coordination read-issue`
    (always with --fresh).
    """
    repo = _get_repo()
    ok, reason = check_issue_provenance(repo, args.issue_num, config,
                                          fresh=args.fresh)
    if ok:
        return
    print(f"pod: provenance check failed for {repo}#{args.issue_num}: {reason}",
          file=sys.stderr)
    sys.exit(1)


def cmd_filter_trusted_issues(config: dict, args) -> None:
    """`pod _filter-trusted-issues [...gh-issue-list-args...]` — runs
    `gh issue list` once, drops untrusted issues in one Python pass, then
    applies `--jq` (if any) to the surviving JSON array. One Python
    startup, regardless of issue count.

    Used by `coordination list-unclaimed`, `coordination orient` blocked
    section, and directly by `plan.md` to replace inline `gh issue list`.
    """
    repo = _get_repo()
    argv: list[str] = ["issue", "list", "--repo", repo]
    for label in (args.label or []):
        argv += ["--label", label]
    if args.state:
        argv += ["--state", args.state]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    # Always include the fields needed to filter, plus whatever the
    # caller asked for.
    requested = set((args.json or "number,title").split(","))
    requested |= {"number"}
    argv += ["--json", ",".join(sorted(requested))]
    r = _gh_cli(*argv, timeout=60)
    if r.returncode != 0:
        print(f"pod: gh issue list failed: {r.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    try:
        items = json.loads(r.stdout or "[]")
    except json.JSONDecodeError as e:
        print(f"pod: unparseable gh issue list output: {e}", file=sys.stderr)
        sys.exit(1)

    sec_enabled = (config.get("security", {}) or {}).get(
        "trust_only_collaborators", True)
    repo_public = _is_repo_public(config) if sec_enabled else False
    filtered: list[dict] = []
    if not (sec_enabled and repo_public):
        filtered = items
    else:
        # Batch the provenance fetch for every candidate issue in one
        # GraphQL round-trip (with disk-cache hits short-circuiting
        # before the network call). Previously this fired one POST per
        # issue per tick per agent, draining the GraphQL bucket.
        nums = [int(it.get("number", 0)) for it in items]
        nums = [n for n in nums if n > 0]
        try:
            provs = _cached_provenances(repo, nums, config, fresh=False)
        except RuntimeError as e:
            print(f"pod: provenance batch failed: {e}", file=sys.stderr)
            sys.exit(1)
        for it in items:
            num = int(it.get("number", 0))
            if num <= 0:
                continue
            prov = provs.get(num)
            if prov is None:
                reason = (f"failed to fetch provenance for #{num}: "
                          "no issue node")
                ok = False
            else:
                ok, reason = is_trusted(prov, config)
            if ok:
                filtered.append(it)
            elif args.include_untrusted:
                marked = dict(it)
                if "title" in marked:
                    marked["title"] = f"[UNTRUSTED: {reason}] {marked['title']}"
                filtered.append(marked)

    out_json = json.dumps(filtered)
    if args.jq:
        jq = subprocess.run(["jq", "-r", args.jq], input=out_json,
                             capture_output=True, text=True, timeout=30)
        if jq.returncode != 0:
            print(f"pod: jq failed: {jq.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        sys.stdout.write(jq.stdout)
    else:
        sys.stdout.write(out_json)
        sys.stdout.write("\n")


def _release_claim(issue_str: str, session_uuid: str, restart_count: int,
                    *, reason: str | None = None) -> bool:
    """Remove the 'claimed' label from an issue and leave an explanatory comment.
    Returns True on success (including idempotent terminal cases — see below),
    False on transient failure (caller may revert history deletion).
    Raises `_GraphQLRateLimited` if a `gh` call returns a rate-limit error,
    so callers can abort the whole release sweep instead of attempting every
    remaining issue against an exhausted bucket.

    Includes a GitHub-side CAS: only releases if the latest claim comment still
    belongs to session_uuid (prevents removing a fresh claim by a different
    agent).

    Idempotency: probes issue state + labels via the REST API first.
    If the `claimed` label is already absent, returns True without spending
    GraphQL quota. If the issue is closed and the label is still present,
    also returns True — pod doesn't read labels off closed issues, and the
    quota-safe choice is to leave them alone.

    `reason`, if given, replaces the default "died after N restart attempts"
    release comment — for callers (e.g. the quota-exhaustion handler) where
    that wording would mislead."""
    import re as _re
    repo = _get_repo()

    client = gh.get_client()

    # Idempotent fast path: REST probe via the layer (ETag-cached).
    # Catches the common stale-history case where the label was already
    # removed (by another reconciler, by hand, or by us on a previous run
    # whose history write was lost). Also handles closed issues.
    try:
        r_probe = client.get(f"/repos/{repo}/issues/{issue_str}", timeout=30)
    except Exception as e:
        log(f"_release_claim: probe failed for #{issue_str}: {e}; falling through")
        r_probe = None
    if r_probe is not None:
        if r_probe.status == 403:
            raise _GraphQLRateLimited(f"REST probe rate-limited for #{issue_str}")
        if r_probe.ok():
            try:
                probe = r_probe.body() or {}
                state = probe.get("state", "")
                labels = [l.get("name", "") for l in probe.get("labels", []) or []]
                if "claimed" not in labels:
                    log(f"_release_claim: #{issue_str} already lacks 'claimed' label "
                        f"(state={state}); treating as released")
                    return True
                if state == "closed":
                    log(f"_release_claim: #{issue_str} is closed; leaving stale 'claimed' "
                        f"label in place (terminal success)")
                    return True
            except (TypeError, ValueError) as e:
                log(f"_release_claim: probe parse failed for #{issue_str}: {e}; falling through")

    # GitHub-side CAS: verify latest claim comment is ours. The REST
    # probe above ensures we only reach this when there's real work.
    try:
        r_cas = _gh_cli(
            "issue", "view", issue_str, "--repo", repo,
            "--json", "comments",
            "--jq", '[.comments[] | select(.body | startswith("Claimed by session"))] | sort_by(.createdAt) | last | .body',
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"Failed to release claim on #{issue_str}: CAS subprocess error: {e}")
        return False
    if _is_rate_limit_error(r_cas):
        raise _GraphQLRateLimited(f"CAS rate-limited for #{issue_str}")
    if r_cas.returncode == 0 and r_cas.stdout.strip():
        m = _re.search(r'Claimed by session `([^`]+)`', r_cas.stdout.strip().strip('"'))
        if m and m.group(1) != session_uuid:
            log(f"Not releasing #{issue_str} — latest claim belongs to {m.group(1)[:8]}, not {session_uuid[:8]}")
            return False

    try:
        r1 = _gh_cli(
            "issue", "edit", issue_str, "--repo", repo,
            "--remove-label", "claimed",
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"Failed to release claim on #{issue_str}: edit subprocess error: {e}")
        return False
    if _is_rate_limit_error(r1):
        raise _GraphQLRateLimited(f"edit rate-limited for #{issue_str}")
    if r1.returncode != 0:
        log(f"Failed to remove claimed label on #{issue_str}: {r1.stderr.strip()}")
        return False
    msg = reason or (
        f"Claim released — worker session `{session_uuid}` died after "
        f"{restart_count} restart attempt(s). Available for reclaim.")
    try:
        r2 = _gh_cli(
            "issue", "comment", issue_str, "--repo", repo,
            "--body", msg, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"Failed to comment on #{issue_str}: {e}")
        return False
    if _is_rate_limit_error(r2):
        raise _GraphQLRateLimited(f"comment rate-limited for #{issue_str}")
    if r2.returncode != 0:
        log(f"Failed to comment on #{issue_str}: {r2.stderr.strip()}")
        return False
    log(f"Released claim on #{issue_str}")
    return True


def sync_claims_from_github():
    """On pod startup, rebuild claim-history.json from GitHub for any claimed
    issues we have no local record of. This lets pod reattach to sessions
    that were running before a pod restart."""
    import re as _re
    repo = _get_repo()

    try:
        r = _gh_cli(
            "issue", "list", "--label", "agent-plan", "--label", "claimed",
            "--state", "open", "--limit", "100", "--json", "number",
            timeout=30,
        )
        if r.returncode != 0:
            return
        issues = [iss["number"] for iss in json.loads(r.stdout)]
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return

    if not issues:
        return

    with _claim_history_filelock():
        history = load_claim_history()
        changed = False
        for issue_num in issues:
            key = str(issue_num)
            if key in history:
                continue  # Already tracked locally
            # Fetch comments to find the most recent claim comment
            try:
                r = _gh_cli(
                    "issue", "view", str(issue_num), "--repo", repo,
                    "--json", "comments",
                    "--jq", '[.comments[] | select(.body | startswith("Claimed by session"))] | sort_by(.createdAt) | last | .body',
                    timeout=60,
                )
                if r.returncode != 0:
                    continue
                comment_body = r.stdout.strip().strip('"')
            except (subprocess.TimeoutExpired, OSError):
                continue
            # Parse: "Claimed by session `UUID` on branch `agent/SHORT_ID`"
            m = _re.search(r'Claimed by session `([^`]+)` on branch `agent/([^`]+)`', comment_body)
            if not m:
                continue
            session_uuid, short_id = m.group(1), m.group(2)
            history[key] = {"session_uuid": session_uuid, "short_id": short_id, "restart_count": 0}
            changed = True
            log(f"Recovered claim: issue #{issue_num} → session {session_uuid[:8]} (short: {short_id})")
        if changed:
            _save_claim_history(history)


def check_dead_claimed_issues(config: dict):
    """Detect locally-known sessions that claimed issues but are now dead.
    Restart them up to max_claim_restarts times, then release the claim.

    Collects actions under the file lock, then executes them outside to avoid
    fork-under-lock (spawn_agent forks) and subprocess-under-lock (_release_claim).
    """
    history = load_claim_history()
    if not history:
        return

    agents = read_all_agents()
    live_uuids = {a.uuid for a in agents if a.status not in ("dead", "stopped")}
    max_restarts = cfg_get(config, "monitor", "max_claim_restarts", default=1)

    to_restart: list[tuple[str, str, str, int]] = []   # (short_id, session_uuid, issue_str, new_count)
    to_release: list[tuple[str, str, int]] = []         # (issue_str, session_uuid, restart_count)

    with _claim_history_filelock():
        history = load_claim_history()  # Re-read under lock
        changed = False
        for issue_str, info in list(history.items()):
            session_uuid = info.get("session_uuid", "")
            if not session_uuid or session_uuid in live_uuids:
                continue  # Still running or malformed entry
            if info.get("released"):
                continue  # Already released, don't restart

            restart_count = info.get("restart_count", 0)
            short_id = info.get("short_id", "")

            if restart_count < max_restarts:
                to_restart.append((short_id, session_uuid, issue_str, restart_count + 1))
                info["restart_count"] = restart_count + 1
                changed = True
            else:
                to_release.append((issue_str, session_uuid, restart_count))
                # Mark as released rather than deleting, so record_claim
                # won't re-add it with restart_count=0 (which would cause
                # an infinite restart loop).
                info["released"] = True
                changed = True

        if changed:
            _save_claim_history(history)
    # Lock released before any fork or subprocess call

    # Deduplicate: if one session claimed N issues, only restart it once.
    seen_uuids: set[str] = set()
    deduped_restart = []
    for entry in to_restart:
        _, session_uuid, _, _ = entry
        if session_uuid not in seen_uuids:
            seen_uuids.add(session_uuid)
            deduped_restart.append(entry)

    # Respect target_agents: don't spawn more agents than the target.
    target = get_effective_target()
    if target is not None:
        current_count = len([a for a in agents if a.status not in ("dead", "stopped")])
        slots = max(0, target - current_count)
        if len(deduped_restart) > slots:
            log(f"Capping restarts at {slots} (target={target}, running={current_count})")
            deduped_restart = deduped_restart[:slots]

    for short_id, session_uuid, issue_str, new_count in deduped_restart:
        log(f"Dead session {session_uuid} claimed #{issue_str}, "
            f"restarting (attempt {new_count}/{max_restarts})")
        # Use a fresh agent_id — reusing the old short_id causes all restart
        # agents to share one state file, making old processes invisible.
        spawn_agent(config, resume_uuid=session_uuid)

    # Re-read live agents for the release check (state may have changed
    # during restart spawning above).
    fresh_agents = read_all_agents()
    live_claimed: dict[int, str] = {}  # issue → uuid of live agent working on it
    for a in fresh_agents:
        if a.status not in ("dead", "stopped", "killed") and a.claimed_issue > 0:
            live_claimed[a.claimed_issue] = a.uuid

    failed_releases: list[tuple[str, str, int]] = []
    # Iterate over a copy so the sentinel handler can re-queue what's left.
    remaining = list(to_release)
    try:
        while remaining:
            issue_str, session_uuid, restart_count = remaining[0]
            issue_int = int(issue_str)
            other = live_claimed.get(issue_int)
            if other and other != session_uuid:
                log(f"Not releasing #{issue_str} — agent {other[:8]} still has it")
                # Rebind history entry to the live owner so it stays tracked
                # (the dead session's tombstone was already written above).
                other_agent = next((a for a in fresh_agents if a.uuid == other), None)
                if other_agent:
                    record_claim(issue_int, other_agent.uuid, other_agent.short_id)
                remaining.pop(0)
                continue
            log(f"Max restarts reached for #{issue_str}, releasing claim")
            if not _release_claim(issue_str, session_uuid, restart_count):
                failed_releases.append((issue_str, session_uuid, restart_count))
            remaining.pop(0)
    except _GraphQLRateLimited as e:
        log(f"check_dead_claimed_issues: aborting release sweep — {e}")
        # Items still in `remaining` (including the one that just raised) had
        # their `released: True` tombstones prewritten above. Without re-queuing
        # them they'd be skipped on every future housekeeping pass — turning a
        # transient rate-limit blip into a permanent leak. Re-queue them so the
        # next pass (after quota recovers) re-attempts the release.
        failed_releases.extend(remaining)

    # Re-add any failed releases so we retry next housekeeping cycle
    if failed_releases:
        with _claim_history_filelock():
            history = load_claim_history()
            for issue_str, session_uuid, restart_count in failed_releases:
                existing = history.get(issue_str, {})
                if not existing or existing.get("released"):
                    history[issue_str] = {
                        "session_uuid": session_uuid,
                        "short_id": "",
                        "restart_count": restart_count,
                    }
            _save_claim_history(history)


_RECONCILE_BATCH_QUERY = """
query ReconcileBatch($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    issues(first: 50, states: OPEN,
           labels: ["agent-plan", "claimed"]) {
      nodes {
        number
        state
        labels(first: 20) { nodes { name } }
        comments(last: 100) {
          nodes { databaseId createdAt body }
        }
      }
    }
  }
}
"""


def reconcile_untracked_github_claims():
    """Periodic safety net: find GitHub issues with 'claimed' label that
    are not tracked in local claim-history.json, and either backfill them
    (if the owning session is still alive) or release them (if dead and
    past the grace period).

    Burner-rewrite (B1): one GraphQL query returns state + labels +
    latest comments for every claimed agent-plan issue in one shot,
    replacing the previous N+1 fan-out (`gh issue list` plus ~3 REST
    `gh issue view`s per issue). Per-issue REST writes (label remove +
    comment) are still made, only for the small set that actually need
    releasing.

    Fail-closed: any GitHub API error skips that issue rather than
    releasing it.
    """
    import re as _re

    repo = _get_repo()
    # 60s is enough to cover the window between an agent posting the
    # "Claimed by session" comment and writing its local claim-history
    # entry — anything older with a dead owner UUID is a real orphan.
    # The previous 10-minute setting compounded with the 10-minute
    # housekeeping cadence to delay orphan cleanup by up to 20 minutes;
    # a session whose GitHub claim survives 60 seconds past its UUID
    # disappearing from `read_all_agents()` is unambiguously dead.
    grace_seconds = 60
    client = gh.get_client()

    owner, _, name = repo.partition("/")
    if not (owner and name):
        return

    try:
        resp = client.graphql(_RECONCILE_BATCH_QUERY,
                              variables={"owner": owner, "name": name})
    except Exception as e:
        log(f"reconcile_untracked_github_claims: batch query failed: {e}")
        return
    if resp.status == 403:
        log("reconcile_untracked_github_claims: aborting — GraphQL rate-limited")
        return
    if not resp.ok():
        return
    body = resp.body() or {}
    nodes = (((body.get("data") or {}).get("repository") or {}
              ).get("issues") or {}).get("nodes") or []
    if not nodes:
        return

    history = load_claim_history()
    tracked = set()
    released_at: dict[int, str] = {}
    for k, v in history.items():
        try:
            issue = int(k)
        except (TypeError, ValueError):
            continue
        if v.get("released"):
            released_at[issue] = v.get("released_comment_time", "")
        else:
            tracked.add(issue)

    agents = read_all_agents()
    live_uuids = {a.uuid for a in agents
                  if a.status not in ("dead", "stopped")}

    now_epoch = time.time()
    released_count = 0
    backfilled_count = 0

    claim_re = _re.compile(
        r'Claimed by session `([^`]+)` on branch `agent/([^`]+)`')

    for node in nodes:
        if not isinstance(node, dict):
            continue
        issue_num = node.get("number")
        if not isinstance(issue_num, int):
            continue
        if issue_num in tracked:
            continue

        labels = [l.get("name") for l in
                  (((node.get("labels") or {}).get("nodes")) or [])]
        if "claimed" not in labels:
            continue
        comments = ((node.get("comments") or {}).get("nodes")) or []
        # Latest "Claimed by session …" comment by createdAt.
        claim_comments = [c for c in comments
                          if isinstance(c, dict)
                          and (c.get("body") or "").startswith(
                              "Claimed by session")]
        if not claim_comments:
            continue
        claim_comments.sort(key=lambda c: c.get("createdAt") or "")
        latest = claim_comments[-1]
        comment_body = latest.get("body", "") or ""
        comment_time = latest.get("createdAt", "") or ""

        m = claim_re.search(comment_body)
        if not m:
            continue
        owner_uuid, short_id = m.group(1), m.group(2)

        # Same-claim short-circuit: we already released this exact comment.
        prev_released_time = released_at.get(issue_num, "")
        if (prev_released_time and comment_time
                and comment_time <= prev_released_time):
            continue

        # Owner alive → backfill into history; don't release.
        if owner_uuid in live_uuids:
            with _claim_history_filelock():
                h = load_claim_history()
                key = str(issue_num)
                if key not in h or h[key].get("released"):
                    h[key] = {"session_uuid": owner_uuid,
                              "short_id": short_id,
                              "restart_count": 0}
                    _save_claim_history(h)
                    backfilled_count += 1
                    log(f"Reconcile: backfilled claim #{issue_num} → "
                        f"session {owner_uuid[:8]}")
            continue

        # Owner dead → grace check.
        try:
            claim_dt = datetime.datetime.fromisoformat(
                comment_time.replace("Z", "+00:00"))
            claim_epoch = claim_dt.timestamp()
        except (ValueError, TypeError):
            continue
        age = now_epoch - claim_epoch
        if age < grace_seconds:
            continue

        # Re-check liveness right before mutating.
        fresh_agents = read_all_agents()
        fresh_live = {a.uuid for a in fresh_agents
                       if a.status not in ("dead", "stopped", "killed")}
        if owner_uuid in fresh_live:
            continue
        if any(a.claimed_issue == issue_num and a.uuid in fresh_live
               for a in fresh_agents):
            continue

        # Per-issue REST writes: label remove + release comment. The
        # batched GraphQL response is our "label still present" probe;
        # the layer's ETag cache makes a follow-up check fast if a
        # caller wants extra paranoia, but in steady state the GraphQL
        # snapshot is accurate within seconds.
        try:
            r3 = _gh_cli("issue", "edit", str(issue_num), "--repo", repo,
                          "--remove-label", "claimed", timeout=30)
            if _is_rate_limit_error(r3):
                log(f"reconcile: rate-limited on label remove for "
                    f"#{issue_num}; aborting sweep")
                break
            if r3.returncode != 0:
                continue
            age_str = f"{int(age // 3600)}h{int((age % 3600) // 60)}m"
            msg = (f"Stale claim released by reconciler — session "
                   f"`{owner_uuid}` is no longer running "
                   f"(claimed {age_str} ago). Available for reclaim.")
            r_comment = _gh_cli("issue", "comment", str(issue_num),
                                 "--repo", repo, "--body", msg, timeout=30)
            if _is_rate_limit_error(r_comment):
                log(f"reconcile: rate-limited on release comment for "
                    f"#{issue_num}; aborting sweep")
                break
            with _claim_history_filelock():
                h = load_claim_history()
                h[str(issue_num)] = {
                    "session_uuid": owner_uuid,
                    "short_id": short_id,
                    "restart_count": 0,
                    "released": True,
                    "released_comment_time": comment_time,
                }
                _save_claim_history(h)
            released_count += 1
            log(f"Reconcile: released stale claim #{issue_num} "
                f"(owner {owner_uuid[:8]}, age {age_str})")
        except (subprocess.TimeoutExpired, OSError):
            continue

    if released_count or backfilled_count:
        log(f"Reconcile: {backfilled_count} backfilled, "
            f"{released_count} released")


def check_dead_pr_claimed_prs(config: dict):
    """Release `repair-claimed` labels on PRs whose owning session is dead.

    Counterpart to check_dead_claimed_issues, but for the PR-claim namespace
    used by repair agents. Unlike issue claims, repair claims are never
    restarted — if the session died, we clear the claim and let a fresh
    repair agent pick the PR up next time.

    Also reconciles untracked `repair-claimed` labels (label present on GitHub
    but no local history entry) and stale-by-time claims where the PID cannot
    be resolved. Stale threshold defaults to 2× `repair.stuck_ci_minutes`.

    Fail-closed: any gh / parsing error on a specific PR skips it.
    """
    repo = _get_repo()
    stuck_minutes = cfg_get(config, "repair", "stuck_ci_minutes", default=120)
    stale_seconds = int(stuck_minutes) * 60 * 2

    # 1. Find all open PRs carrying repair-claimed.
    try:
        r = _gh_cli(
            "pr", "list", "--repo", repo, "--state", "open",
            "--label", "repair-claimed", "--limit", "100",
            "--json", "number", timeout=30,
        )
        if _is_rate_limit_error(r):
            log("check_dead_pr_claimed_prs: aborting — GraphQL rate-limited "
                "on initial pr list")
            return
        if r.returncode != 0:
            return
        labelled = {pr["number"] for pr in json.loads(r.stdout)}
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return

    history = load_pr_claim_history()
    agents = read_all_agents()
    live_uuids = {a.uuid for a in agents if a.status not in ("dead", "stopped", "killed")}
    # Also treat any PR actively tracked by a live agent's state.repair_pr as
    # owned, even if the local history hasn't caught up yet (e.g. within the
    # few-second window between label write and record_pr_claim).
    live_repair_prs = {a.repair_pr for a in agents
                        if a.repair_pr > 0 and a.uuid in live_uuids}

    now_epoch = time.time()
    grace_seconds = 300  # 5 min — don't yank labels out from under fresh claims
    to_release: list[tuple[int, str]] = []  # (pr_num, reason)
    released = 0

    # 2. Labelled PRs: check live-session liveness via history + agent state.
    try:
        for pr_num in labelled:
            if pr_num in live_repair_prs:
                continue  # A live agent reports this PR as its repair_pr
            entry = history.get(str(pr_num))
            if not entry:
                # Untracked label. Could be a leak (label added by a failed
                # claim run that died before record_pr_claim) or a very fresh
                # claim whose record hasn't been written yet. Fetch the latest
                # claim comment to distinguish, and only release if:
                #  - the comment is older than the grace window, AND
                #  - its owner session is not live.
                try:
                    client = gh.get_client()
                    latest: dict | None = None
                    for page in client.paginate(
                            f"/repos/{repo}/issues/{pr_num}/comments",
                            max_pages=10):
                        if not page.ok():
                            if page.status == 403:
                                raise _GraphQLRateLimited(
                                    f"comment fetch rate-limited for PR #{pr_num}")
                            break
                        for c in (page.body() or []):
                            if not isinstance(c, dict):
                                continue
                            body = c.get("body", "") or ""
                            if not body.startswith("Claimed PR repair by session"):
                                continue
                            ca = c.get("created_at", "") or ""
                            if latest is None or ca > (latest.get("created_at", "") or ""):
                                latest = {"body": body, "created_at": ca}
                    if latest is None:
                        # No claim comment at all — label is orphaned from a
                        # failed claim attempt. Grace-release.
                        to_release.append((pr_num, "orphaned label: no claim comment"))
                        continue
                    cdata = latest
                except (TypeError, ValueError):
                    continue
                import re as _re
                m = _re.search(r'Claimed PR repair by session `([^`]+)`', cdata.get("body", ""))
                if not m:
                    continue
                owner_uuid = m.group(1)
                try:
                    from datetime import datetime
                    created_at = cdata.get("created_at", "")
                    claim_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
                except (ValueError, TypeError):
                    continue
                age = now_epoch - claim_ts
                if age < grace_seconds:
                    continue  # Too fresh, might be a claim mid-write
                if owner_uuid in live_uuids:
                    continue  # Owner is alive — leave their label alone
                to_release.append((pr_num, f"untracked: owner {owner_uuid[:8]} not live (age {int(age)}s)"))
                continue

            owner_uuid = entry.get("session_uuid", "")
            if owner_uuid in live_uuids:
                continue  # Still being worked on

            claimed_at = entry.get("claimed_at", 0)
            age = now_epoch - claimed_at if claimed_at else stale_seconds + 1
            if age < grace_seconds:
                continue
            to_release.append((pr_num, f"owner {owner_uuid[:8]} no longer live (age {int(age)}s)"))

        # 3. Issue the releases.
        for pr_num, reason in to_release:
            try:
                r = _gh_cli(
                    "pr", "edit", str(pr_num), "--repo", repo,
                    "--remove-label", "repair-claimed", timeout=30,
                )
                if _is_rate_limit_error(r):
                    raise _GraphQLRateLimited(
                        f"label remove rate-limited for PR #{pr_num}")
                if r.returncode != 0:
                    continue
                r_comment = _gh_cli(
                    "pr", "comment", str(pr_num), "--repo", repo,
                    "--body", f"Repair claim released by reconciler: {reason}.",
                    timeout=30,
                )
                if _is_rate_limit_error(r_comment):
                    raise _GraphQLRateLimited(
                        f"release comment rate-limited for PR #{pr_num}")
                clear_pr_claim(pr_num)
                released += 1
                log(f"check_dead_pr_claimed_prs: released repair claim on PR #{pr_num} ({reason})")
            except (subprocess.TimeoutExpired, OSError):
                continue
    except _GraphQLRateLimited as e:
        log(f"check_dead_pr_claimed_prs: aborting sweep — {e}")

    # 4. Clean history entries for PRs that no longer have the label
    #    (claim was cleared by claim-pr-repair itself, or the PR merged/closed).
    stale_history_keys = [
        k for k in history
        if int(k) not in labelled and (now_epoch - history[k].get("claimed_at", 0)) > 60
    ]
    if stale_history_keys:
        with _claim_history_filelock():
            h = load_pr_claim_history()
            for k in stale_history_keys:
                h.pop(k, None)
            _save_pr_claim_history(h)

    if released:
        log(f"check_dead_pr_claimed_prs: released {released} stale repair claim(s)")


# ---------------------------------------------------------------------------
# Agent Lifecycle
# ---------------------------------------------------------------------------

def _remove_path_any(path: Path) -> None:
    """Remove `path` whatever its type — directory, regular file, symlink
    (including broken symlink).  Best-effort; logs but doesn't raise.

    ``shutil.rmtree`` is a no-op on non-directories, and ``Path.exists()``
    returns False for broken symlinks — both pitfalls have bitten this
    codebase before.  Use ``lexists`` semantics so broken links are seen.
    """
    try:
        # lstat will succeed for broken symlinks where stat won't
        st = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as e:
        log(f"Warning: lstat({path}) failed during scrub: {e}")
        return
    try:
        if stat.S_ISDIR(st.st_mode):
            shutil.rmtree(str(path), ignore_errors=True)
        else:
            # Regular file, symlink (including broken), socket, etc.
            path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        log(f"Warning: failed to remove {path} during scrub: {e}")


def _live_branches(exclude_short_id: str | None = None) -> set[str]:
    """Branches owned by any non-dead live agent state file.

    ``exclude_short_id`` lets a caller filter out a specific agent
    (typically itself) so its own pre-registered branch doesn't show up
    as a "collision" against its own setup.
    """
    owned: set[str] = set()
    for agent in read_all_agents():
        if agent.short_id == exclude_short_id:
            continue
        if agent.branch and agent.status != "dead":
            owned.add(agent.branch)
    return owned


def _scrub_worktree_remnants(short_id: str, wt_dir: Path) -> None:
    """Best-effort removal of any partial worktree/branch state for short_id.

    Handles four kinds of leftovers from a crashed or partially-failed
    ``git worktree add``:
      - the worktree path itself (``worktrees/<short_id>``) — directory,
        regular file, or symlink (including broken symlink)
      - the worktree admin entry (``.git/worktrees/<short_id>``)
      - a stale branch lock (``.git/refs/heads/agent/<short_id>.lock``)
      - the branch ref (``agent/<short_id>``)

    Idempotent: safe to call when nothing exists.  We need this because
    ``git worktree add -b`` is non-atomic — it writes the branch ref before
    the worktree dir is fully checked out, so a partial failure (disk full,
    SIGTERM, lock contention) can leave any subset of these behind.

    Catches ``OSError`` and ``TimeoutExpired`` from individual steps so
    one transient failure (e.g. ``git worktree prune`` timing out) does
    not abort the whole scrub or escape past callers that only handle
    ``CalledProcessError``.
    """
    branch = f"agent/{short_id}"
    # 1. Unlock first so prune below can sweep the admin entry
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "worktree", "unlock", str(wt_dir)],
            capture_output=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        log(f"Warning: 'git worktree unlock {wt_dir}' timed out during scrub")
    # 2. Remove the worktree path — handles dir, file, and (broken) symlink.
    #    Use lexists so broken symlinks are seen.
    if os.path.lexists(str(wt_dir)):
        _remove_path_any(wt_dir)
    # 3. Always prune — clears admin entries even when wt_dir is already gone
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "worktree", "prune"],
            capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        log("Warning: 'git worktree prune' timed out during scrub")
    # 4. Clear stale branch lock that can outlive a crashed git process
    lock_path = PROJECT_DIR / ".git" / "refs" / "heads" / f"{branch}.lock"
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        log(f"Warning: could not remove stale branch lock {lock_path}: {e}")
    # 5. Force-delete the branch ref
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True, timeout=10, cwd=str(PROJECT_DIR),
        )
    except subprocess.TimeoutExpired:
        log(f"Warning: 'git branch -D {branch}' timed out during scrub")


def setup_worktree(config: dict, short_id: str,
                   *, own_agent_id: str | None = None) -> tuple[str, str]:
    """Create a fresh git worktree. Returns (worktree_path, branch_name).

    Self-healing: scrubs any leftover state from a previous crash or
    partial failure before attempting creation, and on failure scrubs
    again so the next retry starts from a clean slate.

    Refuses to scrub a worktree/branch already claimed by a *different*
    live agent (defence against ``session_uuid[:8]`` collisions or
    operator-supplied short_id reuse) — raises ``CalledProcessError``
    instead of destroying another agent's working state.  Pass
    ``own_agent_id`` so the caller's own pre-registered state is not
    treated as a collision against itself.

    Always raises ``subprocess.CalledProcessError`` on any failure
    (timeout, OSError, git error), with the underlying stderr embedded
    in the message so callers can log *why* it failed.  Callers
    therefore only need to catch one exception type.
    """
    base = PROJECT_DIR / cfg_get(config, "project", "worktree_base", default="worktrees")
    base.mkdir(parents=True, exist_ok=True)
    wt_dir = base / short_id
    branch = f"agent/{short_id}"

    # Refuse to clobber a worktree/branch claimed by a *different* live
    # agent.  Defence against the (astronomically rare but possible)
    # session_uuid[:8] collision: two live sessions with the same
    # short_id would otherwise have one's setup_worktree wipe the
    # other's working tree.
    if branch in _live_branches(exclude_short_id=own_agent_id):
        raise subprocess.CalledProcessError(
            1, ("setup_worktree", short_id),
            output=b"",
            stderr=(f"refusing to scrub: branch {branch} is owned by another live agent "
                    f"(short_id collision?)").encode(),
        )

    # Fetch latest default branch (network failure is non-fatal — git uses
    # cached refs.  Timeout is swallowed since worktree add will fail
    # cleanly below if the ref really is missing.)
    base_branch = _get_base_branch()
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "fetch", "origin", base_branch, "--quiet"],
            capture_output=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        log(f"Warning: 'git fetch origin {base_branch}' timed out; using cached refs")

    # Scrub any leftover worktree/branch/admin state from a previous crash
    _scrub_worktree_remnants(short_id, wt_dir)

    # Create worktree.  Normalize all failure modes to CalledProcessError
    # (git non-zero, subprocess timeout, OSError from filesystem) so the
    # caller in agent_process_main only needs one except clause and the
    # partial state always gets scrubbed.
    cmd = ["git", "-C", str(PROJECT_DIR), "worktree", "add", "-b", branch,
           str(wt_dir), f"origin/{base_branch}", "--quiet"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60, check=True)
    except subprocess.CalledProcessError as e:
        _scrub_worktree_remnants(short_id, wt_dir)
        stderr = (e.stderr or b"").decode(errors="replace").strip()
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd,
            output=e.output,
            stderr=(stderr + " [setup_worktree scrubbed partial state]").encode(),
        ) from e
    except subprocess.TimeoutExpired as e:
        _scrub_worktree_remnants(short_id, wt_dir)
        stderr = (e.stderr or b"").decode(errors="replace").strip() if e.stderr else ""
        raise subprocess.CalledProcessError(
            124, cmd,  # 124 = GNU timeout convention
            output=e.stdout,
            stderr=(stderr + f" [git worktree add timed out after {e.timeout}s; "
                    "partial state scrubbed]").encode(),
        ) from e
    except OSError as e:
        _scrub_worktree_remnants(short_id, wt_dir)
        raise subprocess.CalledProcessError(
            1, cmd,
            output=b"",
            stderr=(f"OSError invoking git worktree add: {e} "
                    "[partial state scrubbed]").encode(),
        ) from e

    if not wt_dir.exists():
        _scrub_worktree_remnants(short_id, wt_dir)
        raise subprocess.CalledProcessError(
            1, "git worktree add",
            output=r.stdout,
            stderr=(r.stderr + b" [dir not created; partial state scrubbed]"),
        )

    return str(wt_dir), branch


def _pod_installed_files() -> list[str]:
    """Return list of relative paths that pod delivers into a worktree.

    Scans the bundled agent-config/claude for commands and skills files.
    These are automatically added to the protected-files list so agents
    cannot modify pod-delivered files in PRs (but can create new ones).
    """
    data_config = _data_dir() / "agent-config" / "claude"
    result: list[str] = []
    for subdir in ("commands", "skills"):
        src = data_config / subdir
        if src.is_dir():
            for item in src.rglob("*"):
                if item.is_file():
                    result.append(f".claude/{subdir}/{item.relative_to(src)}")
    result.append("AGENTS.md")
    return result


_POD_CODEX_AGENTS_SENTINEL = "<!-- pod-managed-codex-agents -->"


def _read_text(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _codex_project_guidance(wt_dir: str) -> tuple[str, str]:
    """Return repo AGENTS.md and project .claude/CLAUDE.md text."""
    root = Path(wt_dir)
    repo_agents = _read_text(root / "AGENTS.md")
    if _POD_CODEX_AGENTS_SENTINEL in repo_agents:
        repo_agents = ""
    project_claude = _read_text(root / ".claude" / "CLAUDE.md")
    return repo_agents, project_claude


def _compose_pod_codex_agents(wt_dir: str) -> str:
    """Build the pod-managed AGENTS.md payload for Codex worktrees."""
    repo_agents, project_claude = _codex_project_guidance(wt_dir)
    pod_agents = (_data_dir() / "agent-config" / "codex" / "AGENTS.md").read_text()
    sections = [_POD_CODEX_AGENTS_SENTINEL]
    if project_claude.strip():
        sections.append("# Project .claude/CLAUDE.md\n\n" + project_claude.strip())
    sections.append("# Pod Codex Guidance\n\n" + pod_agents.strip())
    return "\n\n".join(sections) + "\n"


def install_agent_config(wt_dir: str, backend: str = "claude"):
    """Install bundled commands, skills, and agent instructions into worktree.

    For Claude: merges agent-config/claude into .claude/ (commands, skills,
    CLAUDE.md).
    For Codex: appends pod-managed guidance to the worktree's root AGENTS.md,
    creating it when needed. Skills are installed into CODEX_HOME by
    _setup_codex_home(). Commands are read at launch time by _worker_prompt()
    and delivered via stdin.
    """
    data_config = _data_dir() / "agent-config" / backend

    if backend == "codex":
        # Mirror Claude's .claude/CLAUDE.md behavior: append pod guidance to
        # the repo's AGENTS.md in-place, creating it when absent.
        dst_md = Path(wt_dir) / "AGENTS.md"
        existing = _read_text(dst_md)
        agent_text = _compose_pod_codex_agents(wt_dir)
        if _POD_CODEX_AGENTS_SENTINEL in existing:
            base_text = existing.split(_POD_CODEX_AGENTS_SENTINEL, 1)[0].rstrip()
            if base_text:
                dst_md.write_text(base_text + "\n\n" + agent_text)
            else:
                dst_md.write_text(agent_text)
        elif agent_text not in existing:
            if existing.strip():
                dst_md.write_text(existing.rstrip() + "\n\n" + agent_text)
            else:
                dst_md.write_text(agent_text)
        # Skills and commands are handled by _setup_codex_home and
        # _worker_prompt respectively — nothing else to install here.
    else:
        # Claude: install into .claude/
        wt_claude = Path(wt_dir) / ".claude"
        wt_claude.mkdir(parents=True, exist_ok=True)

        # Commands
        src_cmds = data_config / "commands"
        if src_cmds.is_dir():
            dst_cmds = wt_claude / "commands"
            shutil.copytree(str(src_cmds), str(dst_cmds), dirs_exist_ok=True)

        # Skills
        src_skills = data_config / "skills"
        if src_skills.is_dir():
            dst_skills = wt_claude / "skills"
            shutil.copytree(str(src_skills), str(dst_skills), dirs_exist_ok=True)

        # Append agent CLAUDE.md to project CLAUDE.md
        agent_md = data_config / "CLAUDE.md"
        if agent_md.is_file():
            dst_md = wt_claude / "CLAUDE.md"
            existing = dst_md.read_text() if dst_md.is_file() else ""
            agent_text = agent_md.read_text()
            if agent_text not in existing:
                with open(dst_md, "a") as f:
                    f.write("\n\n" + agent_text)


def _once_prompt(backend: str) -> str:
    """Return the `/once` prompt for a one-shot priority dispatch.

    Claude consumes the slash form directly. Codex has no slash-command
    system, so we read the template body from
    `pod/data/agent-config/codex/commands/once.md`. Falls back to the
    literal `/once` if no template exists, matching `_worker_prompt`'s
    fallback behaviour.
    """
    if backend == "codex":
        cmd_file = _data_dir() / "agent-config" / "codex" / "commands" / "once.md"
        if cmd_file.is_file():
            return cmd_file.read_text()
        log("Warning: no Codex command template for 'once', using /once raw")
    return "/once"


def _worker_prompt(config: dict, worker_type: str,
                    backend: str | None = None) -> str:
    """Return the prompt text for a given worker type.

    For Claude: returns the slash command name (e.g. "/work").
    For Codex: reads the command template file and returns its full text,
    since Codex has no slash command system.

    Pass `backend` to override the configured value (required in auto mode).
    """
    if backend is None:
        backend = _backend(config)
    worker_types = cfg_get(config, "worker_types", default={})
    wt_cfg = worker_types.get(worker_type, {})
    prompt = wt_cfg.get("prompt", f"/{worker_type}")

    if backend == "codex" and prompt.startswith("/"):
        # Slash command — resolve to the command template file body
        cmd_name = prompt.lstrip("/")
        cmd_file = _data_dir() / "agent-config" / "codex" / "commands" / f"{cmd_name}.md"
        if cmd_file.is_file():
            prompt = cmd_file.read_text()
        # Fallback: use the command name as-is
        else:
            log(f"Warning: no Codex command template for '{cmd_name}', using raw prompt")
    return prompt


def copy_build_cache(wt_dir: str, config: dict):
    """Seed the build cache directory in a fresh worktree.

    Subdirectories listed in ``build_cache_symlink_subdirs`` (default
    ``["packages"]``) are symlinked back to the source — they hold immutable
    dependency artefacts (e.g. Lake's downloaded packages) that all worktrees
    can safely share. Everything else under ``build_cache_dir`` is rsynced so
    each worktree's own build outputs stay isolated.
    """
    cache_dir = cfg_get(config, "project", "build_cache_dir", default=".lake")
    cache_src = PROJECT_DIR / cache_dir
    timeout = cfg_get(config, "project", "build_cache_timeout", default=600)
    symlink_subdirs = cfg_get(
        config, "project", "build_cache_symlink_subdirs", default=["packages"],
    )
    if not cache_src.is_dir():
        return

    cache_dest = Path(wt_dir) / cache_dir
    cache_dest.mkdir(parents=True, exist_ok=True)

    # Symlink shared subdirs first so rsync skips them.
    rsync_excludes = []
    for name in symlink_subdirs:
        src_sub = cache_src / name
        dest_sub = cache_dest / name
        rsync_excludes.extend(["--exclude", f"/{name}"])
        if not src_sub.is_dir():
            continue
        if dest_sub.exists() or dest_sub.is_symlink():
            continue
        try:
            os.symlink(src_sub.resolve(), dest_sub)
        except OSError as e:
            log(f"Warning: symlink of {cache_dir}/{name} failed: {e}")

    try:
        r = subprocess.run(
            ["rsync", "-a", "--quiet", *rsync_excludes,
             f"{cache_src}/", f"{cache_dest}/"],
            capture_output=True, timeout=timeout,
        )
        if r.returncode != 0:
            log(f"Warning: rsync of {cache_dir} failed (exit {r.returncode}): "
                f"{r.stderr.decode(errors='replace').strip()[:200]}")
            # Remove partial copy to avoid corrupt cache
            if cache_dest.is_dir():
                shutil.rmtree(str(cache_dest), ignore_errors=True)
    except subprocess.TimeoutExpired:
        log(f"Warning: rsync of {cache_dir} timed out after {timeout}s "
            f"(source may be too large)")
        # Remove partial copy to avoid corrupt cache
        if cache_dest.is_dir():
            shutil.rmtree(str(cache_dest), ignore_errors=True)
    except OSError as e:
        log(f"Warning: rsync of {cache_dir} failed: {e}")


def cleanup_worktree(wt_dir: str, branch: str):
    """Remove worktree and delete branch.

    Uses shutil.rmtree instead of ``git worktree remove`` because the latter
    takes a git lock and can time out when many worktrees exist concurrently.
    After removing the directory we run ``git worktree prune`` to update git's
    metadata.  Errors are logged rather than silently suppressed.
    """
    if os.path.isdir(wt_dir):
        # Unlock first so prune can clean up metadata
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "worktree", "unlock", wt_dir],
            capture_output=True, timeout=10,
        )
        shutil.rmtree(wt_dir, ignore_errors=True)
        if os.path.isdir(wt_dir):
            log(f"Warning: failed to fully remove worktree {wt_dir}")

    # Tell git to prune stale worktree metadata (fast — just scans admin dir)
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "worktree", "prune"],
            capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        log("Warning: git worktree prune timed out")

    r = subprocess.run(
        ["git", "branch", "-D", branch],
        capture_output=True, text=True, timeout=10, cwd=str(PROJECT_DIR),
    )
    if r.returncode != 0 and "not found" not in r.stderr:
        log(f"Warning: git branch -D {branch} failed: {r.stderr.strip()}")


CLEANUP_LOCK_PATH = POD_DIR / "cleanup.lock"
CLEANUP_MIN_AGE_SECONDS = 120  # Don't delete worktrees younger than this


def _is_pod_owned_worktree(entry: Path) -> bool:
    """Check whether a worktree directory was created by pod.

    Verifies that a corresponding ``agent/<dirname>`` branch exists, which
    is the naming convention pod uses.  This prevents accidental deletion
    of non-pod directories that happen to live under worktree_base.
    """
    branch = f"agent/{entry.name}"
    r = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
        capture_output=True, timeout=10, cwd=str(PROJECT_DIR),
    )
    return r.returncode == 0


def cleanup_stale_worktrees(config: dict, *, verbose: bool = False,
                            min_age: float = CLEANUP_MIN_AGE_SECONDS,
                            force: bool = False) -> int:
    """Remove worktree directories that no running agent owns.

    Returns the number of worktrees actually removed.  JSONL session logs
    live in the isolated Claude config dir (.pod/claude-config/projects/)
    and are NOT touched by this function.

    Uses a cross-process file lock so only one cleanup runs at a time
    (prevents thundering-herd when many agents hit housekeeping together).
    Skips worktrees younger than *min_age* seconds to avoid racing with
    agents that are still setting up.  Only deletes directories that are
    confirmed pod-owned (have a matching ``agent/<name>`` branch).
    """
    import concurrent.futures

    base = PROJECT_DIR / cfg_get(config, "project", "worktree_base", default="worktrees")
    if not base.is_dir():
        return 0

    # --- Cross-process lock (non-blocking: skip if another process is cleaning) ---
    CLEANUP_LOCK_PATH.touch(exist_ok=True)
    lock_fd = open(CLEANUP_LOCK_PATH)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Another process holds the lock — skip this cycle
        lock_fd.close()
        if verbose:
            log("Skipping stale worktree cleanup (another process holds the lock)")
        return 0

    try:
        return _cleanup_stale_worktrees_locked(
            config, base, verbose=verbose, min_age=min_age, force=force,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _cleanup_stale_worktrees_locked(
    config: dict, base: Path, *, verbose: bool, min_age: float, force: bool,
) -> int:
    """Inner cleanup implementation — caller must hold CLEANUP_LOCK_PATH."""
    import concurrent.futures

    # Collect worktree dirs owned by live agents
    live_worktrees: set[str] = set()
    for agent in read_all_agents():
        if agent.worktree and agent.status != "dead":
            live_worktrees.add(agent.worktree)

    now = time.time()
    stale: list[Path] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if str(entry) in live_worktrees:
            continue
        # Age threshold: skip recently-created worktrees unless force
        if not force:
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age < min_age:
                if verbose:
                    log(f"Skipping young worktree {entry.name} (age {age:.0f}s < {min_age:.0f}s)")
                continue
        # Ownership check: only delete pod-owned worktrees
        if not _is_pod_owned_worktree(entry):
            if verbose:
                log(f"Skipping non-pod worktree {entry.name} (no agent/* branch)")
            continue
        stale.append(entry)

    if not stale:
        return 0

    if verbose:
        log(f"Cleaning {len(stale)} stale worktree(s)...")

    removed = 0
    failed: list[str] = []

    def _remove_one(entry: Path) -> bool:
        """Remove one worktree. Returns True on success."""
        # Only unlock if this is a pod-owned worktree (agent/* branch exists)
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "worktree", "unlock",
             str(entry)],
            capture_output=True, timeout=10,
        )
        # Worktrees may contain large build outputs (e.g. Lake .lake dirs
        # with 100k+ files) that take well over 2 minutes to remove on
        # spinning disks or busy SSDs.  600s gives realistic headroom.
        r = subprocess.run(
            ["rm", "-rf", str(entry)],
            capture_output=True, timeout=600,
        )
        return r.returncode == 0 and not entry.exists()

    # Parallel rm -rf — each is I/O-bound, so threads work well
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_remove_one, e): e for e in stale}
        for fut in concurrent.futures.as_completed(futures):
            entry = futures[fut]
            try:
                if fut.result():
                    removed += 1
                else:
                    failed.append(entry.name)
                    log(f"Failed to remove worktree {entry.name}")
            except Exception as exc:
                failed.append(entry.name)
                log(f"Exception removing worktree {entry.name}: {exc}")

    # Prune metadata, then delete branches (can't delete a branch
    # that git still thinks is checked out in a worktree)
    try:
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "worktree", "prune"],
            capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        log("git worktree prune timed out")

    for entry in stale:
        if entry.name not in failed:
            subprocess.run(
                ["git", "branch", "-D", f"agent/{entry.name}"],
                capture_output=True, timeout=10, cwd=str(PROJECT_DIR),
            )

    if verbose:
        msg = f"Cleaned {removed} stale worktree(s)"
        if failed:
            msg += f" ({len(failed)} failed: {', '.join(failed)})"
        log(msg)

    return removed


# Minimum age before an orphan agent/* branch is eligible for cleanup.
# Protects against racing with a setup_worktree() call that has just
# created the branch but hasn't finished checking out the worktree.
STALE_BRANCH_MIN_AGE_SECONDS = 300


def cleanup_stale_branches(config: dict, *, verbose: bool = False,
                           min_age: float = STALE_BRANCH_MIN_AGE_SECONDS) -> int:
    """Delete ``agent/*`` branches with no worktree and no live owner.

    Why: ``git worktree add -b`` is non-atomic — it creates the branch
    ref before the worktree dir is fully checked out.  When that second
    step fails (disk full, lock contention, signal), the branch persists
    and becomes a permanent leftover.  Over time this accumulates
    hundreds of dead refs that bloat ``.git/refs/heads/agent/`` and can
    collide with future session UUIDs.

    Safety:
      - Only deletes branches whose worktree dir does **not** exist.
      - Skips branches referenced by any live agent state file.
      - Skips branches younger than ``min_age`` seconds (default 5 min)
        to avoid racing with an in-flight ``setup_worktree`` call.
    """
    base = PROJECT_DIR / cfg_get(config, "project", "worktree_base", default="worktrees")
    refs_dir = PROJECT_DIR / ".git" / "refs" / "heads" / "agent"
    if not refs_dir.is_dir():
        return 0

    # Branches referenced by live (non-dead) agents — never delete these
    live_branches = _live_branches()

    now = time.time()
    deleted = 0
    skipped_young = 0
    skipped_live = 0
    skipped_has_worktree = 0
    for ref in sorted(refs_dir.iterdir()):
        if not ref.is_file():
            continue
        short_id = ref.name
        branch = f"agent/{short_id}"
        if branch in live_branches:
            skipped_live += 1
            continue
        if (base / short_id).exists():
            skipped_has_worktree += 1
            continue
        try:
            age = now - ref.stat().st_mtime
        except OSError:
            continue
        if age < min_age:
            skipped_young += 1
            continue
        r = subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True, text=True, timeout=10, cwd=str(PROJECT_DIR),
        )
        if r.returncode == 0:
            deleted += 1
        elif verbose:
            log(f"cleanup_stale_branches: failed to delete {branch}: {r.stderr.strip()}")

    if verbose and (deleted or skipped_young or skipped_live or skipped_has_worktree):
        log(f"cleanup_stale_branches: deleted={deleted} "
            f"skipped(young={skipped_young}, live={skipped_live}, "
            f"has_worktree={skipped_has_worktree})")
    return deleted


def _distinct_mounts(*paths: Path) -> list[Path]:
    """Return one representative path per distinct filesystem device.

    Walks each input path upward until it finds an existing ancestor
    (``shutil.disk_usage`` rejects non-existent paths), then dedupes by
    ``st_dev``.  Used by ``_disk_low`` so we check every volume the
    project writes to, not just ``PROJECT_DIR``.
    """
    seen_devs: set[int] = set()
    out: list[Path] = []
    for p in paths:
        # Walk upward until something exists — covers configured paths
        # whose leaf hasn't been created yet.
        probe: Path = p
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if not probe.exists():
            continue
        try:
            dev = probe.stat().st_dev
        except OSError:
            continue
        if dev in seen_devs:
            continue
        seen_devs.add(dev)
        out.append(probe)
    return out


def _disk_low(config: dict) -> str | None:
    """Return a human-readable reason if free disk is below the configured
    threshold on any volume the project writes to, else None.

    Checks every distinct filesystem hosting ``PROJECT_DIR``,
    ``project.worktree_base``, or ``project.session_dir`` — so an
    absolute ``session_dir`` on a separate volume is covered, not just
    the project filesystem.

    Used to pause dispatch before a disk-full event corrupts worktree
    state (the canonical failure mode: ``git worktree add`` half-
    completes, leaving an orphan branch that blocks future dispatches).
    """
    try:
        min_gb = float(cfg_get(config, "project", "min_free_disk_gb", default=2))
    except (TypeError, ValueError):
        min_gb = 2.0
    if min_gb <= 0:
        return None

    worktree_base = PROJECT_DIR / cfg_get(config, "project", "worktree_base", default="worktrees")
    session_dir = PROJECT_DIR / cfg_get(config, "project", "session_dir", default="sessions")
    targets = _distinct_mounts(PROJECT_DIR, worktree_base, session_dir)
    if not targets:
        targets = [PROJECT_DIR]

    for probe in targets:
        try:
            usage = shutil.disk_usage(str(probe))
        except OSError as e:
            return f"disk_usage({probe}) failed: {e}"
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_gb:
            return f"{free_gb:.2f} GiB free at {probe} < {min_gb:.2f} GiB threshold"
    return None


def _log_credential_state(session_uuid: str,
                            claude_config_dir: Path | None = None,
                            expected_label: str = ""):
    """Log which credential Claude Code will actually use at agent launch time.

    Pure logging — does *not* mutate the keychain. Account selection
    happens earlier in the iteration (under the lease meta-lock), and
    ``accounts.mirror_canonical_to_isolated`` writes the chosen
    account's blob into the per-agent keychain entry. By the time
    ``launch_agent`` calls us, the right credential is already there;
    we just confirm and warn if it doesn't match ``expected_label``.
    """
    parts = [f"session={session_uuid[:8]}"]
    resolved = accounts.resolve_claude_credential(claude_config_dir)
    parts.append(
        f"resolved=[{resolved['accountLabel']}] "
        f"token={resolved['tokenPrefix']}... via {resolved['source']}"
    )
    if expected_label and resolved["accountLabel"] != expected_label:
        parts.append(
            f"WARNING expected [{expected_label}] but isolated "
            f"keychain has [{resolved['accountLabel']}]"
        )
    log(f"Credential state at launch: {' | '.join(parts)}")


_codex_command_resolved: str | None = None


def _resolve_codex_command() -> str:
    """Return a codex command/path that runs successfully on this host.

    Memoised — the actual probe runs at most once per pod process.

    On macOS, AppleSystemPolicy can permanently deny a specific Mach-O
    *path* after a failed Gatekeeper first-launch policy evaluation. Once
    that happens, every subsequent spawn from that path is held in the
    kernel `launched-suspended` state forever (96 KB resident, 0% CPU,
    no syscalls, no stdout/stderr) — externally indistinguishable from a
    deeply hung codex. The denial is keyed on path, not file content:
    the same bytes copied to a different path run fine. Common trigger:
    `brew install codex` deposits the binary at
    `/opt/homebrew/Caskroom/codex/<v>/codex-aarch64-apple-darwin` with
    `com.apple.quarantine` + `com.apple.provenance` xattrs; on first
    launch syspolicyd shows a confirmation dialog to a GUI session that
    isn't there (pod is over SSH), the response never arrives, and the
    path enters a denied state.

    Algorithm:
    1. Probe `codex --version` with a 5 s timeout. Clean exit → return
       the bare name `"codex"` (PATH lookup, normal behaviour).
    2. On timeout, confirm via `vmmap` that the spawn is in
       `launched-suspended` state. Real hangs are *not* touched.
    3. Copy the binary content (preserving its original notarized
       signature — do NOT ad-hoc resign, AMFI rejects ad-hoc Mach-Os) to
       a stable per-user cache path that ASP hasn't blocked, clear
       xattrs on the copy, and verify it runs.
    4. Return the absolute path to the copy. Callers (i.e.
       `launch_agent`) invoke codex by that path so the spawn never
       transits the poisoned location.

    The user-facing `/opt/homebrew/bin/codex` symlink is left alone —
    homebrew owns it and would clobber any rewrite on next upgrade. We
    keep our remediation pod-local.

    Why no ad-hoc resign: a previous version of this helper resigned the
    rotated binary with `codesign --force -s -`. That broke on Sonoma /
    Sequoia where AMFI denies ad-hoc-signed binaries with
    `-423 "The file is adhoc signed or signed by an unknown certificate
    chain"`. Keeping the original Developer ID + hardened-runtime
    signature works fine; ASP only objected to the *path*, not the
    signature.

    Freshness: if the cached copy already exists from a previous pod
    run, we still re-probe (the probe is sub-second when healthy and
    catches a stale cache where the user `brew upgrade`d codex
    underneath us — `which codex`'s realpath would now have a different
    mtime/size than the cache copy). On detected drift we re-copy.

    No-op (returns `"codex"`) on non-Darwin platforms.
    """
    global _codex_command_resolved
    if _codex_command_resolved is not None:
        return _codex_command_resolved

    def _decide() -> str:
        if sys.platform != "darwin":
            return "codex"

        codex = shutil.which("codex")
        if not codex:
            return "codex"

        # Probe via PATH first.
        if _codex_probe_ok(codex):
            return "codex"

        # Determine if we're seeing the launched-suspended path-denial.
        if not _codex_probe_launched_suspended(codex):
            log("codex --version is unresponsive but vmmap does not show "
                "launched-suspended; this is some other hang. Not "
                "remediating.")
            return "codex"

        real = os.path.realpath(codex)
        log(f"codex at {real} is held in macOS launched-suspended state "
            f"(AppleSystemPolicy path denial after a Gatekeeper "
            f"first-launch prompt was queued without a GUI to answer "
            f"it). Copying the binary to a per-user safe path and "
            f"invoking codex from there.")

        cache_dir = Path.home() / ".cache" / "pod" / "codex"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_bin = cache_dir / "codex"

        # Refresh the cache if missing or stale relative to upstream.
        try:
            need_copy = (
                not safe_bin.exists()
                or safe_bin.stat().st_size != os.stat(real).st_size
                or safe_bin.stat().st_mtime < os.stat(real).st_mtime
            )
            if need_copy:
                tmp = safe_bin.with_name(safe_bin.name + ".tmp")
                shutil.copyfile(real, tmp)
                shutil.copymode(real, tmp)
                subprocess.run(["xattr", "-c", str(tmp)], check=False)
                os.replace(tmp, safe_bin)
        except Exception as e:
            log(f"codex cache refresh failed: {e!r}")
            return "codex"

        if _codex_probe_ok(str(safe_bin)):
            log(f"codex remediation succeeded; using {safe_bin}.")
            return str(safe_bin)

        log(f"codex copy at {safe_bin} also fails to run; not "
            f"remediating. Manual fix likely required (e.g. log in to "
            f"the Mac via GUI and approve the binary in Terminal once, "
            f"or `sudo spctl --add` it).")
        return "codex"

    _codex_command_resolved = _decide()
    return _codex_command_resolved


def _codex_probe_ok(cmd: str) -> bool:
    """Return True iff `<cmd> --version` exits 0 within 5 s."""
    try:
        subprocess.run(
            [cmd, "--version"],
            timeout=5,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        # Non-timeout failure (missing binary, dyld error, etc.) is not
        # the launched-suspended bug — caller should propagate.
        return False


def _codex_probe_launched_suspended(cmd: str) -> bool:
    """Spawn `<cmd> --version` and check vmmap for launched-suspended."""
    proc = subprocess.Popen(
        [cmd, "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1)  # Let the process reach the suspended state.
        try:
            vm = subprocess.run(
                ["vmmap", str(proc.pid)],
                timeout=5,
                capture_output=True,
                text=True,
            ).stdout
        except Exception:
            return False
        return "launched-suspended" in vm
    finally:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def launch_agent(config: dict, session_uuid: str, prompt: str,
                   wt_dir: str,
                   claude_config_dir: Path | None = None,
                   backend: str | None = None,
                   model: str | None = None,
                   account_label: str = "") -> subprocess.Popen:
    """Launch agent subprocess (Claude or Codex) in the worktree directory.

    ``backend`` is required when running in auto mode; for fixed-backend
    modes it defaults to the configured value.

    ``model`` should be passed by callers that have already selected a
    tier via ``acquire_backend`` (which honours ``accepted_models``).
    Falls back to re-reading ``[agent.<backend>].model`` from disk only
    when omitted, preserving the pre-leasing call contract.

    ``account_label`` is used purely for the credential-state log line
    — it warns when the isolated keychain entry's label doesn't match
    the selected one.
    """
    if backend is None:
        backend = _reload_config_value("agent", "backend", default="claude")
        if backend == "auto":
            raise ValueError(
                "launch_agent: explicit backend= required in auto mode")
    if model is None:
        # Legacy path: no caller-supplied selection. Re-read default.
        model = _reload_config_value(
            "agent", backend, "model", default="opus")
    session_dir = PROJECT_DIR / cfg_get(config, "project", "session_dir", default="sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = session_dir / f"{session_uuid}.stdout"

    env = dict(os.environ)
    env["POD_SESSION_ID"] = session_uuid
    # Inject bundled data dir into PATH so agents find `coordination`
    env["PATH"] = str(_data_dir()) + os.pathsep + env.get("PATH", "")

    # Surface dispatcher state so the planner can size its output to the
    # actual deficit (min_queue - queue_depth). Mirrors the merge logic in
    # dispatch_queue_balance: planner advisory shrinks but never grows the
    # configured min_queue.
    try:
        _qd = get_queue_depth(config)
        _cfg_mq = max(0, cfg_get(config, "dispatch", "min_queue", default=3))
        _adv_mq = read_planner_min_queue()
        _eff_mq = max(0, min(_cfg_mq, _adv_mq)) if _adv_mq is not None else _cfg_mq
        env["POD_QUEUE_DEPTH"] = str(_qd)
        env["POD_MIN_QUEUE"] = str(_eff_mq)
        env["POD_QUEUE_DEFICIT"] = str(max(0, _eff_mq - _qd))
    except Exception as e:
        log(f"launch_agent: failed to read queue state for env injection: {e}")

    stdin_pipe = None  # Only used for Codex (stdin prompt delivery)

    if backend == "codex":
        # --- Codex launch ---
        # Resolve the codex command/path. Returns "codex" on healthy hosts
        # (ordinary PATH lookup) and a per-user safe-path on macOS hosts
        # where the homebrew-installed binary is held in
        # launched-suspended state by AppleSystemPolicy's path-based
        # denial cache.
        codex_cmd = _resolve_codex_command()

        # Pass model via -c (config override) rather than -m. The -m flag
        # routes through codex's API-account validation path, which rejects
        # GPT-5.x slugs ("not supported when using Codex with a ChatGPT
        # account") even when the underlying ChatGPT-account session does
        # support them. -c overlays onto config.toml without that check, and
        # works for both auth modes.
        codex_args = [codex_cmd, "exec", "--json",
                      "--dangerously-bypass-approvals-and-sandbox",
                      "--skip-git-repo-check",
                      "-c", f"model={model}",
                      "-C", wt_dir,
                      "-"]  # Read prompt from stdin
        env["OPENAI_API_KEY"] = ""  # Force subscription auth
        env["POD_IS_RESUME"] = "0"

        # Set up isolated CODEX_HOME if configured
        codex_home = _setup_codex_home(config, wt_dir)
        if codex_home:
            env["CODEX_HOME"] = str(codex_home)
            log(f"Using strict pod-managed CODEX_HOME={codex_home}")

        # Write a sidecar manifest so the historical-cost scanner knows
        # which pricing table to use for rollouts of this session
        # (rollout JSONLs do not carry a stable model-name field).
        _write_codex_manifest(session_uuid, backend, model, wt_dir)

        stdin_pipe = subprocess.PIPE
        agent_args = codex_args
    else:
        # --- Claude launch (existing behavior) ---
        jsonl_dir = _claude_projects_dir(claude_config_dir) / wt_dir.replace("/", "-")
        local_jsonl = jsonl_dir / f"{session_uuid}.jsonl"

        claude_args = ["claude", "--model", model]
        if local_jsonl.exists():
            claude_args += ["--resume", session_uuid]
        else:
            claude_args += ["--session-id", session_uuid]
        if claude_config_dir is not None or _use_bubble(config):
            claude_args += ["--dangerously-skip-permissions"]
        claude_args += ["-p", prompt]

        env["ANTHROPIC_API_KEY"] = ""  # Force subscription auth
        env["POD_IS_RESUME"] = "1" if local_jsonl.exists() else "0"
        if claude_config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = str(claude_config_dir)

        # Log credential state at launch for debugging account-swap issues
        _log_credential_state(session_uuid, claude_config_dir,
                              expected_label=account_label)

        agent_args = claude_args

    # --- Optional: wrap in `bubble` for filesystem isolation ---
    if _use_bubble(config):
        bubble_name = _bubble_name(session_uuid)
        in_repo = _bubble_in_container_repo(wt_dir)
        # Inner command to run inside the bubble. Quote agent_args
        # individually, and wrap in `bash -lc` so /etc/profile.d (which
        # exports GH_TOKEN for the auth proxy) is sourced. Empty
        # ANTHROPIC_API_KEY/OPENAI_API_KEY/CLAUDECODE force subscription
        # auth via the mounted .credentials.json. /opt/pod-data prepended
        # to PATH so `coordination` is found.
        inner_cmd = " ".join(shlex.quote(a) for a in agent_args)
        bash_inner = (
            f"cd {shlex.quote(in_repo)} && "
            f"ANTHROPIC_API_KEY= OPENAI_API_KEY= CLAUDECODE= "
            f"PATH=/opt/pod-data:$PATH exec {inner_cmd}"
        )
        bash_lc = "bash -lc " + shlex.quote(bash_inner)
        # Outer single-quote so bubble's shlex.split() preserves bash_lc
        # as one token (works around bubble's host-side argv splitting).
        bubble_command = shlex.quote(bash_lc)
        agent_args = [
            "bubble", "open",
            "--path", wt_dir,
            "--shell",
            "--no-claude-config",
            "--claude-credentials",
            "--no-codex-credentials",
            # Pod's coordination script uses `gh issue list --label X`, which
            # gh implements as a multi-top-level GraphQL search() that the
            # default `allowlist-write-graphql` proxy correctly rejects.
            # `write-graphql` widens GraphQL to account-wide for this bubble
            # only; REST stays repo-scoped. Per-launch override leaves the
            # user's other bubbles on the safer default.
            "--github-security", "write-graphql",
            "--name", bubble_name,
            "--mount", f"{_data_dir()}:/opt/pod-data:ro",
            "--command", bubble_command,
        ]
        # Drop CLAUDE_CONFIG_DIR — agent runs inside bubble, host
        # isolated_config doesn't apply.
        env.pop("CLAUDE_CONFIG_DIR", None)
        # Stash the bubble name globally so the run-loop and the
        # SIGTERM handler can pop the container after exit.
        global _agent_bubble_name
        _agent_bubble_name = bubble_name
        log(f"Agent {session_uuid[:8]}: launching inside bubble {bubble_name}")

    stdout_fd = os.open(str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    proc = subprocess.Popen(
        agent_args,
        stdin=stdin_pipe,
        stdout=stdout_fd,
        stderr=subprocess.STDOUT,
        cwd=wt_dir,
        env=env,
        start_new_session=True,  # Create process group for clean killing
    )
    os.close(stdout_fd)  # Child inherited it; parent no longer needs it

    # For Codex, deliver prompt via stdin and close
    if backend == "codex" and proc.stdin is not None:
        try:
            proc.stdin.write(prompt.encode())
            proc.stdin.close()
        except OSError:
            pass  # Process may have exited immediately

    return proc


def _codex_rollouts_dir() -> Path:
    """Project-level directory where all Codex rollouts land, via CODEX_HOME symlink."""
    return PROJECT_DIR / ".pod" / "codex-sessions"


def _codex_manifest_dir() -> Path:
    """Directory for per-session sidecar manifests (backend, model, wt_dir, started_at)."""
    return _codex_rollouts_dir() / "manifests"


def _write_codex_manifest(session_uuid: str, backend: str, model: str, wt_dir: str) -> None:
    """Persist a tiny sidecar so the historical scanner can price rollouts.

    The rollout JSONL does not carry a stable model-name field; we write
    `<project>/.pod/codex-sessions/manifests/<uuid>.json` with the info
    known at launch time.
    """
    try:
        mdir = _codex_manifest_dir()
        mdir.mkdir(parents=True, exist_ok=True)
        mpath = mdir / f"{session_uuid}.json"
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "session_id": session_uuid,
            "backend": backend,
            "model": model,
            "wt_dir": wt_dir,
            "started_at": time.time(),
        }) + "\n")
        tmp.replace(mpath)
    except OSError as e:
        log(f"Warning: failed to write Codex manifest for {session_uuid}: {e}")


def _setup_codex_home(config: dict, wt_dir: str) -> Path | None:
    """Create an isolated CODEX_HOME directory for a Codex agent.

    Sets up a strict pod-managed home: bundled skills, a minimal pod-owned
    config.toml, an auth symlink to ~/.codex/auth.json for login reuse, and
    a `sessions` symlink to `<project>/.pod/codex-sessions/` so Codex rollout
    JSONLs land in the project (scannable by historical cost, shared across
    worktrees) rather than being lost when the home is rebuilt.

    Refreshes only pod-owned paths in place rather than wiping the whole
    home. Wiping would race with any concurrent launch and would also blow
    away Codex-managed state (cache, logs, sqlite dbs) that has no business
    being pod-owned. Pod-owned paths: `auth.json`, `config.toml`, `skills/`,
    `sessions` (symlink). Everything else is Codex's to manage.

    Returns the CODEX_HOME path, or None if isolation is disabled.
    """
    if not _backend_cfg(config, "isolated_config", backend="codex", default=True):
        return None

    codex_home = Path(wt_dir) / ".pod-codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)

    def _replace_symlink(link: Path, target: Path) -> None:
        """Atomically point `link` at `target` regardless of prior state."""
        tmp = link.with_name(link.name + ".tmp")
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        tmp.symlink_to(target)
        os.replace(tmp, link)  # atomic on POSIX

    # Ensure the rollout dir lives in the project, not in the worktree.
    shared_sessions = PROJECT_DIR / ".pod" / "codex-sessions"
    shared_sessions.mkdir(parents=True, exist_ok=True)
    _replace_symlink(codex_home / "sessions", shared_sessions)

    # Symlink auth from user's global Codex config
    real_codex = Path.home() / ".codex"
    auth_target = real_codex / "auth.json"
    if auth_target.exists():
        _replace_symlink(codex_home / "auth.json", auth_target)

    # Write pod-owned minimal config for non-interactive execution.
    model = _reload_config_value("agent", "codex", "model", default="gpt-5.4")
    (codex_home / "config.toml").write_text(
        f'model = "{model}"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "danger-full-access"\n'
    )

    # Install bundled skills (refresh to match current package).
    data_skills = _data_dir() / "agent-config" / "codex" / "skills"
    if data_skills.is_dir():
        dst_skills = codex_home / "skills"
        if dst_skills.exists():
            shutil.rmtree(dst_skills)
        shutil.copytree(str(data_skills), str(dst_skills))

    return codex_home


def get_jsonl_path(wt_dir: str, session_uuid: str,
                   claude_config_dir: Path | None = None,
                   backend: str = "claude",
                   config: dict | None = None) -> str:
    """Compute JSONL file path for a session.

    For Claude: reads from ~/.claude/projects/{wt_dir}/{uuid}.jsonl,
    or — when running in bubble mode — from
    ~/.bubble/ai-projects/<bubble-name>/<encoded-in-bubble-cwd>/{uuid}.jsonl.
    For Codex: reads from sessions/{uuid}.stdout (the --json stream).
    """
    if backend == "codex":
        session_dir = PROJECT_DIR / "sessions"
        return str(session_dir / f"{session_uuid}.stdout")
    if config is not None and _use_bubble(config):
        bubble_name = _bubble_name(session_uuid)
        in_repo = _bubble_in_container_repo(wt_dir)
        return str(_bubble_jsonl_dir(bubble_name, in_repo) / f"{session_uuid}.jsonl")
    jsonl_dir = _claude_projects_dir(claude_config_dir) / wt_dir.replace("/", "-")
    return str(jsonl_dir / f"{session_uuid}.jsonl")


# ---------------------------------------------------------------------------
# Agent Process (forked background process)
# ---------------------------------------------------------------------------

# Globals for signal handlers
_agent_state: AgentState | None = None
_agent_proc: subprocess.Popen | None = None
_agent_config: dict = {}
_agent_bubble_name: str | None = None  # Set when running in bubble mode


def _pop_agent_bubble() -> None:
    """Pop the bubble container associated with the current agent, if any.

    Idempotent: clears `_agent_bubble_name` so repeated calls (e.g. from
    both the SIGTERM handler and the normal-exit path) do nothing.
    """
    global _agent_bubble_name
    name = _agent_bubble_name
    if not name:
        return
    _agent_bubble_name = None
    try:
        subprocess.run(
            ["bubble", "pop", "--force", name],
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"Warning: failed to pop bubble {name}: {e}")


def _release_agent_resources(reason: str, skip_msg: str) -> None:
    """Tear down this agent's resources before exit.

    Kills any child claude/codex subprocess, unclaims the agent's GitHub
    issue (if any), releases its dispatch lock, cleans up its worktree,
    and removes its state file.

    Used both by the SIGTERM path and the unhandled-exception path in
    `spawn_agent`. Without this on the crash path, an unhandled exception
    in `agent_process_main` (e.g. a stale `_data_dir()` path after an
    editable-install switcheroo, a transient GitHub failure, an OOM in a
    helper) would leave the GitHub `claimed` label and the
    `claim-history.json` entry orphaned: the issue would sit unclaimable
    until the next housekeeping cycle's dead-claim sweep, and the
    operator would just see "agent disappeared from the TUI" with no
    indication of why.

    `reason` becomes the agent's `status` ("killed", "crashed", …) and
    is excluded from the "another live agent has this claim" check so
    the unclaim path is taken correctly.

    `skip_msg` is the operator-visible reason recorded by
    `coordination skip`. Callers should embed enough detail (session
    UUID, exception text) to identify the failure post-hoc.

    Idempotent. Safe to call from a signal handler or from a
    `try/except` in the grandchild process.
    """
    global _agent_state, _agent_proc, _agent_config
    state = _agent_state
    config = _agent_config

    if state:
        state.status = reason
        try:
            state.write()
        except OSError:
            pass

    # Release the Claude account lease + harvest any refreshed token
    # before tearing down the rest of the agent. Done early so a
    # blocked downstream cleanup doesn't leave the lease orphaned.
    if state and state.account_label:
        try:
            claude_cfg = (
                Path(state.claude_config_dir) if state.claude_config_dir else None)
            _release_account_lease(state, claude_cfg)
            try:
                state.write()
            except OSError:
                pass
        except Exception as e:
            log(f"_release_agent_resources: lease release failed: {e}")

    # Kill the child agent subprocess (claude/codex), if still running.
    if _agent_proc and _agent_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_agent_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            _agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_agent_proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    # Pop bubble container (no-op when not in bubble mode)
    _pop_agent_bubble()

    if state and config:
        # Unclaim issue if claimed and no PR yet — but only if no other
        # live agent is also working on this issue (prevents unclaiming
        # when killing duplicate claimants).
        # Skip for `repair`/`replan` workers: they do not own a claimed
        # issue, so `claimed_issue` is meaningless for them (if it
        # somehow holds a stale number, calling `coordination skip` on
        # it just thrashes on closed issues).
        if (state.claimed_issue > 0 and state.pr_number == 0
                and state.worker_type not in ("repair", "replan")):
            try:
                other_agents = read_all_agents()
            except Exception:
                other_agents = []
            other_live = any(
                a.claimed_issue == state.claimed_issue
                and a.uuid != state.uuid
                and a.status not in ("dead", "stopped", "killed", "crashed")
                for a in other_agents
            )
            if other_live:
                log(f"Agent {state.short_id}: not unclaiming #{state.claimed_issue} — another agent has it")
                try:
                    clear_claim(state.claimed_issue, session_uuid=state.uuid)
                except Exception:
                    pass
            else:
                try:
                    coordination(
                        config, "skip", str(state.claimed_issue),
                        skip_msg,
                        env_extra={"POD_SESSION_ID": state.uuid},
                    )
                    clear_claim(state.claimed_issue, session_uuid=state.uuid)
                    log(f"Unclaimed issue #{state.claimed_issue}")
                except Exception as e:
                    # Don't clear_claim — leave in history so housekeeping
                    # can retry the GitHub label removal later.
                    log(f"Failed to skip #{state.claimed_issue} on {reason} ({e}), keeping in history")

        # Release lock if held
        if state.lock_held:
            try:
                coordination(config, f"unlock-{state.lock_held}")
                log(f"Released {state.lock_held} lock")
            except Exception:
                pass

        # Cleanup worktree
        if state.worktree and state.branch:
            try:
                cleanup_worktree(state.worktree, state.branch)
            except Exception as e:
                log(f"Worktree cleanup failed on {reason}: {e}")

        # Cleanup per-agent CLAUDE_CONFIG_DIR (best-effort; keychain
        # entries linger but are orphaned and harmless).
        if state.claude_config_dir:
            try:
                accounts.cleanup_agent_claude_config_dir(
                    POD_DIR, state.short_id)
            except OSError as e:
                log(f"claude-config cleanup failed on {reason}: {e}")

        try:
            state.remove_file()
        except OSError:
            pass


def _sigterm_handler(signum, frame):
    """Handle SIGTERM: kill claude, unclaim, cleanup, exit."""
    state = _agent_state
    if state:
        log(f"Agent {state.short_id} received SIGTERM")
        skip_msg = f"Agent killed by operator (session {state.uuid})"
    else:
        skip_msg = "Agent killed by operator"
    _release_agent_resources("killed", skip_msg)
    os._exit(0)


def _sigusr1_handler(signum, frame):
    """Handle SIGUSR1: set finishing flag."""
    global _agent_state
    if _agent_state:
        _agent_state.finishing = True
        _agent_state.status = "finishing"
        _agent_state.write()
        log(f"Agent {_agent_state.short_id} marked as finishing")


def _resume_jitter(config: dict) -> float:
    """Random jitter (seconds) added when a paused agent waits for its resume
    slot, so agents that wake together scatter rather than stampede."""
    return float(cfg_get(config, "quota", "resume_jitter_secs", default=10))


def _resume_slot_wait(config: dict) -> float:
    """Global rate-limiter for quota-resume launches across all agents.

    Returns ``0.0`` and reserves the next slot if the caller may resume now;
    otherwise returns the seconds until the next slot frees (caller should
    sleep, then retry — which also re-checks quota). Serializes resumes to one
    per ``resume_gap_secs`` so a freshly-rotated account is not stampeded by
    every paused agent at once (in shared mode there are no per-agent leases to
    space them out).

    A *separate* lock file guards a plain-timestamp data file. We must not flock
    the data file itself and then rewrite it: ``_flock`` locks the inode, and a
    rewrite/rename would swap the inode out from under other waiters.
    """
    gap = float(cfg_get(config, "quota", "resume_gap_secs", default=30))
    lockpath = POD_DIR / "quota-resume.lock"
    nextpath = POD_DIR / "quota-resume-next"
    with accounts._flock(lockpath, exclusive=True, blocking=True):
        now = time.time()
        try:
            nxt = float(nextpath.read_text().strip())
        except (OSError, ValueError):
            nxt = 0.0
        if now >= nxt:
            nextpath.write_text(str(now + gap))
            return 0.0
        return nxt - now


def _release_paused_session(state: "AgentState", config: dict,
                            claude_config_dir: Path | None, *,
                            reason: str) -> None:
    """Give up on a paused/resuming agent and release everything it was holding:
    its GitHub claim, its preserved worktree, and the account lease; then clear
    the pause flags. Used when a resume cannot proceed (pause cap exceeded,
    resume launch failed, graceful finish mid-pause). Mirrors the quota
    fall-through cleanup so a held claim/worktree is never leaked."""
    if (state.claimed_issue > 0 and state.pr_number == 0
            and state.worker_type not in ("repair", "replan")):
        try:
            if _release_claim(str(state.claimed_issue), state.uuid, 0,
                              reason=f"Claim released — {reason}."):
                clear_claim(state.claimed_issue, session_uuid=state.uuid)
        except _GraphQLRateLimited:
            pass
    if state.worktree:
        try:
            cleanup_worktree(state.worktree, state.branch)
        except Exception:
            pass
        state.worktree = ""
        state.branch = ""
    _release_account_lease(state, claude_config_dir)
    state.resuming_after_pause = False
    state.quota_paused_at = 0.0


def agent_process_main(config: dict, agent_id: str | None = None,
                        resume_uuid: str | None = None,
                        target_issue: int = 0,
                        target_type: str = ""):
    """Entry point for a forked agent process. Runs the agent loop.

    When `target_issue` is set the agent runs in one-shot mode: skip
    dispatch_queue_balance, skip the dispatch quota cap (the caller must
    set the corresponding `force_quota` desire — see `cmd_once`), point
    the prompt at the specific issue, and exit after one iteration.
    """
    global _agent_state, _agent_proc, _agent_config
    _agent_config = config

    short_id = agent_id or uuid.uuid4().hex[:8]

    my_pid = os.getpid()
    raw_backend = _backend(config)
    # In auto mode, leave backend/model blank; both are filled per iteration.
    initial_backend = "" if raw_backend == "auto" else raw_backend
    initial_model = "" if raw_backend == "auto" else (
        _backend_cfg(config, "model", default="") or "")
    state = AgentState(
        short_id=short_id,
        pid=my_pid,
        pid_start_time=_get_pid_start_time(my_pid),
        status="starting",
        resume_session_uuid=resume_uuid or "",
        backend=initial_backend,
        model=initial_model,
        target_issue=target_issue,
        target_type=target_type,
        force_quota=bool(target_issue),
    )
    _agent_state = state
    state.write()

    # Install signal handlers
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    poll_interval = cfg_get(config, "monitor", "poll_interval", default=2)
    # In auto mode, the [agent] level retry applies before a backend is
    # chosen. Fixed mode falls back to the per-backend value.
    if raw_backend == "auto":
        quota_retry = cfg_get(config, "agent", "quota_retry_seconds", default=60)
    else:
        quota_retry = _backend_cfg(config, "quota_retry_seconds", default=60)
    worker_types = cfg_get(config, "worker_types", default={})

    claude_config_dir = ensure_isolated_config(config, short_id)
    if claude_config_dir:
        log(f"Agent {short_id}: claude isolated config at {claude_config_dir}"
            + (" (auto mode — claude iterations only)" if raw_backend == "auto" else ""))
        state.claude_config_dir = str(claude_config_dir)
    if raw_backend in ("codex", "auto"):
        log(f"Agent {short_id}: codex iterations use CODEX_HOME set per session")
    if raw_backend == "auto":
        order = _auto_backend_order(config)
        log(f"Agent {short_id}: auto backend selection enabled, prefer={order[0]}")

    log(f"Agent {short_id} started (PID {os.getpid()})")

    iteration = 0
    consecutive_wait = 0  # Consecutive dispatch-returned-None iterations (for backoff)
    HOUSEKEEPING_INTERVAL = 600  # seconds between housekeeping runs (check-blocked, check-has-pr, dead claims)
    while not state.finishing:
        iteration += 1
        state.loop_iteration = iteration

        # --- Backend / account selection + quota check ---
        # Resume sessions are pinned to the backend AND Claude account
        # they originally ran on: only Claude has a real `--resume`,
        # and a refreshed token on the wrong account would silently
        # fork the conversation or fail.
        # Pin the backend (only Claude has a real `--resume`) for both kinds of
        # resume. Pin the *account* only for a spawn-time resume; a quota-pause
        # resume deliberately leaves the account unpinned (account_label was
        # cleared on pause) so it follows swap-account to a fresh, in-quota
        # account — verified that `claude --resume` continues across accounts.
        resume_pin_backend = (
            state.backend
            if (state.resume_session_uuid or state.resuming_after_pause)
            and state.backend
            else None)
        resume_pin_label = (
            state.account_label
            if state.resume_session_uuid and state.account_label
            else None)
        state.status = "waiting_quota"
        state.write()
        selection: accounts.Candidate | None = None
        while True:
            selection = acquire_backend(
                config,
                state=state,
                claude_config_dir=claude_config_dir,
                force=state.force_quota,
                pin_backend=resume_pin_backend,
                pin_label=resume_pin_label,
            )
            if selection is not None:
                # Stagger quota-resume launches: only one agent resumes per
                # `resume_gap_secs`, so a freshly-rotated account is not
                # stampeded by every paused agent at once. If our slot is not
                # ready, sleep and re-loop (which re-checks quota too) rather
                # than launching onto a soon-to-be-re-exhausted account.
                if state.resuming_after_pause and not state.finishing:
                    _gap = _resume_slot_wait(config)
                    if _gap > 0.0:
                        _wait = min(_gap, 60.0) + random.uniform(
                            0.0, _resume_jitter(config))
                        log(f"Agent {short_id}: staggering resume — waiting "
                            f"{_wait:.0f}s for global resume slot")
                        time.sleep(_wait)
                        continue
                break
            if state.finishing:
                break
            # Re-read state file to pick up force_quota toggled by TUI
            try:
                sf = AGENTS_DIR / f"{short_id}.json"
                d = json.loads(sf.read_text())
                state.force_quota = d.get("force_quota", False)
            except (OSError, json.JSONDecodeError):
                pass
            if state.force_quota:
                log(f"Agent {short_id}: force_quota enabled, skipping wait")
                selection = acquire_backend(
                    config,
                    state=state,
                    claude_config_dir=claude_config_dir,
                    force=True,
                    pin_backend=resume_pin_backend,
                    pin_label=resume_pin_label,
                )
                if selection is None:
                    # No accounts at all — fall back to bare "claude"
                    # for the one-shot path; it will surface a clearer
                    # error inside launch_agent than waiting forever.
                    selection = accounts.Candidate(
                        backend=resume_pin_backend or "claude",
                        label="", model="opus", account_num=0)
                break
            # Enforce the max-pause cap WHILE waiting: if a resuming agent can
            # find no in-quota account (e.g. every account weekly-exhausted),
            # this loop would otherwise sleep forever and hold the claim. Give
            # up past the cap and release everything.
            if state.resuming_after_pause and state.quota_paused_at:
                _cap = float(cfg_get(config, "quota", "max_pause_secs",
                                     default=5400))
                if (time.time() - state.quota_paused_at) > _cap:
                    log(f"Agent {short_id}: quota pause exceeded {_cap:.0f}s "
                        f"while waiting on #{state.claimed_issue}; giving up")
                    _release_paused_session(state, config, claude_config_dir,
                                            reason="quota pause cap exceeded")
                    _abort_one_shot_iteration(state, "quota pause cap exceeded")
                    break
            target = resume_pin_backend or "any backend"
            if resume_pin_label:
                target += f"/{resume_pin_label}"
            # In shared mode a stale external pin (e.g. a wedged swap-account
            # timer) is the usual reason there's no quota here. Drive the
            # manager to re-pick before sleeping so we recover within one cycle
            # instead of waiting on the (possibly dead) timer. Rate-limited and
            # a no-op outside shared mode.
            _maybe_rotate_shared_account(config, min_interval=quota_retry)
            log(f"Agent {short_id}: no quota for {target}, sleeping {quota_retry}s")
            time.sleep(quota_retry)
        if state.finishing or selection is None:
            # If we're stopping mid-pause (graceful finish / SIGUSR1, or the cap
            # give-up above), release the preserved claim + worktree so they
            # aren't leaked until housekeeping reaps them.
            if state.resuming_after_pause:
                _release_paused_session(state, config, claude_config_dir,
                                        reason="agent finishing during quota pause")
            break

        chosen_backend = selection.backend
        chosen_model = selection.model
        chosen_account_label = selection.label
        chosen_account_num = selection.account_num
        state.backend = chosen_backend
        state.model = chosen_model
        # state.account_label and state.lease_acquired_at are set inside
        # acquire_backend on the Claude path; cleared for codex.
        state.write()

        # --- Housekeeping (time-based, every 10 minutes to conserve GH API calls) ---
        # Globally serialised across agents via fcntl.flock so N agents don't
        # each fire their own sweep per window. The stamp file is written at
        # the *end* of the sweep, so a crash mid-sweep doesn't suppress the
        # next attempt for a full HOUSEKEEPING_INTERVAL.
        with _housekeeping_filelock() as owns_sweep:
            if owns_sweep and _housekeeping_due(HOUSEKEEPING_INTERVAL):
                # GitHub-touching steps: skip when GraphQL quota is low. The
                # quota probe itself uses REST and is cached for 30s, so this
                # gate is free when quota is healthy. `_gh_quota_ok()` is
                # threshold-based and fail-open, so the per-call sentinel
                # raised from `_release_claim` and friends is the inner
                # protection if quota goes from healthy to exhausted mid-sweep.
                if _gh_quota_ok():
                    try:
                        coordination(config, "check-blocked")
                    except Exception:
                        pass
                    try:
                        coordination(config, "check-has-pr")
                    except Exception:
                        pass
                    try:
                        check_dead_claimed_issues(config)
                    except Exception:
                        pass
                    try:
                        reconcile_untracked_github_claims()
                    except Exception:
                        pass
                    try:
                        check_dead_pr_claimed_prs(config)
                    except Exception:
                        pass
                else:
                    log(f"Agent {short_id}: housekeeping skipping GitHub steps — "
                        f"GraphQL quota low (remaining="
                        f"{_gh_quota_cache.get('graphql_remaining')})")
                # Local housekeeping always runs — these never touch GitHub.
                try:
                    cleanup_stale_worktrees(config, verbose=True)
                except Exception:
                    pass
                try:
                    cleanup_stale_branches(config, verbose=True)
                except Exception:
                    pass
                # Evict orphan account leases — covers agents that
                # crashed without releasing their lease (the per-agent
                # short_id is no longer in .pod/agents/).
                try:
                    with accounts.lease_critical_section():
                        live = {a.short_id for a in read_all_agents()}
                        live.add(short_id)
                        accounts.evict_orphan_leases(live)
                except Exception:
                    pass
                # Keep the shared-mode account pin fresh (rotation + token
                # refresh) even when no agent is currently starved — this is
                # what replaces the external systemd timer pod must not depend
                # on. No-op outside shared mode.
                try:
                    _maybe_rotate_shared_account(
                        config, min_interval=HOUSEKEEPING_INTERVAL)
                except Exception:
                    pass
                _housekeeping_mark_done()

        # --- Disk-space guard ---
        # Pause dispatch when free disk drops below the configured threshold.
        # A disk-full event during ``git worktree add`` corrupts state in ways
        # that take manual cleanup to recover from (orphan branch refs that
        # block all future dispatches), so it's much cheaper to back off here.
        disk_reason = _disk_low(config)
        if disk_reason:
            log(f"Agent {short_id}: pausing dispatch — disk low ({disk_reason})")
            state.status = "waiting_disk"
            state.write()
            time.sleep(60)
            continue

        # --- Dispatch (sets state.lock_held atomically if lock acquired) ---
        # If this agent was spawned to resume a specific session, skip dispatch.
        _resume_uuid = state.resume_session_uuid
        if state.resuming_after_pause:
            # Resuming a session paused for a usage limit: reuse the SAME
            # session, worktree and GitHub claim, and take precedence over
            # one-shot dispatch. Continue the existing conversation via
            # `--resume`; no new claim, no new worktree.
            _resume_uuid = state.uuid
            chosen_type = state.worker_type or "work"
            prompt = ("You were interrupted by a usage limit and have been "
                      "resumed (possibly on a different account). Review your "
                      "conversation history and continue exactly where you "
                      "left off.")
            lock_name = ""
            wt_config = {}
            state.status = "dispatching"
            state.write()
            log(f"Agent {short_id}: resuming after quota pause — session "
                f"{_resume_uuid} on [{chosen_account_label}]")
        elif state.target_issue:
            # One-shot mode (set by `pod once`). Skip dispatch, pin the
            # work type the caller chose, and append a directive to the
            # prompt so the agent claims #N specifically instead of
            # picking from the unclaimed queue. No lock taken — once
            # workers live outside the dispatch pool by design.
            _resume_uuid = None
            chosen_type = state.target_type or "feature"
            wt_config = worker_types.get(chosen_type, {})
            lock_name = ""
            # Clear the `waiting_quota` status set before the (force_quota-
            # short-circuited) quota loop, so the TUI shows worktree-setup
            # / build-cache-copy progress rather than a stale wait label.
            state.status = "dispatching"
            state.worker_type = chosen_type
            state.write()
            log(f"Agent {short_id}: one-shot mode, target issue #{state.target_issue} "
                f"as {chosen_type}")
        elif _resume_uuid:
            state.resume_session_uuid = ""
            chosen_type = "work"
            prompt = "You were interrupted mid-task. Review your conversation history and continue where you left off."
            lock_name = ""
            wt_config = {}
            state.status = "dispatching"
            state.worker_type = chosen_type
            state.write()
            log(f"Agent {short_id}: resuming session {_resume_uuid}")
        else:
            _resume_uuid = None
            state.status = "dispatching"
            state.write()
            chosen_type = dispatch(config, state)
            if chosen_type is None:
                state.status = "waiting_dispatch"
                state.write()
                consecutive_wait += 1
                # If GH rate limit is exhausted, sleep until reset instead
                # of burning more calls every 60s
                rl_wait = _gh_rate_limit_wait()
                if rl_wait > 0:
                    wait = rl_wait + 30
                    log(f"Agent {short_id}: GH rate limit low, sleeping {wait}s until reset")
                    time.sleep(wait)
                    _clear_gh_cache()
                else:
                    # Linear backoff: 60s, 120s, 180s, ..., capped at 300s
                    wait = min(quota_retry * consecutive_wait, 300)
                    log(f"Agent {short_id}: dispatch returned None (waiting {wait}s, attempt {consecutive_wait})")
                    time.sleep(wait)
                continue

            consecutive_wait = 0  # Reset backoff on successful dispatch
            wt_config = worker_types.get(chosen_type, {})
            lock_name = wt_config.get("lock", "")

            state.worker_type = chosen_type
            state.write()

            log(f"Agent {short_id}: dispatched as {chosen_type}")

        # --- Session setup ---
        if state.resuming_after_pause:
            # Resume after a quota pause: keep the SAME session, claim, worktree
            # and cumulative counters. (The JSONL monitor keeps tokens/activity
            # current; resetting claimed_issue here would drop the held claim.)
            session_uuid = state.uuid
        else:
            session_uuid = _resume_uuid if _resume_uuid else str(uuid.uuid4())
            state.uuid = session_uuid
            state.session_start = time.time()
            state.claimed_issue = 0
            state.pr_number = 0
            state.tokens_in = 0
            state.tokens_out = 0
            state.cache_read = 0
            state.cache_create = 0
            state.last_text = ""
            state.last_activity = 0.0

        if state.resuming_after_pause:
            # Reuse the preserved worktree/branch. setup_worktree() scrubs
            # leftover state before recreating, which would delete the
            # in-progress work we paused to keep.
            wt_dir = state.worktree
            branch = state.branch
            if not wt_dir or not os.path.isdir(wt_dir):
                log(f"Agent {short_id}: resume-after-pause but worktree "
                    f"{wt_dir!r} is gone; releasing claim and aborting")
                if (state.claimed_issue > 0 and state.pr_number == 0
                        and state.worker_type not in ("repair", "replan")):
                    try:
                        if _release_claim(
                                str(state.claimed_issue), state.uuid, 0,
                                reason="Claim released — paused session's "
                                       "worktree vanished; cannot resume."):
                            clear_claim(state.claimed_issue,
                                        session_uuid=state.uuid)
                    except _GraphQLRateLimited:
                        pass
                state.resuming_after_pause = False
                state.quota_paused_at = 0.0
                state.worktree = ""
                state.branch = ""
                _abort_one_shot_iteration(state, "resume worktree missing")
                state.write()
                if state.finishing:
                    break
                continue
            state.git_start = _git_rev(wt_dir)
        else:
            # Use session UUID prefix for worktree/branch to avoid branch reuse
            # across sessions.  When the same agent process iterates, a persistent
            # short_id would reuse the same branch name, hitting an existing remote
            # PR from the *previous* issue.  A per-session prefix gives each
            # session its own branch (agent/<session-prefix>), preventing the
            # create-pr "PR already exists" shortcut from silently linking the
            # wrong issue.
            session_short = session_uuid[:8]

            # Pre-register worktree in state BEFORE creation so that
            # cleanup_stale_worktrees() won't delete it during setup.
            base = PROJECT_DIR / cfg_get(config, "project", "worktree_base", default="worktrees")
            wt_dir_expected = str(base / session_short)
            branch_expected = f"agent/{session_short}"
            state.worktree = wt_dir_expected
            state.branch = branch_expected
            state.write()

            try:
                wt_dir, branch = setup_worktree(config, session_short, own_agent_id=short_id)
            except subprocess.CalledProcessError as e:
                stderr_text = (e.stderr or b"").decode(errors="replace").strip()
                log(f"Agent {short_id}: worktree setup failed: {e}"
                    + (f" — stderr: {stderr_text}" if stderr_text else ""))
                state.worktree = ""
                state.branch = ""
                if lock_name:
                    coordination(config, f"unlock-{lock_name}")
                    state.lock_held = ""
                state.status = "error"
                _abort_one_shot_iteration(state, "worktree setup failed")
                state.write()
                if state.finishing:
                    break
                time.sleep(10)
                continue

            if not os.path.isdir(wt_dir):
                log(f"Agent {short_id}: worktree dir missing after setup_worktree returned: {wt_dir}")
                state.worktree = ""
                state.branch = ""
                if lock_name:
                    coordination(config, f"unlock-{lock_name}")
                    state.lock_held = ""
                state.status = "error"
                _abort_one_shot_iteration(state, "worktree dir missing")
                state.write()
                if state.finishing:
                    break
                time.sleep(10)
                continue

            state.worktree = wt_dir
            state.branch = branch
            state.git_start = _git_rev(wt_dir)

        install_agent_config(wt_dir, backend=chosen_backend)

        if not _resume_uuid:
            if state.target_issue:
                # One-shot mode: use the dedicated `/once` template
                # (which explicitly claims the target issue rather than
                # listing the queue) and append a structured directive
                # carrying the issue number + inferred worker type. The
                # in-process claim-mismatch guard in the JSONL monitor
                # is the runtime backstop if the model slips.
                prompt = _once_prompt(chosen_backend)
                prompt = (
                    f"{prompt}\n\n"
                    f"---\n"
                    f"ISSUE NUMBER: {state.target_issue}\n"
                    f"WORKER TYPE: {chosen_type}\n"
                    f"---"
                )
            else:
                prompt = _worker_prompt(config, chosen_type,
                                         backend=chosen_backend)

        if wt_config.get("copy_build_cache", wt_config.get("copy_lake_cache", False)):
            copy_build_cache(wt_dir, config)

        # --- Start JSONL monitor ---
        jsonl_path = get_jsonl_path(wt_dir, session_uuid, claude_config_dir,
                                    backend=chosen_backend, config=config)
        stop_monitor = threading.Event()
        monitor_thread = threading.Thread(
            target=jsonl_monitor,
            args=(jsonl_path, state, stop_monitor, chosen_backend),
            daemon=True,
        )
        monitor_thread.start()

        # --- Launch agent ---
        state.status = "running"
        state.write()
        log(f"Agent {short_id}: launching {chosen_backend} session {session_uuid} in {wt_dir}")

        try:
            _agent_proc = launch_agent(config, session_uuid, prompt, wt_dir,
                                        claude_config_dir,
                                        backend=chosen_backend,
                                        model=chosen_model,
                                        account_label=chosen_account_label)
        except (OSError, FileNotFoundError) as e:
            log(f"Agent {short_id}: failed to launch {chosen_backend}: {e}")
            stop_monitor.set()
            if state.resuming_after_pause:
                # A resume launch failed. The generic path would scrub the
                # preserved worktree and drop the held claim silently; release
                # the claim explicitly instead.
                _release_paused_session(state, config, claude_config_dir,
                                        reason="resume launch failed")
            else:
                cleanup_worktree(wt_dir, branch)
            if lock_name:
                coordination(config, f"unlock-{lock_name}")
                state.lock_held = ""
            state.status = "error"
            _abort_one_shot_iteration(state, "agent launch failed")
            state.write()
            if state.finishing:
                break
            time.sleep(10)
            continue

        # --- Monitor until claude exits ---
        _last_tracked_issue = 0
        _last_tracked_repair_pr = 0
        _stuck_first_detected = 0.0   # monotonic time of first stuck detection
        _stuck_kill_pending = False    # True after phase-1 detection, waiting for confirm
        _last_stuck_check = 0.0       # monotonic time of last health check

        while _agent_proc.poll() is None:
            state.write()
            # Track claim changes: write to history when claimed, clear
            # when PR created. `repair`/`replan` workers do not own a
            # claimed issue (repair tracks via `repair_pr`, replan edits
            # directly), so skip the issue-claim bookkeeping for them —
            # otherwise a stray `claimed_issue` from a buggy parser
            # would pollute claim-history.json.
            if (state.claimed_issue > 0 and state.pr_number == 0
                    and state.worker_type not in ("repair", "replan")):
                if state.claimed_issue != _last_tracked_issue:
                    record_claim(state.claimed_issue, state.uuid, state.short_id)
                    _last_tracked_issue = state.claimed_issue
            elif state.pr_number > 0 and state.claimed_issue > 0:
                clear_claim(state.claimed_issue, session_uuid=state.uuid)
                _last_tracked_issue = 0

            # One-shot guard: terminate the agent if it claimed an
            # issue that doesn't match the `pod once` target. Prompt
            # directives are advisory; this is the hard backstop that
            # protects against a model slip claiming a different
            # unclaimed issue.
            if (state.target_issue
                    and state.claimed_issue
                    and state.claimed_issue != state.target_issue):
                log(f"Agent {short_id}: one-shot CLAIM MISMATCH — "
                    f"claimed #{state.claimed_issue} but target was "
                    f"#{state.target_issue}. Terminating session.")
                try:
                    _agent_proc.terminate()
                except (OSError, AttributeError):
                    pass
                state.finishing = True
                break
            # Track repair-PR claim lifecycle (parallel to issue-claim). A
            # repair agent never has a claimed_issue, so these live on a
            # separate branch keyed by state.repair_pr.
            if state.repair_pr > 0 and state.repair_pr != _last_tracked_repair_pr:
                record_pr_claim(state.repair_pr, state.uuid, state.short_id)
                _last_tracked_repair_pr = state.repair_pr
            elif state.repair_pr == 0 and _last_tracked_repair_pr > 0:
                clear_pr_claim(_last_tracked_repair_pr, session_uuid=state.uuid)
                _last_tracked_repair_pr = 0

            # --- Stuck-agent detection ---
            stuck_initial = _reload_config_value(
                "monitor", "stuck_initial_timeout", default=3600)
            stuck_confirm = _reload_config_value(
                "monitor", "stuck_confirm_timeout", default=1200)
            stuck_interval = _reload_config_value(
                "monitor", "stuck_check_interval", default=30)

            # Determine idle duration (wall-clock, since last_activity uses time.time)
            if state.last_activity > 0:
                idle = time.time() - state.last_activity
            elif state.session_start > 0:
                # Fallback: session never produced assistant output
                idle = time.time() - state.session_start
            else:
                idle = 0

            if idle >= stuck_initial and _agent_proc is not None:
                now_mono = time.monotonic()
                if now_mono - _last_stuck_check >= stuck_interval:
                    _last_stuck_check = now_mono
                    is_stuck, detail = _check_process_stuck(_agent_proc.pid)

                    if is_stuck and not _stuck_kill_pending:
                        # Phase 1: first stuck detection — log and start confirm timer
                        _stuck_first_detected = now_mono
                        _stuck_kill_pending = True
                        log(f"Agent {short_id}: stuck detected "
                            f"(idle {human_duration(idle)}, {detail}). "
                            f"Will re-check in {human_duration(stuck_confirm)}.")
                    elif is_stuck and _stuck_kill_pending and \
                            now_mono - _stuck_first_detected >= stuck_confirm:
                        # Phase 2: confirmed stuck — kill subprocess
                        log(f"Agent {short_id}: KILLING stuck subprocess "
                            f"(idle {human_duration(idle)}, confirmed after "
                            f"{human_duration(now_mono - _stuck_first_detected)}). "
                            f"Detail: {detail}")
                        _kill_stuck_subprocess(_agent_proc)
                        # Don't break — let poll() return non-None on next iteration
                    elif not is_stuck and _stuck_kill_pending:
                        # Activity detected — reset
                        log(f"Agent {short_id}: stuck detection reset — "
                            f"process showed activity: {detail}")
                        _stuck_kill_pending = False
                        _stuck_first_detected = 0.0
            elif _stuck_kill_pending:
                # New JSONL output arrived (idle dropped below threshold) — reset
                _stuck_kill_pending = False
                _stuck_first_detected = 0.0

            time.sleep(poll_interval)

        exit_code = _agent_proc.returncode
        _agent_proc = None

        # --- Session ended ---
        stop_monitor.set()
        monitor_thread.join(timeout=5)

        # Pop bubble container (no-op when not in bubble mode). Done here
        # rather than in cleanup_worktree so the bubble is reclaimed even
        # if a later cleanup step throws.
        _pop_agent_bubble()

        # (Final JSONL drain handled by the monitor thread's exit path.)

        elapsed = time.time() - state.session_start
        git_end = _git_rev(wt_dir)
        tok = token_summary(state, config)
        log(f"Agent {short_id}: session finished exit={exit_code} "
            f"duration={human_duration(elapsed)} {tok} "
            f"git:{state.git_start}..{git_end}")

        # This iteration's resume launch (if any) has now returned; clear the
        # flag so a clean completion exits instead of looping as a resume. A
        # fresh usage-limit hit (below) re-arms it.
        state.resuming_after_pause = False

        # --- Quota exhaustion detection from stdout ---
        # Must run BEFORE claim cleanup: coordination skip marks issues as
        # "replan", but quota exhaustion is temporary — the issue is fine,
        # only the quota is depleted.  Release the claim without replan.
        # Read the session stdout tail once and reuse it to classify the
        # failure: quota exhaustion here, auth failure in the circuit
        # breaker below.
        stdout_tail = (_read_session_stdout_tail(session_uuid, config)
                       if exit_code != 0 else "")
        _is_quota_exhausted = any(m in stdout_tail for m in _QUOTA_MARKERS)
        # Transient server-side API errors (e.g. "529 Overloaded", other 5xx)
        # are not a quota problem — the account is fine, the API just needs a
        # moment. Pause-and-resume the same way (keep claim/worktree/session),
        # with a short backoff first so the resume doesn't immediately re-hit
        # the overload storm.
        _is_transient_api = (
            exit_code != 0
            and not _is_quota_exhausted
            and any(m in stdout_tail.lower() for m in _TRANSIENT_API_MARKERS))
        _should_pause = _is_quota_exhausted or _is_transient_api

        if not _should_pause:
            # Session got past any usage limit / transient error — reset budget.
            state.quota_paused_at = 0.0

        if _should_pause:
            # --- Pause-and-resume (preferred over abort) --------------------
            # Keep the claim, worktree and session; wait for quota to return
            # (swap-account rotates the canonical to a fresh account, or the 5h
            # session window resets), then `--resume` and continue. Bounded by
            # `quota.max_pause_secs` so a claim/worktree is not held for days
            # when every account is weekly-exhausted.
            # Only Claude has a real `--resume`; and bubble mode stores the
            # session JSONL outside the path the launcher checks, so a "resume"
            # there would silently start a fresh conversation. Restrict to
            # non-bubble Claude one-shot agents with live, resumable work.
            _can_resume = (
                state.target_issue > 0
                and state.claimed_issue > 0
                and state.pr_number == 0
                and state.worker_type not in ("repair", "replan")
                and chosen_backend == "claude"
                and not _use_bubble(config)
                and bool(wt_dir) and os.path.isdir(wt_dir))
            if _can_resume and state.quota_paused_at == 0.0:
                state.quota_paused_at = time.time()
            _pause_cap = float(cfg_get(config, "quota", "max_pause_secs",
                                       default=5400))
            _paused_for = (time.time() - state.quota_paused_at
                           if state.quota_paused_at else 0.0)
            if _can_resume and _paused_for <= _pause_cap:
                if _is_transient_api:
                    # Clamp the backoff: reject 0/negative (fast-loop or a
                    # time.sleep ValueError) and cap to the remaining pause
                    # budget so a transient pause can't overshoot max_pause_secs.
                    _remaining = max(1.0, _pause_cap - _paused_for)
                    _backoff = max(1.0, min(_remaining, float(cfg_get(
                        config, "quota", "transient_backoff_secs",
                        default=60))))
                    log(f"Agent {short_id}: transient API error (529/5xx) — "
                        f"pausing session {state.uuid} on "
                        f"#{state.claimed_issue} "
                        f"(paused {_paused_for:.0f}s / cap {_pause_cap:.0f}s); "
                        f"backing off {_backoff:.0f}s then resuming")
                    time.sleep(_backoff)
                else:
                    log(f"Agent {short_id}: usage limit hit — pausing session "
                        f"{state.uuid} on #{state.claimed_issue} "
                        f"(paused {_paused_for:.0f}s / cap {_pause_cap:.0f}s); "
                        f"will resume on a fresh account")
                # Release the account lease either way: for quota it frees the
                # exhausted account; for a transient error it frees the healthy
                # account so other queued work can use it during our backoff (we
                # re-acquire the best available account on resume). Clearing
                # account_label un-pins the resume so it follows swap-account.
                _release_account_lease(state, claude_config_dir)
                state.resuming_after_pause = True
                state.force_quota = False   # one-shot starts True; must wait
                state.status = "waiting_quota"
                state.write()
                continue  # → top-of-loop quota wait, then resume
            if state.quota_paused_at and _paused_for > _pause_cap:
                log(f"Agent {short_id}: quota pause exceeded {_pause_cap:.0f}s "
                    f"on #{state.claimed_issue}; giving up (releasing)")
            state.quota_paused_at = 0.0
            # --- Fall through: original abort/release path ------------------
            log(f"Agent {short_id}: session stdout indicates quota exhaustion "
                f"or transient API errors past the pause cap, will re-check "
                f"quota before next dispatch")
            # Release claim without replan — issue is fine, just quota-gated.
            # Must call `_release_claim` (not just `clear_claim`) so the GitHub
            # `claimed` label is actually removed: this session UUID is about
            # to be discarded when the next dispatch generates a fresh one,
            # at which point no live agent owns the GitHub claim and any
            # `clear_claim`-only path would leak it as an orphan label that
            # `check_dead_claimed_issues` then skips forever (it ignores
            # `released:True` entries). On transient GitHub failure (likely
            # under quota pressure), leave the history entry untouched so the
            # next housekeeping reconcile picks it up.
            # `repair`/`replan` workers do not legitimately own a claimed
            # issue (see the symmetric cleanup-skip path below).
            if (state.claimed_issue > 0 and state.pr_number == 0
                    and state.worker_type not in ("repair", "replan")):
                released_ok = False
                try:
                    released_ok = _release_claim(
                        str(state.claimed_issue), state.uuid, 0,
                        reason=(f"Claim released — worker session "
                                f"`{state.uuid}` exited due to quota "
                                f"exhaustion. Available for reclaim."))
                except _GraphQLRateLimited as e:
                    log(f"Agent {short_id}: GH rate-limited releasing "
                        f"#{state.claimed_issue} ({e}); will retry via housekeeping")
                if released_ok:
                    clear_claim(state.claimed_issue, session_uuid=state.uuid)
                    log(f"Agent {short_id}: released claim on #{state.claimed_issue} (quota exhaustion, not replan)")
                else:
                    log(f"Agent {short_id}: GitHub release failed for "
                        f"#{state.claimed_issue} (quota exhaustion); "
                        f"leaving history entry for housekeeping retry")
            cleanup_worktree(wt_dir, branch)
            state.worktree = ""
            state.branch = ""
            if lock_name:
                try:
                    coordination(config, f"unlock-{lock_name}")
                except Exception:
                    pass
                state.lock_held = ""
            # Release the Claude account lease (and harvest any refreshed
            # token) so the next iteration is free to pick a different
            # account — the exhausted one would just trigger this same
            # path again until it resets.
            _release_account_lease(state, claude_config_dir)
            _abort_one_shot_iteration(state, "quota exhaustion")
            state.write()
            if state.finishing:
                break
            continue  # loops back to top-of-loop quota check

        # --- Clear uncompleted claims so check_dead_claimed_issues doesn't
        #     mistake our finished session for a dead one and spawn a new agent.
        #     Without this, every session that ends without a PR leaves a stale
        #     claim entry pointing to our old session UUID; the next housekeeping
        #     cycle from any agent sees that UUID as dead and forks a new agent,
        #     causing unbounded agent proliferation.
        # Only runs for workers that actually claim issues (e.g. `work`,
        # `plan`). `repair` claims PRs (tracked in `repair_pr`) and `replan`
        # edits issues directly without claiming, so calling `coordination
        # skip` on whatever happens to be in `claimed_issue` for those
        # types just churns on closed-issue errors.
        if (state.claimed_issue > 0 and state.pr_number == 0
                and state.worker_type not in ("repair", "replan")):
            skip_ok = False
            try:
                coordination(
                    config, "skip", str(state.claimed_issue),
                    f"Session ended without PR (session {state.uuid})",
                    env_extra={"POD_SESSION_ID": state.uuid},
                    cwd=wt_dir,
                )
                skip_ok = True
            except Exception:
                log(f"Agent {short_id}: coordination skip #{state.claimed_issue} failed")
            if skip_ok:
                clear_claim(state.claimed_issue, session_uuid=state.uuid)
                log(f"Agent {short_id}: cleared claim on #{state.claimed_issue} (session ended without PR)")
            else:
                # Leave in claim-history so check_dead_claimed_issues or
                # reconcile_untracked_github_claims can clean it up later.
                log(f"Agent {short_id}: keeping claim #{state.claimed_issue} in history (skip failed, will retry via housekeeping)")

        # --- Circuit breaker: sessions that never reached the model ---
        # A zero-token session that either exited fast (<15s) or exited
        # non-zero is a startup failure (auth/login, missing config,
        # crash), not real work. The non-zero arm matters because an auth
        # failure on a logged-out account takes ~20s to fail — longer than
        # a bare `elapsed < 15` bound — so without it the agent would
        # re-dispatch onto the broken account indefinitely.
        if _session_is_broken(exit_code, state.tokens_in, state.tokens_out, elapsed):
            rapid_failures = getattr(state, '_rapid_failures', 0) + 1
            state._rapid_failures = rapid_failures
            backoff = min(60 * rapid_failures, 300)
            if exit_code != 0 and _looks_like_auth_failure(stdout_tail):
                log(f"Agent {short_id}: authentication failed on account "
                    f"'{state.account_label or '<none>'}' (exit {exit_code}, "
                    f"0 tokens) — the account is likely logged out; re-login "
                    f"or check `pod accounts`. Backing off {backoff}s "
                    f"(failure #{rapid_failures})")
            else:
                log(f"Agent {short_id}: session exited in {elapsed:.0f}s with "
                    f"0 tokens (rapid failure #{rapid_failures}), backing off "
                    f"{backoff}s")
            cleanup_worktree(wt_dir, branch)
            state.worktree = ""
            state.branch = ""
            if lock_name:
                try:
                    coordination(config, f"unlock-{lock_name}")
                except Exception:
                    pass
                state.lock_held = ""
            # Rapid-failure sessions release the account lease too:
            # we're about to back off for several minutes and another
            # agent may want this account in the meantime.
            _release_account_lease(state, claude_config_dir)
            state.status = "backoff"
            _abort_one_shot_iteration(state, f"rapid failure #{rapid_failures}")
            state.write()
            if state.finishing:
                break
            time.sleep(backoff)
            continue
        else:
            # Reset rapid failure counter on successful session
            state._rapid_failures = 0

        # --- Cleanup ---
        cleanup_worktree(wt_dir, branch)
        state.worktree = ""
        state.branch = ""

        if lock_name:
            try:
                coordination(config, f"unlock-{lock_name}")
            except Exception:
                pass
            state.lock_held = ""

        # One-shot mode (set by `pod once`): exit after a single iteration
        # regardless of whether the session succeeded. The user explicitly
        # asked for "agent goes away when the issue is done" — picking up
        # a *different* issue on the next loop would defeat that.
        if state.target_issue:
            log(f"Agent {short_id}: one-shot mode complete, exiting "
                f"(target_issue=#{state.target_issue})")
            state.finishing = True

        state.write()

    # --- Agent loop exited ---
    log(f"Agent {short_id} exiting (finishing={state.finishing})")
    state.status = "stopped"
    # Release any held Claude account lease + harvest refreshed token
    # back to canonical before the orphan-eviction sweep notices we're
    # gone. Skipping this leaves the lease blocking other agents for up
    # to one housekeeping cycle.
    _release_account_lease(state, claude_config_dir)
    state.write()
    # Tear down the per-agent CLAUDE_CONFIG_DIR — keychain entries
    # survive but are orphaned (cheap on disk; harmless).
    if claude_config_dir is not None:
        accounts.cleanup_agent_claude_config_dir(POD_DIR, short_id)
    # Leave state file briefly so TUI can see "stopped", then remove
    time.sleep(2)
    state.remove_file()


def _session_is_broken(exit_code: int, tokens_in: int, tokens_out: int,
                       elapsed: float) -> bool:
    """True when a session never reached the model and should trip the
    circuit breaker.

    A session that produced zero tokens and either exited fast (<15s) or
    exited non-zero is a startup failure — auth/login, missing config, or
    a crash. The non-zero-exit arm is what catches a logged-out / expired
    account, whose auth failure takes ~20s and so slips past a bare
    ``elapsed < 15`` bound.
    """
    if tokens_in or tokens_out:
        return False
    return exit_code != 0 or elapsed < 15


# Substrings in a failed session's stdout tail that mean "quota depleted"
# (temporary; the account is fine) rather than a broken account.
_QUOTA_MARKERS = ("hit your limit", "rate limit", "quota exceeded")

# Substrings that mean the failure was a transient server-side API error
# (HTTP 529 "Overloaded" or other 5xx) rather than quota or a code crash.
# The account/quota is fine — the API just needs a moment. Handled like a
# quota pause (keep claim/worktree/session, --resume) but with a short backoff
# first. Matched case-insensitively against the lowercased stdout tail; kept
# TIGHT — require the CLI's "API Error: <5xx>" framing or an HTTP/status-
# qualified 529 — so ordinary agent output (a Lean file mentioning "overloaded"
# notation, "line 529", a fixture saying "service unavailable") is NOT
# misclassified as a transient API failure.
_TRANSIENT_API_MARKERS = (
    "api error: 5",            # "API Error: 500/502/503/504/529 ..."
    "529 overloaded",
    "http 529", "status 529", "error code 529",
)

# Substrings that mean the failure was authentication — a logged-out /
# expired / cleared OAuth token or bad API key — rather than quota or a
# code crash. Matched case-insensitively against the lowercased tail.
_AUTH_FAILURE_MARKERS = (
    "invalid api key",
    "please run /login",
    "oauth token has expired",
    "token has expired",
    "token expired",
    "authentication failed",
    "authentication error",
    "unauthorized",
    "not logged in",
    "login required",
    "invalid bearer token",
    "401",
)


def _read_session_stdout_tail(session_uuid: str, config: dict,
                              nbytes: int = 4096) -> str:
    """Return the last ``nbytes`` of a session's stdout, lowercased.

    Empty string on any read error. Used to classify why a failed session
    exited (quota exhaustion vs authentication failure).
    """
    try:
        path = (PROJECT_DIR
                / cfg_get(config, "project", "session_dir", default="sessions")
                / f"{session_uuid}.stdout")
        with open(path, "rb") as f:
            f.seek(0, 2)  # end
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace").lower()
    except OSError:
        return ""


def _looks_like_auth_failure(stdout_tail: str) -> bool:
    """True if the lowercased stdout tail contains an auth-failure marker."""
    return any(m in stdout_tail for m in _AUTH_FAILURE_MARKERS)


def _git_rev(wt_dir: str) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", wt_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def spawn_agent(config: dict, agent_id: str | None = None,
                resume_uuid: str | None = None,
                target_issue: int = 0,
                target_type: str = "") -> int:
    """Fork a new agent process. Returns PID of the intermediate child.

    Uses double-fork so the agent is orphaned (adopted by init) and never
    becomes a zombie — without touching SIGCHLD in the calling process.
    Corrupting SIGCHLD in an agent process that calls spawn_agent (via
    check_dead_claimed_issues) would break git's internal waitpid and cause
    silent failures in git worktree add.
    """
    # Gate every dispatch path. TTL-cached, so back-to-back spawns from
    # the TUI loop or auto-spawn don't each hit the GitHub API.
    check_repo_security(config)
    pid = os.fork()
    if pid > 0:
        # Parent: wait for the short-lived intermediate child (exits immediately).
        # Retry on EINTR so a signal doesn't cause a spurious launch failure.
        while True:
            try:
                os.waitpid(pid, 0)
                break
            except ChildProcessError:
                break
            except InterruptedError:
                continue
        return pid
    # Intermediate child: fork the actual agent, then exit immediately so
    # the agent is adopted by init (no zombie, no SIGCHLD games needed).
    try:
        gc_pid = os.fork()
    except Exception:
        os._exit(1)
    if gc_pid > 0:
        os._exit(0)
    # Grandchild: the actual agent process, adopted by init upon exit.
    try:
        os.setsid()
        devnull_r = os.open(os.devnull, os.O_RDONLY)
        devnull_w = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_r, 0)
        os.dup2(devnull_w, 1)
        os.dup2(devnull_w, 2)
        os.close(devnull_r)
        os.close(devnull_w)
        agent_process_main(config, agent_id, resume_uuid,
                            target_issue=target_issue,
                            target_type=target_type)
    except Exception as e:
        # Log the full traceback so the operator can diagnose post-hoc.
        # The agent's stdout went to /dev/null after the dup2 above, so
        # this `log` call (which writes to .pod/pod.log) is the only
        # surviving record.
        log(f"Agent process crashed: {e}\n{traceback.format_exc()}")
        # Best-effort cleanup so the GitHub claim is released and the
        # state file is removed; otherwise the issue sits in
        # claim-history.json with `released: false` and the GitHub
        # `claimed` label, blocking other agents from picking it up
        # until the next housekeeping dead-claim sweep notices the
        # session UUID is dead. Wrap in its own try/except because the
        # crash itself may have come from a broken environment in which
        # the cleanup helpers will also raise (e.g. a stale `_data_dir()`
        # path after an editable-install switcheroo).
        sid = _agent_state.short_id if _agent_state else "?"
        uid = _agent_state.uuid if _agent_state else "?"
        skip_msg = f"Agent crashed: {e!s} (session {uid})"
        try:
            _release_agent_resources("crashed", skip_msg)
        except Exception as cleanup_e:
            log(f"Agent {sid}: crash cleanup also failed: {cleanup_e}\n"
                f"{traceback.format_exc()}")
    finally:
        os._exit(0)


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

def run_tui(config: dict):
    """Run the interactive curses TUI."""
    check_repo_security(config)
    curses.wrapper(_tui_main, config)


def _tui_main(stdscr, config: dict):
    # Rebuild claim history from GitHub before starting, so pod can reattach
    # to sessions that were running before a pod restart.
    try:
        sync_claims_from_github()
    except Exception:
        pass

    curses.curs_set(0)
    stdscr.timeout(1000)  # 1 second refresh
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)    # Running / merged
    curses.init_pair(2, curses.COLOR_YELLOW, -1)    # Finishing / blocked
    curses.init_pair(3, curses.COLOR_RED, -1)        # Dead/error / failing
    curses.init_pair(4, curses.COLOR_CYAN, -1)       # Header / closed
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)    # PR open / has-pr

    selected_section = "agents"  # "agents" or "items"
    selected_idx = 0
    message = ""
    message_time = 0.0
    input_mode = ""  # "", "kill_confirm"
    cached_queue: int | None = None
    queue_fetch_time = 0.0
    cached_items: list[GHItem] = []
    displayed_items: list[GHItem] = []  # subset shown on screen (active-first order)
    items_fetch_time = 0.0
    cached_blocked_deps: dict[int, list[int]] = {}
    blocked_deps_fetch_time = 0.0
    cached_has_pr_links: dict[int, list[int]] = {}
    has_pr_links_fetch_time = 0.0
    cached_lock_status: str | None = None  # "locked" or "unlocked"
    lock_fetch_time = 0.0
    # (The repair worker type, if configured, is dispatched directly by
    # `dispatch_queue_balance` — no TUI-level tracking needed.)
    # Check for pre-existing return-to-human signal synchronously at startup.
    # If it was already set before this TUI session, we honour it (target=0, banner)
    # but do NOT send SIGUSR1 — the human restarted pod intentionally.
    try:
        cached_return_to_human = get_return_to_human(config)
    except Exception:
        cached_return_to_human = False
    return_to_human_fetch_time = 0.0
    # _acted_on_return_to_human: True once this session has already taken the
    # one-shot side effects (write_target=0, banner) for the current label
    # presence.  Reset to False when the label disappears so a re-application
    # re-fires those side effects.
    _acted_on_return_to_human = cached_return_to_human
    # Agents alive when this TUI started up while the label was already set
    # are "grandfathered": we won't send them SIGUSR1, on the assumption that
    # the operator restarted pod intentionally and wants to keep observing
    # them.  Any agent that appears after startup (including a stale orphan
    # TUI's looping agent that was missed by the original one-shot fire) is
    # signalled normally.  Identified by (short_id, pid_start_time) so a
    # recycled short_id with a different process can't accidentally inherit
    # grandfather status.
    _grandfathered_agents: set[tuple[str, float]] = set()
    if cached_return_to_human:
        write_target(0)
        _grandfathered_agents = {
            (a.short_id, a.pid_start_time)
            for a in read_all_agents()
            if a.status not in ("stopped", "dead") and a.pid > 0
        }
    CACHE_SECS = 60           # Refresh interval for primary GitHub API data (queue, items)
    CACHE_SECS_SLOW = 120     # Refresh interval for less-critical data (blocked deps, lock, return-to-human)

    # Background fetch infrastructure: all GH/coordination calls run in daemon
    # threads so the TUI never blocks on network I/O.
    _bg_data: dict = {"queue": None, "items": None, "blocked_deps": None,
                      "has_pr_links": None, "lock_status": None,
                      "return_to_human": None}
    _bg_active: set = set()
    _bg_mutex = threading.Lock()

    def _bg_run(key, fn, *args):
        try:
            result = fn(*args)
            with _bg_mutex:
                # Only commit if we weren't cancelled while the call was in
                # flight.  A synchronous handler (e.g. 'r' clearing
                # return-to-human, see #30) cancels a pending fetch by
                # discarding the key from _bg_active so its sampled-before-
                # clear result can't race back in and re-set stale state.
                if key in _bg_active:
                    _bg_data[key] = result
        except Exception:
            pass
        finally:
            with _bg_mutex:
                _bg_active.discard(key)

    def _bg_fetch(key, fn, *args):
        with _bg_mutex:
            if key in _bg_active:
                return
            _bg_active.add(key)
        threading.Thread(target=_bg_run, args=(key, fn) + args, daemon=True).start()

    def _get_lock_status_str():
        r = coordination(config, "lock-status")
        return r.stdout.strip()

    last_auto_spawn_time = 0.0  # Timestamp of last TUI-initiated auto-spawn
    auto_spawn_failures = 0     # Consecutive auto-spawns where agents died instantly
    auto_spawn_paused = False   # True when crash-loop detected

    show_costs = cfg_get(config, "monitor", "show_costs", default=False)

    # Compute all-time historical cost once at startup
    claude_config_dir = get_isolated_config_dir(config)
    historical_cost = compute_historical_cost(config, claude_config_dir) if show_costs else 0.0
    # Track session-accumulated costs (persists across agent deaths)
    session_agent_costs: dict[str, float] = {}  # agent short_id -> last known cost
    # Accumulated cost from previous iterations (when token counters reset)
    session_cost_offsets: dict[str, float] = {}  # agent short_id -> sum of prior iterations' costs
    # Baseline costs for agents already running when pod started (to avoid double-counting)
    baseline_costs: dict[str, float] = {}
    for a in read_all_agents():
        if a.status not in ("stopped", "dead"):
            baseline_costs[a.short_id] = a.cost(config)

    while True:
        agents = read_all_agents()
        # Accumulate costs from all agents (including dying ones) before cleanup
        for a in agents:
            current = a.cost(config)
            prev = session_agent_costs.get(a.short_id)
            if prev is not None and current < prev:
                # Token counters were reset (new loop iteration) — accumulate previous cost
                session_cost_offsets[a.short_id] = session_cost_offsets.get(a.short_id, 0.0) + prev
            session_agent_costs[a.short_id] = current
        # Clean up dead agent state files
        for a in agents:
            if a.status == "dead":
                a.remove_file()
        agents = [a for a in agents if a.status != "dead"]

        height, width = stdscr.getmaxyx()
        stdscr.erase()

        # --- Header ---
        # session_cost = delta from agents already running at startup + full cost of new agents
        session_cost = sum(
            session_cost_offsets.get(sid, 0.0) + cost - baseline_costs.get(sid, 0.0)
            for sid, cost in session_agent_costs.items()
        )
        session_runs = sum(1 for sid in session_agent_costs if sid not in baseline_costs)
        # Auto-spawn maintains `target` *regular* agents. One-shot
        # agents (launched via `pod once`) live outside the pool by
        # design — see `_is_regular_agent` for the predicate.
        running = sum(1 for a in agents if _is_regular_agent(a))

        # Kick off background refreshes (non-blocking) and read latest results.
        now = time.time()
        if cached_queue is None or now - queue_fetch_time > CACHE_SECS:
            _bg_fetch("queue", get_queue_depth, config)
            queue_fetch_time = now
        if now - items_fetch_time > CACHE_SECS:
            _bg_fetch("items", fetch_issues_and_prs)
            items_fetch_time = now
        if now - blocked_deps_fetch_time > CACHE_SECS_SLOW:
            _bg_fetch("blocked_deps", fetch_blocked_deps)
            blocked_deps_fetch_time = now
        if now - has_pr_links_fetch_time > CACHE_SECS_SLOW:
            _bg_fetch("has_pr_links", fetch_has_pr_links)
            has_pr_links_fetch_time = now
        # Return-to-human signal: planner labels sentinel issue when no work remains.
        # Poll continuously so we observe both label re-applications (planner re-fires
        # after each empty-queue tick) and external clears (`gh issue edit --remove-label`).
        if now - return_to_human_fetch_time > CACHE_SECS:
            _bg_fetch("return_to_human", get_return_to_human, config)
            return_to_human_fetch_time = now
        # Lock status: check agent state first (cheap), fall back to background API
        if now - lock_fetch_time > CACHE_SECS_SLOW:
            lock_fetch_time = now
            agent_holds_lock = any(a.lock_held for a in agents
                                  if a.status not in ("stopped", "dead"))
            if agent_holds_lock:
                with _bg_mutex:
                    _bg_data["lock_status"] = "locked"
            else:
                _bg_fetch("lock_status", _get_lock_status_str)
        # Apply any completed background results to cached display values
        with _bg_mutex:
            if _bg_data["queue"] is not None:
                cached_queue = _bg_data["queue"]
            if _bg_data["items"] is not None:
                cached_items = _bg_data["items"]
            if _bg_data["blocked_deps"] is not None:
                cached_blocked_deps = _bg_data["blocked_deps"]
            if _bg_data["has_pr_links"] is not None:
                cached_has_pr_links = _bg_data["has_pr_links"]
            if _bg_data["lock_status"] is not None:
                cached_lock_status = _bg_data["lock_status"]
            if _bg_data["return_to_human"] is not None:
                cached_return_to_human = _bg_data["return_to_human"]

        # React to return-to-human signal: set target=0 and gracefully finish agents.
        # The latched flag (`_acted_on_return_to_human`) gates one-shot side effects —
        # writing target=0 and showing the status banner — so they don't repeat each tick.
        # The per-agent re-signal loop runs on every tick whenever the label is present,
        # so any running agent that hasn't already received SIGUSR1 (including a stale
        # orphan TUI's looping agent that the original one-shot fire missed) gets stopped.
        # `_grandfathered_agents` exempts agents present at startup-with-label-present so
        # an intentional restart doesn't kill agents the operator is observing.
        if cached_return_to_human:
            if not _acted_on_return_to_human:
                _acted_on_return_to_human = True
                write_target(0)
                message = "Return-to-human: planner found no remaining work. Target set to 0; agents finishing."
                message_time = now
            for a in agents:
                if (a.short_id, a.pid_start_time) in _grandfathered_agents:
                    continue
                # One-shot agents (`pod once`) are explicitly out-of-band:
                # the user dispatched them to a specific issue and they
                # exit on their own after a single iteration. The
                # planner's "no remaining work" signal is about the
                # regular queue, not priority work, so don't preempt
                # them here.
                if a.target_issue:
                    continue
                if (a.pid > 0
                        and not a.finishing
                        and a.status not in ("stopped", "dead", "finishing")
                        and _pid_is_valid(a.pid, a.pid_start_time)):
                    try:
                        os.kill(a.pid, signal.SIGUSR1)
                    except (ProcessLookupError, OSError):
                        pass
        elif _acted_on_return_to_human:
            # Label cleared (externally or via 'r' keypress path below).  Reset the
            # latch and the grandfather set so a future re-application starts fresh:
            # any agents still alive at re-application time will be signalled.
            _acted_on_return_to_human = False
            _grandfathered_agents = set()

        # Auto-spawn agents to maintain target (only if no agents are finishing).
        target = get_effective_target()
        desired = 0 if target is None else target
        if target is not None and running < desired and now - last_auto_spawn_time > 5.0:
            finishing_count = sum(1 for a in agents if a.status == "finishing")
            if finishing_count == 0:
                if auto_spawn_paused:
                    pass  # Crash-loop detected; suppress auto-spawn (user can press 'a')
                else:
                    # Detect crash-loop: if we spawned recently and all agents are already gone,
                    # they died instantly.
                    if last_auto_spawn_time > 0 and running == 0:
                        auto_spawn_failures += 1
                    else:
                        auto_spawn_failures = 0
                    if auto_spawn_failures >= 3:
                        log("Auto-spawn crash-loop detected (3 consecutive rapid deaths). "
                            "Pausing auto-spawn. Press 'a' to retry.")
                        auto_spawn_paused = True
                    else:
                        n_to_spawn = desired - running
                        for _ in range(n_to_spawn):
                            spawn_agent(config)
                        last_auto_spawn_time = now

        lock_indicator = ""
        if cached_lock_status == "locked":
            lock_indicator = " | LOCK"
        if FORCE_QUOTA_FILE.exists():
            lock_indicator += " | FORCE"
        # Items panel staleness — survives a single missed refresh tick
        # without flashing the indicator. Computed against the cache's
        # original `fetched_at`, so a snapshot loaded from disk on
        # startup shows its true age (potentially hours), not a fresh-
        # looking few seconds.
        tui_cache_ts, _tui_cache_body = _TUI_REFRESH_CACHE
        if tui_cache_ts > 0 and _tui_cache_body is not None:
            age = time.time() - tui_cache_ts
            if age >= _TUI_VERY_STALE_AFTER_S:
                lock_indicator += f" | items very stale: {human_duration(age)}"
            elif age >= _TUI_STALE_AFTER_S:
                lock_indicator += f" | items stale: {human_duration(age)}"
        # Quota low-water indicator — the stderr warning from the layer
        # gets overwritten by curses, so surface a compact marker here
        # while the bucket is still in its depleted window.
        try:
            low = gh.get_client().low_water_buckets()
        except Exception:
            low = set()
        for b in sorted(low):
            lock_indicator += f" | {b}:!"
        effective = target  # already computed via get_effective_target() above
        user_t = read_target()
        if effective is not None and user_t is not None and effective < user_t:
            agent_str = f"{running}/{effective}({user_t})"
        else:
            agent_str = f"{running}/{effective}" if effective is not None else str(running)
        agent_word = "agent" if (effective or running) == 1 else "agents"
        if show_costs:
            all_time = historical_cost + session_cost
            session_info = f"${session_cost:.2f} this session, {session_runs} run{'s' if session_runs != 1 else ''}"
            cost_str = f" | ${all_time:.2f} total ({session_info})"
        else:
            cost_str = ""
        if cached_queue is not None:
            header = f" pod -- {agent_str} {agent_word} running | queue: {cached_queue}{lock_indicator}{cost_str}"
        else:
            header = f" pod -- {agent_str} {agent_word} running{lock_indicator}{cost_str}"

        _addstr(stdscr, 0, 0, header[:width], curses.color_pair(4) | curses.A_BOLD)
        _addstr(stdscr, 1, 0, "─" * min(width, 80), curses.color_pair(4))

        # --- Agent table header ---
        col_fmt = "  {:>2} {:8} {:16} {:>6} {:>15} {}"
        hdr = " " + col_fmt.format("#", "ID", "Type", "Time", "Tokens", "Activity")
        _addstr(stdscr, 2, 0, hdr[:width], curses.A_DIM)

        # --- Clamp agent selection (items clamped after panel renders) ---
        if selected_section == "agents":
            if not agents:
                selected_idx = 0
            else:
                selected_idx = min(selected_idx, len(agents) - 1)

        is_agent_selected = selected_section == "agents"

        # --- Agent rows ---
        agents_rendered = 0
        for i, agent in enumerate(agents):
            row = 3 + i
            if row >= height - 3:
                break
            agents_rendered = i + 1

            # Mode label
            mode = agent.worker_type or "???"
            if agent.claimed_issue > 0:
                if agent.pr_number > 0:
                    mode = f"{mode} #{agent.claimed_issue}->PR"
                else:
                    mode = f"{mode} #{agent.claimed_issue}"
            if agent.finishing:
                mode += " (fin)"
            if agent.force_quota:
                mode += " !"

            # Elapsed
            elapsed = ""
            if agent.session_start > 0:
                elapsed = human_duration(time.time() - agent.session_start)

            # Tokens
            tok = token_summary(agent, config, show_costs)

            # Activity text: for non-running states show status, not stale last_text
            if agent.status in ("running", "finishing"):
                activity = agent.last_text or agent.status
            else:
                activity = agent.status
            # Thinking detection: if JSONL is stale but process is alive
            if (agent.last_activity > 0 and
                    time.time() - agent.last_activity > 10 and
                    agent.status == "running"):
                stale = int(time.time() - agent.last_activity)
                activity = f"thinking {human_duration(stale)}"
            # Prefix with which Claude account holds the lease (if any),
            # so operators can see at a glance which account each agent
            # is burning quota on. Applied last so it survives the
            # thinking-detection overwrite above.
            if agent.account_label:
                activity = f"[{agent.account_label}] {activity}"

            # Sanitize: collapse newlines/tabs to spaces
            activity = " ".join(activity.split())

            # Truncate activity to fit (1 extra for marker char)
            prefix = col_fmt.format(i + 1, agent.short_id, mode[:16], elapsed, tok, "")
            max_act = width - len(prefix) - 1  # -1 for marker
            if max_act > 10 and len(activity) > max_act:
                activity = activity[:max_act - 3] + "..."

            marker = ">" if is_agent_selected and i == selected_idx else " "
            line = f"{marker}" + col_fmt.format(i + 1, agent.short_id, mode[:16], elapsed, tok, activity)

            attr = curses.A_NORMAL
            if is_agent_selected and i == selected_idx:
                attr = curses.color_pair(5) | curses.A_BOLD
            elif agent.status == "finishing" or agent.finishing:
                attr = curses.color_pair(2)
            elif agent.status in ("dead", "error", "killed"):
                attr = curses.color_pair(3)
            elif agent.status == "running":
                attr = curses.color_pair(1)

            _addstr(stdscr, row, 0, line[:width], attr)

        # --- Issues/PRs panel ---
        # Row budget: 3 top (header+sep+colhdr) + rendered agents + 3 bottom (sep+help+msg)
        agents_end = 3 + agents_rendered
        footer_fixed = 4 if cached_return_to_human else 3  # separator + help + message [+ banner]
        avail_for_items = height - agents_end - footer_fixed
        # Need: 1 blank + 1 separator + 1 header + at least 1 item row = 4
        items_shown = 0
        items_start_row = 0
        displayed_items = []  # reset each frame
        if cached_items and avail_for_items >= 4:
            max_item_rows = avail_for_items - 3  # blank + separator + header
            # Open items are always shown. Closed/merged fill remaining slots
            # (newest first), and are the first to go when space is tight.
            active = [it for it in cached_items if it.state not in ("closed", "merged")]
            inactive = [it for it in cached_items if it.state in ("closed", "merged")]
            slots_for_inactive = max(0, max_item_rows - len(active))
            inactive_tail = inactive[:slots_for_inactive]

            # Build a tree over the active set under the rule "a parent is a
            # blocker": things indent under whatever they are waiting on, so
            # an issue with open `depends-on:` deps indents under those deps,
            # and a `has-pr` issue indents under its open closing PR. Cycles
            # are broken arbitrarily but deterministically (first-input wins).
            MAX_INDENT = 6
            by_num_active = {it.number: it for it in active}

            # Precompute raw parents per issue, split by edge kind so the
            # annotation can keep dependency-issue and closing-PR wording
            # distinct ("Blocked on" vs "PR"). `raw_parents_of` is the
            # concatenation in tree-pick priority order (blocked deps first,
            # PR links second — pure heuristic).
            raw_blocked_of: dict[int, list[int]] = {}
            raw_pr_of: dict[int, list[int]] = {}
            raw_parents_of: dict[int, list[int]] = {}
            for it in active:
                if it.kind != "issue":
                    continue
                bd = list(cached_blocked_deps.get(it.number, []))
                pl = list(cached_has_pr_links.get(it.number, []))
                if bd:
                    raw_blocked_of[it.number] = bd
                if pl:
                    raw_pr_of[it.number] = pl
                combined = bd + pl
                if combined:
                    raw_parents_of[it.number] = combined

            parent_of: dict[int, int | None] = {}

            def _would_cycle(child_num: int, candidate: int) -> bool:
                cur: int | None = candidate
                seen: set[int] = set()
                while cur is not None and cur not in seen:
                    if cur == child_num:
                        return True
                    seen.add(cur)
                    cur = parent_of.get(cur)
                return False

            for it in active:
                chosen: int | None = None
                for p in raw_parents_of.get(it.number, ()):
                    if (p in by_num_active and p != it.number
                            and not _would_cycle(it.number, p)):
                        chosen = p
                        break
                parent_of[it.number] = chosen

            children_of: dict[int, list] = {it.number: [] for it in active}
            roots: list = []
            for it in active:
                p = parent_of[it.number]
                if p is None:
                    roots.append(it)
                else:
                    children_of[p].append(it)

            active_rows: list[tuple] = []  # (item, depth)

            def _dfs(node, depth: int):
                active_rows.append((node, depth))
                for ch in children_of[node.number]:
                    _dfs(ch, depth + 1)

            for r in roots:
                _dfs(r, 0)

            def _annotation(item) -> str:
                if item.kind != "issue":
                    return ""
                chosen_p = parent_of.get(item.number)
                raw_b = raw_blocked_of.get(item.number, [])
                raw_p = raw_pr_of.get(item.number, [])
                parts: list[str] = []
                if chosen_p is None:
                    # Root row: show everything it's waiting on, split by kind.
                    if raw_b:
                        parts.append("Blocked on "
                                     + ", ".join(f"#{n}" for n in raw_b))
                    if raw_p:
                        parts.append("PR "
                                     + ", ".join(f"#{n}" for n in raw_p))
                else:
                    # Nested: list off-tree parents, preserving issue-vs-PR
                    # wording so an unchosen closing PR doesn't masquerade
                    # as a blocked-dep number.
                    other_b = [n for n in raw_b if n != chosen_p]
                    other_p = [n for n in raw_p if n != chosen_p]
                    if other_b:
                        parts.append("Also blocked on "
                                     + ", ".join(f"#{n}" for n in other_b))
                    if other_p:
                        parts.append("PR "
                                     + ", ".join(f"#{n}" for n in other_p))
                # Orphan `has-pr` label (every named PR has merged/closed).
                if ("has-pr" in item.labels
                        and cached_has_pr_links.get(item.number) == []):
                    parts.append("PR ?")
                return "".join(f"[{p}] " for p in parts)

            render_rows: list[tuple] = []  # (item, depth, annotation)
            for it, d in active_rows:
                render_rows.append((it, d, _annotation(it)))
            for it in inactive_tail:
                render_rows.append((it, 0, ""))

            items_shown = len(render_rows)

            items_start_row = agents_end + 1  # skip 1 blank line
            _addstr(stdscr, items_start_row, 0, "─" * min(width, 80), curses.color_pair(4))
            item_fmt = "  {:<6} {:<10} {:<8}  {}"
            item_hdr = " " + item_fmt.format("#", "State", "When", "Title")
            _addstr(stdscr, items_start_row + 1, 0, item_hdr[:width], curses.A_DIM)

            for j, (item, depth, annotation) in enumerate(render_rows):
                irow = items_start_row + 2 + j
                if irow >= height - footer_fixed:
                    items_shown = j
                    break

                # Unified state: combines kind, labels, CI, and state
                if item.kind == "issue":
                    if item.state == "closed":
                        state_str, state_color = "closed", curses.color_pair(4)
                    elif "claimed" in item.labels:
                        state_str, state_color = "claimed", curses.color_pair(1)
                    elif "has-pr" in item.labels:
                        state_str, state_color = "has-pr", curses.color_pair(6)
                    elif "blocked" in item.labels:
                        state_str, state_color = "blocked", curses.color_pair(2)
                    elif "replan" in item.labels:
                        state_str, state_color = "replan", curses.color_pair(2)
                    else:
                        state_str, state_color = "open", curses.A_NORMAL
                else:  # pr
                    if item.state == "merged":
                        state_str, state_color = "merged", curses.color_pair(4)
                    elif item.state == "closed":
                        state_str, state_color = "closed", curses.color_pair(4)
                    elif item.ci_status == "fail":
                        state_str, state_color = "failing", curses.color_pair(3)
                    else:
                        state_str, state_color = "open", curses.color_pair(6)

                age = timeago(item.timestamp)
                kind_prefix = "PR" if item.kind == "pr" else "I"
                state_display = f"{kind_prefix} {state_str}"

                indent = " " * (2 * min(depth, MAX_INDENT))
                title = indent + annotation + item.title
                # Truncate title to fit
                prefix_len = 31
                max_title = width - prefix_len - 1
                if max_title > 10 and len(title) > max_title:
                    title = title[:max_title - 3] + "..."

                is_item_sel = not is_agent_selected and j == selected_idx
                marker = ">" if is_item_sel else " "
                num_str = f"#{item.number}"
                line = marker + item_fmt.format(num_str, state_display, age, title)

                attr = state_color
                if is_item_sel:
                    attr = curses.color_pair(5) | curses.A_BOLD

                _addstr(stdscr, irow, 0, line[:width], attr)

            # Sync displayed_items to actually-rendered rows so selection
            # navigation (and `o`/open) indexes match what is on screen.
            displayed_items = [it for (it, _, _) in render_rows[:items_shown]]

        # --- Clamp items selection (now that items_shown is known) ---
        if selected_section == "items":
            if items_shown == 0:
                selected_section = "agents"
                selected_idx = max(0, agents_rendered - 1)
                is_agent_selected = True
            else:
                selected_idx = min(selected_idx, items_shown - 1)

        # --- Footer separator ---
        footer_row = max(agents_end, height - footer_fixed)
        if items_shown > 0:
            footer_row = max(items_start_row + 2 + items_shown, height - footer_fixed)
        if footer_row < height - 1:
            _addstr(stdscr, footer_row, 0, "─" * min(width, 80), curses.color_pair(4))

        # --- Footer ---
        footer_row2 = footer_row + 1
        if input_mode == "kill_confirm":
            if agents and is_agent_selected and 0 <= selected_idx < len(agents):
                footer_text = f" Kill agent {agents[selected_idx].short_id}? (y/n)"
            else:
                footer_text = " No agent selected"
                input_mode = ""
        else:
            footer_text = " [a]dd  [f]inish  [k]ill  [o]pen  [!]force  [F]orce-all  [L]ock  [q]uit  [Q]uit all  ↑↓/1-9"
            if cached_return_to_human:
                footer_text = " [r]esume work  " + footer_text.lstrip()

        if footer_row2 < height:
            _addstr(stdscr, footer_row2, 0, footer_text[:width], curses.A_DIM)

        # --- Message line ---
        msg_active = bool(message and time.time() - message_time < 3)
        msg_row = footer_row2 + 1
        if msg_active and msg_row < height:
            _addstr(stdscr, msg_row, 0, f" {message}"[:width], curses.A_BOLD)

        # --- Return-to-human banner (persistent) ---
        if cached_return_to_human:
            banner_row = msg_row + 1 if msg_active else msg_row
            if banner_row < height:
                banner = " *** return-to-human: planner found no remaining work — target=0, agents finishing ***"
                _addstr(stdscr, banner_row, 0, banner[:width], curses.color_pair(2) | curses.A_BOLD)

        stdscr.refresh()

        # --- Input ---
        try:
            ch = stdscr.getch()
        except curses.error:
            continue

        if ch == -1:
            continue

        if input_mode == "kill_confirm":
            if ch in (ord("y"), ord("Y")):
                if agents and is_agent_selected and 0 <= selected_idx < len(agents):
                    agent = agents[selected_idx]
                    _kill_agent(config, agent)
                    message = f"Killed agent {agent.short_id}"
                    message_time = time.time()
                    cur_target = read_target()
                    if cur_target is not None and cur_target > 0:
                        write_target(cur_target - 1)
            else:
                message = "Kill cancelled"
                message_time = time.time()
            input_mode = ""
            continue

        # Normal mode
        if ch == ord("q"):
            break
        elif ch == ord("Q"):
            # Quit all: finish all agents, then wait
            write_target(0)  # Don't auto-respawn after quitting all
            for a in agents:
                if a.pid > 0 and a.status not in ("stopped", "dead") and _pid_is_valid(a.pid, a.pid_start_time):
                    try:
                        os.kill(a.pid, signal.SIGUSR1)
                    except (ProcessLookupError, OSError):
                        pass
            message = "Sent finish signal to all agents. Waiting..."
            message_time = time.time()
            stdscr.refresh()
            # Wait for all to stop (with timeout display)
            deadline = time.time() + 600  # 10 min max
            while time.time() < deadline:
                live = [a for a in read_all_agents() if a.status not in ("stopped", "dead")]
                if not live:
                    break
                time.sleep(2)
            break
        elif ch == ord("a") or ch == ord("A"):
            spawn_agent(config)
            cur_target = read_target()
            write_target((cur_target or running) + 1)
            # Clear planner advisories so user intent takes immediate effect
            PLANNER_TARGET_FILE.unlink(missing_ok=True)
            PLANNER_MIN_QUEUE_FILE.unlink(missing_ok=True)
            last_auto_spawn_time = time.time()
            auto_spawn_failures = 0
            auto_spawn_paused = False
            message = "Launched 1 agent"
            message_time = time.time()
        elif ch == ord("f"):
            if is_agent_selected and agents and 0 <= selected_idx < len(agents):
                agent = agents[selected_idx]
                if agent.pid > 0 and _pid_is_valid(agent.pid, agent.pid_start_time):
                    try:
                        os.kill(agent.pid, signal.SIGUSR1)
                        message = f"Finish signal sent to {agent.short_id}"
                        cur_target = read_target()
                        if cur_target is not None and cur_target > 0:
                            write_target(cur_target - 1)
                    except (ProcessLookupError, OSError):
                        message = f"Agent {agent.short_id} not running"
                else:
                    message = f"Agent {agent.short_id} not running"
                message_time = time.time()
        elif ch == ord("k") or ch == ord("K"):
            if is_agent_selected and agents and 0 <= selected_idx < len(agents):
                input_mode = "kill_confirm"
        elif ch == ord("o") or ch == ord("O"):
            if is_agent_selected:
                # Open agent's claimed issue
                if agents and 0 <= selected_idx < len(agents):
                    agent = agents[selected_idx]
                    issue = agent.claimed_issue
                    if issue > 0:
                        try:
                            subprocess.Popen(
                                ["gh", "issue", "view", str(issue), "--web"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                cwd=str(PROJECT_DIR),
                            )
                            message = f"Opening issue #{issue}"
                        except OSError as e:
                            message = f"Failed to open issue: {e}"
                    else:
                        message = f"Agent {agent.short_id} has no claimed issue"
                    message_time = time.time()
            else:
                # Open selected issue/PR
                if displayed_items and 0 <= selected_idx < len(displayed_items):
                    item = displayed_items[selected_idx]
                    gh_cmd = "issue" if item.kind == "issue" else "pr"
                    try:
                        subprocess.Popen(
                            ["gh", gh_cmd, "view", str(item.number), "--web"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            cwd=str(PROJECT_DIR),
                        )
                        message = f"Opening {item.kind} #{item.number}"
                    except OSError as e:
                        message = f"Failed to open: {e}"
                    message_time = time.time()
        elif ch == ord("!"):
            if is_agent_selected and agents and 0 <= selected_idx < len(agents):
                agent = agents[selected_idx]
                # Toggle force_quota in the agent's state file
                try:
                    sf = AGENTS_DIR / f"{agent.short_id}.json"
                    d = json.loads(sf.read_text())
                    new_val = not d.get("force_quota", False)
                    d["force_quota"] = new_val
                    tmp = sf.with_suffix(".tmp")
                    tmp.write_text(json.dumps(d, indent=2) + "\n")
                    tmp.rename(sf)
                    label = "ON" if new_val else "OFF"
                    message = f"Force quota {label} for {agent.short_id}"
                except (OSError, json.JSONDecodeError) as e:
                    message = f"Failed to toggle force: {e}"
                message_time = time.time()
        elif ch == ord("F"):
            # Toggle global force-quota override
            if FORCE_QUOTA_FILE.exists():
                FORCE_QUOTA_FILE.unlink()
                message = "Global force-quota OFF"
            else:
                FORCE_QUOTA_FILE.write_text("")
                message = "Global force-quota ON (all agents skip quota checks)"
            message_time = time.time()
        elif ch == ord("r") or ch == ord("R"):
            # Clear return-to-human signal and resume normal operation
            if cached_return_to_human:
                try:
                    clear_return_to_human(config)
                    cached_return_to_human = False
                    _acted_on_return_to_human = False
                    _grandfathered_agents = set()
                    # Invalidate the background cache so the next main-loop
                    # iteration doesn't re-apply a stale True and re-fire the
                    # one-shot side effects before the next poll (see #30).
                    # Also cancel any in-flight fetch whose result (sampled
                    # before the label was cleared) would otherwise race back
                    # in and re-latch True.
                    with _bg_mutex:
                        _bg_data["return_to_human"] = False
                        _bg_active.discard("return_to_human")
                    message = "Return-to-human signal cleared. Press [a] to add an agent."
                except Exception as e:
                    message = f"Failed to clear signal: {e}"
                message_time = time.time()
        elif ch == ord("l") or ch == ord("L"):
            # Toggle planner lock
            try:
                if cached_lock_status == "locked":
                    coordination(config, "force-unlock-planner")
                    cached_lock_status = "unlocked"
                    lock_fetch_time = time.time()
                    message = "Planner lock released"
                else:
                    r = coordination(config, "lock-planner")
                    if r.returncode == 0:
                        cached_lock_status = "locked"
                        lock_fetch_time = time.time()
                        message = "Planner lock acquired"
                    else:
                        message = "Failed to acquire planner lock"
            except Exception as e:
                message = f"Lock toggle failed: {e}"
            message_time = time.time()
        elif ch == curses.KEY_UP:
            if selected_section == "agents":
                if selected_idx > 0:
                    selected_idx -= 1
                # At top of agents — stay (nowhere to go)
            else:
                if selected_idx > 0:
                    selected_idx -= 1
                elif agents:
                    # Jump to last agent
                    selected_section = "agents"
                    selected_idx = len(agents) - 1
        elif ch == curses.KEY_DOWN:
            if selected_section == "agents":
                if agents and selected_idx < len(agents) - 1:
                    selected_idx += 1
                elif items_shown > 0:
                    # Jump to first item
                    selected_section = "items"
                    selected_idx = 0
            else:
                if selected_idx < items_shown - 1:
                    selected_idx += 1
        elif 49 <= ch <= 57:  # 1-9
            idx = ch - 49
            if idx < len(agents):
                selected_section = "agents"
                selected_idx = idx


def _addstr(win, y: int, x: int, s: str, attr: int = 0):
    """Safe addstr that doesn't crash on edge writes."""
    try:
        win.addstr(y, x, s, attr)
    except curses.error:
        pass


def _kill_agent(config: dict, agent: AgentState):
    """Send SIGTERM to an agent process."""
    if agent.pid > 0 and _pid_is_valid(agent.pid, agent.pid_start_time):
        try:
            os.kill(agent.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            # Process already dead — clean up manually
            if agent.claimed_issue > 0 and agent.pr_number == 0:
                try:
                    coordination(
                        config, "skip", str(agent.claimed_issue),
                        f"Agent killed by operator (session {agent.uuid})",
                        env_extra={"POD_SESSION_ID": agent.uuid},
                    )
                except Exception:
                    pass
            if agent.lock_held:
                try:
                    coordination(config, f"unlock-{agent.lock_held}")
                except Exception:
                    pass
            if agent.worktree and agent.branch:
                cleanup_worktree(agent.worktree, agent.branch)
            agent.remove_file()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_list(config: dict, args):
    """Show running agents."""
    agents = read_all_agents()

    if not agents:
        print("No agents running.")
        return

    alive = [a for a in agents if a.status != "dead"]
    dead = [a for a in agents if a.status == "dead"]

    # Clean up dead state files
    for a in dead:
        a.remove_file()

    if not alive:
        print("No agents running.")
        return

    fmt = "{:>2}  {:8}  {:16}  {:>6}  {:>15}  {}"
    print(fmt.format("#", "ID", "Type", "Time", "Tokens", "Status"))
    print("─" * 74)
    for i, a in enumerate(alive):
        mode = a.worker_type or "???"
        if a.claimed_issue > 0:
            if a.pr_number > 0:
                mode = f"{mode} #{a.claimed_issue}->PR"
            else:
                mode = f"{mode} #{a.claimed_issue}"
        if a.finishing:
            mode += " (fin)"

        elapsed = human_duration(time.time() - a.session_start) if a.session_start > 0 else ""
        tok = token_summary(a, config)
        if a.status in ("running", "finishing"):
            activity = a.last_text or a.status
        else:
            activity = a.status

        print(fmt.format(i + 1, a.short_id, mode[:16], elapsed, tok, activity[:40]))


def cmd_add(config: dict, args):
    """Launch new agents and update the target count."""
    n = args.count if args.count else 1
    for _ in range(n):
        pid = spawn_agent(config)
        say(f"Launched agent (PID {pid})")
    # Only count regular agents toward the target. A one-shot agent
    # launched via `pod once` lives outside the pool by design — see
    # `_is_regular_agent` — so counting it here would silently raise the
    # regular pool's target while the priority job runs.
    regular_alive = sum(1 for a in read_all_agents()
                        if _is_regular_agent(a))
    cur_target = read_target()
    new_target = max(regular_alive, (cur_target or 0) + n)
    write_target(new_target)
    print(f"Launched {n} agent{'s' if n != 1 else ''}. Target: {new_target}.")


def cmd_once(config: dict, args):
    """Launch a one-shot agent targeting a specific issue.

    Bypasses the dispatch queue and the dispatch quota cap (the agent
    is spawned outside the `target` pool so the regular auto-spawn loop
    is unaffected; see `_is_regular_agent`). The agent runs a single
    iteration and exits, regardless of whether the issue completes
    successfully.

    Preflight checks (issue exists, is open, isn't already claimed,
    has a label matching a configured worker type) run *before* the
    fork so the user gets a clean error rather than a worker that
    cleanups itself a few seconds in.
    """
    issue = args.issue
    work_type = args.work_type
    worker_types = cfg_get(config, "worker_types", default={}) or {}

    # Preflight: pull labels and state in a single gh call so we can
    # diagnose every refusal mode without a second round-trip.
    try:
        info_json = subprocess.check_output(
            ["gh", "issue", "view", str(issue),
             "--json", "labels,state,title"],
            cwd=str(PROJECT_DIR), text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to read issue #{issue}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        info = json.loads(info_json)
    except json.JSONDecodeError as e:
        print(f"Could not parse `gh issue view #{issue}` JSON: {e}",
              file=sys.stderr)
        sys.exit(2)

    if info.get("state") != "OPEN":
        print(f"Issue #{issue} is {info.get('state')!r}; refusing to "
              f"launch a one-shot worker on a non-open issue.",
              file=sys.stderr)
        sys.exit(2)

    label_names = [
        lab.get("name", "") for lab in info.get("labels", [])
    ]
    if "claimed" in label_names:
        print(f"Issue #{issue} already has the `claimed` label "
              f"(another agent holds it). Refusing to spawn a "
              f"one-shot worker — wait for the existing claim to "
              f"finish, or release it via `pod coordination skip`.",
              file=sys.stderr)
        sys.exit(2)

    # Explicit `--type` validation. The inferred path is already
    # validated against `worker_types` below; the explicit path was
    # falling through to `_worker_prompt`'s `/{worker_type}` fallback,
    # which produced a fast-failing agent rather than a clean error.
    if work_type and work_type not in worker_types:
        print(
            f"Unknown --type {work_type!r}. Configured worker types: "
            f"{', '.join(sorted(worker_types)) or '(none)'}.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Auto-detect work type from issue labels when the user didn't
    # specify one. We honour every configured worker type, so any label
    # the dispatch loop recognises (`feature`, `plan`, `repair`,
    # `replan`, `review`, etc.) will pick the matching prompt.
    if not work_type:
        for label in label_names:
            if label in worker_types:
                work_type = label
                break
        if not work_type and "work" in worker_types:
            work_type = "work"
            print(f"Defaulting --type work for issue #{issue} "
                  f"(labels={label_names!r} matched no worker type).")
        elif not work_type:
            print(
                f"Could not infer worker type for issue #{issue} from "
                f"labels {label_names!r}, and no `work` worker type is "
                f"configured. Pass --type explicitly "
                f"(one of: {', '.join(sorted(worker_types)) or 'feature'}).",
                file=sys.stderr,
            )
            sys.exit(2)
        else:
            print(f"Inferred --type {work_type} from labels {label_names!r}.")

    pid = spawn_agent(config, target_issue=issue, target_type=work_type)
    print(
        f"Launched one-shot agent (PID {pid}) on issue #{issue} as "
        f"{work_type!r}. Bypasses target / quota; exits after one "
        f"iteration."
    )


def cmd_finish(config: dict, args):
    """Signal agent(s) to finish after current work."""
    agents = read_all_agents()
    alive = [a for a in agents if a.status not in ("stopped", "dead")]

    if args.target == "all":
        targets = alive
    else:
        targets = [a for a in alive if a.short_id.startswith(args.target)]
        if not targets:
            print(f"No running agent matching '{args.target}'")
            return

    signaled_regular = 0
    for a in targets:
        if not _pid_is_valid(a.pid, a.pid_start_time):
            print(f"Agent {a.short_id} not running (PID {a.pid})")
            continue
        try:
            os.kill(a.pid, signal.SIGUSR1)
            print(f"Finish signal sent to {a.short_id} (PID {a.pid})")
            # One-shot agents don't occupy a target slot
            # (`_is_regular_agent`) — finishing one shouldn't free a
            # slot for auto-spawn to refill.
            if _is_regular_agent(a):
                signaled_regular += 1
        except (ProcessLookupError, OSError):
            print(f"Agent {a.short_id} not running (PID {a.pid})")

    # Decrement target so auto-spawn doesn't immediately refill the pool.
    cur_target = read_target()
    if cur_target is not None and signaled_regular > 0:
        write_target(max(0, cur_target - signaled_regular))


def cmd_kill(config: dict, args):
    """Kill agent(s) immediately."""
    agents = read_all_agents()
    alive = [a for a in agents if a.status not in ("stopped", "dead")]

    if args.target == "all":
        targets = alive
    else:
        targets = [a for a in alive if a.short_id.startswith(args.target)]
        if not targets:
            print(f"No running agent matching '{args.target}'")
            return

    killed_regular = 0
    for a in targets:
        _kill_agent(config, a)
        print(f"Killed {a.short_id} (PID {a.pid})")
        # One-shot agents don't occupy a target slot — same rationale
        # as `cmd_finish` above.
        if _is_regular_agent(a):
            killed_regular += 1

    # Decrement target so auto-spawn doesn't immediately refill the pool.
    cur_target = read_target()
    if cur_target is not None and killed_regular > 0:
        write_target(max(0, cur_target - killed_regular))


def cmd_status(config: dict, args):
    """Show aggregate status."""
    agents = read_all_agents()
    alive = [a for a in agents if a.status not in ("stopped", "dead")]
    show_costs = cfg_get(config, "monitor", "show_costs", default=False)

    total_in = sum(a.tokens_in + a.cache_read + a.cache_create for a in alive)
    total_out = sum(a.tokens_out for a in alive)

    try:
        queue = get_queue_depth(config)
        print(f"Queue depth:    {queue}")
    except Exception:
        print("Queue depth:    (unavailable)")

    print(f"Running agents: {len(alive)}")

    types = {}
    for a in alive:
        t = a.worker_type or "unknown"
        types[t] = types.get(t, 0) + 1
    if types:
        parts = [f"{v} {k}" for k, v in types.items()]
        print(f"  Breakdown:    {', '.join(parts)}")

    print(f"Total tokens:   {fmt_tokens(total_in)} in / {fmt_tokens(total_out)} out")
    if show_costs:
        session_cost = sum(a.cost(config) for a in alive)
        claude_config_dir = get_isolated_config_dir(config)
        historical_cost = compute_historical_cost(config, claude_config_dir)
        print(f"Running cost:   ${session_cost:.2f}")
        print(f"All-time cost:  ${historical_cost:.2f}")


def cmd_log(config: dict, args):
    """Tail agent's session stdout."""
    agents = read_all_agents()
    if args.target:
        matches = [a for a in agents if a.short_id.startswith(args.target)]
    else:
        # Default to most recent agent
        matches = sorted(agents, key=lambda a: a.session_start, reverse=True)[:1]

    if not matches:
        print("No matching agent found.")
        return

    agent = matches[0]
    session_dir = PROJECT_DIR / cfg_get(config, "project", "session_dir", default="sessions")
    stdout_path = session_dir / f"{agent.uuid}.stdout"
    if not stdout_path.exists():
        print(f"No log file for agent {agent.short_id}")
        return

    # Print last 50 lines
    lines = stdout_path.read_text().splitlines()
    for line in lines[-50:]:
        print(line)


def cmd_config(config: dict, args):
    """Print current configuration."""
    if args.edit:
        editor = os.environ.get("EDITOR", "vi")
        os.execlp(editor, editor, str(CONFIG_PATH))
    else:
        print(CONFIG_PATH.read_text(), end="")


def cmd_gh_stats(config: dict, args) -> None:
    """Summarise `.pod/gh-access.log` to find which callers / paths are
    consuming the most GitHub API budget.

    The log is JSONL, one row per call. We aggregate by the column named
    in `--by` and print: count, ETag-cache-hit ratio, total wall-ms,
    total bytes (best-effort).
    """
    import collections

    log_path = Path(args.log) if args.log else (POD_DIR / "gh-access.log")
    if not log_path.exists():
        print(f"No GitHub access log at {log_path}.", file=sys.stderr)
        print("It is created on first GitHub interaction.", file=sys.stderr)
        sys.exit(1)

    # Parse --since.
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    s = args.since.strip().lower()
    try:
        if s and s[-1] in units:
            since_seconds = int(s[:-1]) * units[s[-1]]
        else:
            since_seconds = int(s)  # seconds
    except ValueError:
        print(f"Invalid --since value: {args.since!r} "
              "(expected e.g. 10m, 1h, 24h, 7d).", file=sys.stderr)
        sys.exit(1)
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=since_seconds)

    # Aggregate.
    by_key: dict[str, dict] = collections.defaultdict(
        lambda: {"count": 0, "cache_hits": 0, "ms": 0, "bytes": 0,
                 "errors": 0})
    total = {"count": 0, "cache_hits": 0, "ms": 0, "bytes": 0, "errors": 0}

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("ts", "")
            try:
                row_dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if row_dt < cutoff:
                continue
            key = str(row.get(args.by, "?"))
            for buf in (by_key[key], total):
                buf["count"] += 1
                if row.get("cache_hit"):
                    buf["cache_hits"] += 1
                buf["ms"] += int(row.get("ms") or 0)
                buf["bytes"] += int(row.get("bytes") or 0)
                if int(row.get("status") or 0) >= 400:
                    buf["errors"] += 1

    if total["count"] == 0:
        print(f"No log rows in the last {args.since} window.")
        return

    rows = sorted(by_key.items(), key=lambda kv: kv[1]["count"], reverse=True)
    rows = rows[: args.top]

    # Format.
    width = max((len(k) for k, _ in rows), default=10)
    width = min(width, 70)
    print(f"GitHub access log: {log_path}")
    print(f"Window: last {args.since}  |  Group by: {args.by}  |  "
          f"Top: {args.top}")
    print(f"Total: {total['count']} calls  "
          f"({100 * total['cache_hits'] / max(1, total['count']):.0f}% ETag-cache hit, "
          f"{total['errors']} errors, "
          f"{total['ms'] / 1000:.1f}s wall, "
          f"{total['bytes'] / 1024:.1f} KiB body)")
    print()
    print(f"  {'#':>5}  {'cache%':>6}  {'errs':>4}  "
          f"{'ms':>7}  {'KiB':>5}  {args.by}")
    print(f"  {'-' * 5}  {'-' * 6}  {'-' * 4}  {'-' * 7}  {'-' * 5}  "
          f"{'-' * width}")
    for key, v in rows:
        cache_pct = 100 * v["cache_hits"] / max(1, v["count"])
        print(f"  {v['count']:>5}  {cache_pct:>5.0f}%  "
              f"{v['errors']:>4}  {v['ms']:>7}  {v['bytes']/1024:>5.1f}  "
              f"{key[:width]}")


def cmd_cleanup(config: dict, args):
    """Remove stale worktrees that no running agent owns."""
    base = PROJECT_DIR / cfg_get(config, "project", "worktree_base", default="worktrees")
    if not base.is_dir():
        print("No worktrees directory found.")
        return

    # Show current state
    all_dirs = [d for d in base.iterdir() if d.is_dir()]
    live_worktrees: set[str] = set()
    for agent in read_all_agents():
        if agent.worktree and agent.status != "dead":
            live_worktrees.add(agent.worktree)

    now = time.time()
    min_age = 0 if args.force else CLEANUP_MIN_AGE_SECONDS
    stale = []
    skipped_young = 0
    skipped_non_pod = 0
    for d in sorted(all_dirs):
        if str(d) in live_worktrees:
            continue
        if not args.force:
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if age < min_age:
                skipped_young += 1
                continue
        if not _is_pod_owned_worktree(d):
            skipped_non_pod += 1
            continue
        stale.append(d)

    if not stale:
        parts = [f"All {len(all_dirs)} worktrees are owned by running agents"]
        if skipped_young:
            parts.append(f"{skipped_young} skipped (too young, use --force)")
        if skipped_non_pod:
            parts.append(f"{skipped_non_pod} skipped (not pod-owned)")
        print(". ".join(parts) + ".")
        return

    print(f"Found {len(stale)} stale worktrees (of {len(all_dirs)} total).")
    if skipped_young:
        print(f"  {skipped_young} skipped (younger than {CLEANUP_MIN_AGE_SECONDS}s, use --force)")
    if skipped_non_pod:
        print(f"  {skipped_non_pod} skipped (not pod-owned)")

    if args.dry_run:
        for d in stale:
            try:
                age = now - d.stat().st_mtime
            except OSError:
                age = 0
            print(f"  would remove: {d.name}  (age {age:.0f}s)")
        print(f"Dry run: {len(stale)} worktrees would be removed.")
        return

    removed = cleanup_stale_worktrees(config, verbose=True, force=args.force)
    print(f"Removed {removed} stale worktrees.")


# ---------------------------------------------------------------------------
# Init / Update / Coordination subcommands
# ---------------------------------------------------------------------------


def _populate_agent_config():
    """Copy bundled agent-config/claude into .pod/claude-config/.

    Only copies credentials/settings files — commands/ and skills/ are
    installed directly into each worktree's .claude/ by install_agent_config,
    not into the isolated config dir.
    """
    src = _data_dir() / "agent-config" / "claude"
    dst = ISOLATED_CONFIG_DIR
    dst.mkdir(parents=True, exist_ok=True)
    EXCLUDE = {"commands", "skills", "CLAUDE.md"}
    for item in src.iterdir():
        if item.name in EXCLUDE:
            continue
        d = dst / item.name
        if item.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(str(item), str(d))
        else:
            shutil.copy2(str(item), str(d))


REQUIRED_LABELS = {
    "agent-plan": "1D76DB",
    "claimed": "FBCA04",
    "blocked": "B60205",
    "has-pr": "5319E7",
    "replan": "D93F0B",
    "coordination": "0E8A16",
    "critical-path": "E11D48",
    "repair-claimed": "7057FF",   # PR is claimed by a repair agent
    "unsalvageable": "5C5C5C",    # PR closed by a repair agent as unsalvageable
}


def _ensure_github_labels():
    """Create any missing GitHub labels required by pod."""
    try:
        r = _gh_cli("label", "list", "--limit", "100",
                     "--json", "name", "--jq", ".[].name", timeout=15)
        if r.returncode != 0:
            print("  warning: could not list labels (gh CLI issue)")
            return
        existing = set(r.stdout.strip().splitlines())
        created = []
        for label, color in REQUIRED_LABELS.items():
            if label not in existing:
                cr = _gh_cli(
                    "label", "create", label, "--color", color,
                    "--description", "pod coordination", timeout=15,
                )
                if cr.returncode == 0:
                    created.append(label)
        if created:
            print(f"  created GitHub labels: {', '.join(created)}")
        else:
            print("  GitHub labels already exist")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  warning: could not ensure GitHub labels (gh CLI not available)")


def _ensure_repo_merge_settings():
    """Enable auto-merge, squash merge, and head-branch auto-delete on the repo.

    Auto-delete replaces `gh pr merge --delete-branch`: the flag makes gh
    check out the default branch locally before deleting, which parks an
    agent worktree on master and locks it against the human's main checkout.
    Letting GitHub delete the merged head branch server-side avoids that.
    """
    try:
        r = _gh_cli("repo", "edit",
                     "--enable-auto-merge", "--enable-squash-merge",
                     "--delete-branch-on-merge",
                     timeout=15)
        if r.returncode == 0:
            print("  enabled auto-merge, squash merge, and head-branch "
                  "auto-delete on GitHub repo")
        else:
            print("  warning: could not enable auto-merge/squash-merge/"
                  "head-branch auto-delete (may need admin access — enable "
                  "manually in repo Settings → General; note that rulesets or "
                  "branch protection can also block automatic deletion)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  warning: could not configure repo merge settings (gh CLI not available)")


def _ensure_branch_protection(init_config):
    """Configure branch protection on the default branch (best effort).

    Required for `gh pr merge --auto` to actually queue PRs — without a
    required status check on the default branch, --auto silently no-ops
    and green PRs sit unmerged.
    """
    required = init_config.get("merge", {}).get("required_checks", []) or []
    if not required:
        print("  branch protection: skipped (empty [merge] required_checks)")
        print("    → Auto-merge on PRs will NOT work until you configure this.")
        print("    → Edit .pod/config.toml, set e.g.:")
        print("        [merge]")
        print("        required_checks = [\"build-and-test\"]")
        print("      using the `name:` of your CI job(s) in .github/workflows/,")
        print("      then re-run `pod init`.")
        return

    client = gh.get_client()
    slug = _get_repo()
    try:
        resp = client.get(f"/repos/{slug}", timeout=15)
        if not resp.ok():
            print("  warning: could not detect default branch (gh CLI issue)")
            return
        default_branch = ((resp.body() or {}).get("default_branch") or "").strip()
        if not default_branch:
            print("  warning: could not detect default branch (empty response)")
            return
    except Exception:
        print("  warning: could not detect default branch (gh CLI not available)")
        return

    payload = {
        "required_status_checks": {
            "strict": False,
            "contexts": list(required),
        },
        "enforce_admins": False,
        "required_pull_request_reviews": None,
        "restrictions": None,
        "allow_force_pushes": False,
        "allow_deletions": False,
    }

    try:
        resp = client.put(
            f"/repos/{slug}/branches/{default_branch}/protection",
            json=payload, timeout=15,
        )
        if resp.ok():
            print(f"  configured branch protection on `{default_branch}` "
                  f"(required: {', '.join(required)})")
        else:
            print(f"  warning: could not set branch protection on `{default_branch}` "
                  f"(HTTP {resp.status} — may need admin access)")
            print(f"    → Settings → Branches → add rule for `{default_branch}`")
            print(f"      with required status checks: {', '.join(required)}")
    except Exception:
        print("  warning: could not configure branch protection (gh CLI not available)")


def cmd_accounts(config: dict, args):
    """Inspect / manage Claude account leases held by pod agents.

    Subactions:
      list                       Show all known accounts and current
                                 leases (`label  owner  pid  age
                                 project`). Free accounts also shown.
      release <label> [--force]  Remove the lease for ``label``.
                                 Refuses if the owning agent is still
                                 live; ``--force`` overrides. The
                                 owning agent's next iteration will
                                 re-acquire (possibly a different
                                 account) on its own.
      evict-orphans              Take the meta-lock and reap any
                                 leases whose owner short_id is no
                                 longer in .pod/agents/.
    """
    action = getattr(args, "accounts_action", None) or "list"
    if action == "list":
        accts = accounts.list_claude_accounts()
        leases = {l.label: l for l in accounts.list_leases()}
        live_short_ids = {a.short_id for a in read_all_agents()
                          if a.status not in ("stopped", "dead")}
        if not accts and not leases:
            print("(no accounts found in ~/.claude/credentials*.json)")
            return
        print(f"{'LABEL':<14} {'OWNER':<10} {'PID':>6} "
              f"{'AGE':>8}  PROJECT")
        for acct in accts:
            lease = leases.get(acct.label)
            if lease is None:
                print(f"{acct.label:<14} {'(free)':<10} {'':>6} {'':>8}  -")
                continue
            age = human_duration(time.time() - lease.acquired_at)
            owner = lease.short_id
            if lease.short_id not in live_short_ids:
                owner += "*"  # orphan
            print(f"{acct.label:<14} {owner:<10} {lease.pid:>6} "
                  f"{age:>8}  {lease.project_dir}")
        # Surface orphans that survived account enumeration (e.g. account
        # file deleted but lease lingered) — also worth reaping.
        unmatched = [l for l in leases.values()
                      if l.label not in {a.label for a in accts}]
        for lease in unmatched:
            age = human_duration(time.time() - lease.acquired_at)
            owner = lease.short_id
            if lease.short_id not in live_short_ids:
                owner += "*"
            print(f"{lease.label:<14} {owner:<10} {lease.pid:>6} "
                  f"{age:>8}  {lease.project_dir} (no account file)")
        if any(owner.endswith("*") for owner in ()):  # never-true; just suppress
            pass
        if any(l.short_id not in live_short_ids for l in leases.values()):
            print("\n* = orphan (owning agent is no longer live). "
                  "`pod accounts evict-orphans` to reap.")
    elif action == "release":
        label = args.label
        force = bool(getattr(args, "force", False))
        leases = {l.label: l for l in accounts.list_leases()}
        lease = leases.get(label)
        if lease is None:
            print(f"No lease held on '{label}'.")
            return
        live_short_ids = {a.short_id for a in read_all_agents()
                          if a.status not in ("stopped", "dead")}
        if lease.short_id in live_short_ids and not force:
            print(f"Refusing to release '{label}': owning agent "
                  f"{lease.short_id} is still live. "
                  f"Use --force to release anyway "
                  f"(the agent's next iteration will re-acquire).")
            sys.exit(1)
        ok = accounts.release_lease(label, lease.short_id)
        if ok:
            print(f"Released '{label}' (was owned by {lease.short_id}).")
        else:
            print(f"Failed to release '{label}'.")
            sys.exit(1)
    elif action == "evict-orphans":
        with accounts.lease_critical_section():
            live = {a.short_id for a in read_all_agents()
                    if a.status not in ("stopped", "dead")}
            evicted = accounts.evict_orphan_leases(live)
        if evicted:
            print(f"Evicted {len(evicted)} orphan lease(s): "
                  f"{', '.join(evicted)}")
        else:
            print("No orphan leases.")
    else:
        print(f"Unknown action: {action}")
        sys.exit(2)


def cmd_init(args):
    """Bootstrap .pod/ in the current git repo."""
    # Verify we're in a git repo
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        print("Not in a git repository.", file=sys.stderr)
        sys.exit(1)
    git_root = Path(r.stdout.strip())
    pod_dir = git_root / ".pod"
    config_path = pod_dir / "config.toml"

    pod_dir.mkdir(parents=True, exist_ok=True)
    (pod_dir / "agents").mkdir(exist_ok=True)

    # .gitignore for .pod/
    gitignore = pod_dir / ".gitignore"
    gitignore.write_text(
        "agents/\n"
        "pod.log\n"
        "claim-history.*\n"
        "claude-config/\n"
        "codex-home/\n"
        "codex-sessions/\n"
        "force-quota\n"
    )

    # Ensure .claude/.pod-checksums is gitignored in the project root
    proj_gitignore = git_root / ".gitignore"
    checksums_pattern = ".claude/.pod-checksums"
    if proj_gitignore.exists():
        existing = proj_gitignore.read_text()
        if checksums_pattern not in existing:
            proj_gitignore.write_text(existing.rstrip("\n") + f"\n{checksums_pattern}\n")
            print(f"  added {checksums_pattern} to .gitignore")
    else:
        proj_gitignore.write_text(f"{checksums_pattern}\n")
        print(f"  created .gitignore with {checksums_pattern}")

    # config.toml
    if not config_path.exists() or getattr(args, "force", False):
        config_path.write_text(DEFAULT_CONFIG)
        print(f"  wrote {config_path.relative_to(git_root)}")
    else:
        print(f"  {config_path.relative_to(git_root)} already exists (use --force to overwrite)")

    # Backend-specific config setup
    # Read backend from the just-written config
    with open(config_path, "rb") as f:
        init_config = tomllib.load(f)
    init_config = _migrate_legacy_config(init_config)
    init_backend = _backend(init_config)

    warning = _instruction_file_warning(git_root, init_backend)
    if warning:
        print(f"  {warning}")

    if init_backend in ("claude", "auto"):
        global ISOLATED_CONFIG_DIR
        ISOLATED_CONFIG_DIR = pod_dir / "claude-config"
        _populate_agent_config()
        rel = ISOLATED_CONFIG_DIR.relative_to(git_root)
        if init_backend == "auto":
            print(f"  populated {rel}/ (auto mode — Codex installs per-session)")
        else:
            print(f"  populated {rel}/")
    else:
        print(f"  {init_backend} backend — agent config installed per-session via CODEX_HOME")

    # Ensure required GitHub labels exist
    _ensure_github_labels()

    # GitHub repo merge settings (required for coordination create-pr auto-merge to work)
    _ensure_repo_merge_settings()

    # Branch protection (also required for --auto on `gh pr merge` to take effect)
    _ensure_branch_protection(init_config)

    print("pod init complete.")


def cmd_update(args):
    """Re-populate agent config from installed package."""
    if not POD_DIR.is_dir():
        print("No .pod/ directory found. Run 'pod init' first.", file=sys.stderr)
        sys.exit(1)
    config = ensure_config()
    backend = _backend(config)
    if backend in ("claude", "auto"):
        # Auto mode may invoke Claude in any iteration, so its config
        # bundle still needs to be installed.
        _populate_agent_config()
        suffix = " (auto mode — Codex installs per-session)" if backend == "auto" else ""
        print(f"Updated .pod/claude-config/ from installed package.{suffix}")
    else:
        print(f"{backend} backend — agent config is installed per-session, nothing to update.")

    # Migration: installs that predate head-branch auto-delete never had it
    # enabled (it is only applied at `pod init`). Re-apply the repo merge
    # settings on update so existing repos stop accumulating stale remote
    # branches now that the merge paths no longer pass --delete-branch.
    _ensure_repo_merge_settings()


def cmd_coordination(args):
    """Pass-through to bundled coordination script."""
    script = str(_data_dir() / "coordination")
    env = dict(os.environ)
    # Pass protected-files from config if available
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            config = tomllib.load(f)
        pf = cfg_get(config, "project", "protected_files", default=["PLAN.md"])
        if isinstance(pf, list):
            pf = list(pf)
        else:
            pf = pf.split(":")
        for rel in _pod_installed_files():
            if rel not in pf:
                pf.append(rel)
        env["POD_PROTECTED_FILES"] = ":".join(pf)
    result = subprocess.run(
        [script] + args.coordination_args,
        cwd=str(PROJECT_DIR), env=env,
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="pod",
        description="pod — multi-agent manager",
    )
    sub = parser.add_subparsers(dest="command")

    # Subcommands that don't require ensure_config()
    p_init = sub.add_parser("init", help="Bootstrap .pod/ in current git repo")
    p_init.add_argument("--force", action="store_true",
                         help="Overwrite existing config.toml")

    sub.add_parser("update", help="Re-populate .pod/claude-config/ from package")

    p_coord = sub.add_parser("coordination",
                              help="Run bundled coordination script")
    p_coord.add_argument("coordination_args", nargs=argparse.REMAINDER,
                          help="Arguments passed to coordination")

    # Subcommands that require an existing .pod/config.toml
    sub.add_parser("list", help="Show running agents")

    p_add = sub.add_parser("add", help="Launch new agents")
    p_add.add_argument("count", type=int, nargs="?", default=1,
                        help="Number of agents to launch (default: 1)")

    p_target = sub.add_parser("target", help="Set target agent count")
    p_target.add_argument("count", type=int, help="Target number of agents")

    p_finish = sub.add_parser("finish", help="Signal agent to finish current work")
    p_finish.add_argument("target", help="Agent ID prefix or 'all'")

    p_kill = sub.add_parser("kill", help="Kill agent immediately")
    p_kill.add_argument("target", help="Agent ID prefix or 'all'")

    p_once = sub.add_parser(
        "once",
        help="Launch a one-shot agent against a specific issue, "
             "bypassing the dispatch queue and quota cap.",
    )
    p_once.add_argument("issue", type=int, help="Issue number to work on")
    p_once.add_argument(
        "--type", dest="work_type", default=None,
        help="Worker type to use (e.g. feature, plan, repair). "
             "If omitted, inferred from the issue's labels.",
    )

    sub.add_parser("status", help="Show aggregate status")
    p_cleanup = sub.add_parser("cleanup", help="Remove stale worktrees not owned by any agent")
    p_cleanup.add_argument("--dry-run", "-n", action="store_true",
                            help="Show what would be removed without deleting")
    p_cleanup.add_argument("--force", "-f", action="store_true",
                            help="Skip age threshold (remove all stale worktrees)")

    p_log = sub.add_parser("log", help="Tail agent session output")
    p_log.add_argument("target", nargs="?", default=None,
                        help="Agent ID prefix (default: most recent)")

    p_config = sub.add_parser("config", help="Show configuration")
    p_config.add_argument("--edit", action="store_true",
                           help="Open config in $EDITOR")

    p_accounts = sub.add_parser(
        "accounts",
        help="Inspect / manage Claude account leases (list, release, evict-orphans)",
    )
    p_accounts_sub = p_accounts.add_subparsers(dest="accounts_action")
    p_accounts_sub.add_parser("list",
        help="Show current leases (default action)")
    p_accounts_release = p_accounts_sub.add_parser(
        "release",
        help="Release a lease by label")
    p_accounts_release.add_argument("label",
        help="Account label (e.g. lean-fro)")
    p_accounts_release.add_argument("--force", "-f", action="store_true",
        help="Release even if the owning agent is still live")
    p_accounts_sub.add_parser("evict-orphans",
        help="Reap leases whose owning agent is no longer live")

    p_ghstats = sub.add_parser(
        "gh-stats",
        help="Aggregate .pod/gh-access.log to find which callers burn quota",
    )
    p_ghstats.add_argument("--since", default="1h",
                            help="Window: e.g. 10m, 1h, 24h, 7d (default: 1h)")
    p_ghstats.add_argument("--by", choices=("caller", "path", "bucket", "transport"),
                            default="caller",
                            help="Group rows by this column (default: caller)")
    p_ghstats.add_argument("--top", type=int, default=20,
                            help="Show this many top groups (default: 20)")
    p_ghstats.add_argument("--log",
                            help="Override log path (default: .pod/gh-access.log)")

    # Internal helpers shelled out to from the bundled `coordination` script.
    # Underscore-prefixed names keep them out of casual `--help` browsing.
    p_check = sub.add_parser("_check-provenance",
                              help=argparse.SUPPRESS)
    p_check.add_argument("issue_num", type=int)
    p_check.add_argument("--fresh", action="store_true",
                          help="Bypass cache and re-fetch")

    p_filter = sub.add_parser("_filter-trusted-issues",
                               help=argparse.SUPPRESS)
    p_filter.add_argument("--label", action="append",
                           help="Forwarded to gh issue list (repeatable)")
    p_filter.add_argument("--state", default="open")
    p_filter.add_argument("--limit", type=int, default=50)
    p_filter.add_argument("--json", default="number,title",
                           help="Forwarded to gh issue list")
    p_filter.add_argument("--jq", default=None,
                           help="Applied to filtered JSON array")
    p_filter.add_argument("--include-untrusted", action="store_true",
                           help="Surface untrusted issues with [UNTRUSTED: ...] prefix")

    # Hidden dispatcher used by the `coordination` shim script. Carries
    # the bash script's argv shape: first arg is the subcommand, rest
    # are its args. Logic lives in `pod/coordination.py`.
    p_coordination_dispatch = sub.add_parser(
        "_coordination", help=argparse.SUPPRESS,
    )
    p_coordination_dispatch.add_argument(
        "coord_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Handle subcommands that don't require ensure_config()
    if args.command == "init":
        cmd_init(args)
        return
    elif args.command == "update":
        cmd_update(args)
        return
    elif args.command == "coordination":
        cmd_coordination(args)
        return
    elif args.command == "_coordination":
        from pod.coordination import main as _coord_main
        sys.exit(_coord_main(args.coord_args or []))

    config = ensure_config()

    if args.command is None:
        # No subcommand → TUI
        run_tui(config)
    elif args.command == "list":
        cmd_list(config, args)
    elif args.command == "add":
        cmd_add(config, args)
    elif args.command == "target":
        write_target(args.count)
        # Clear planner advisories so user target takes immediate effect
        PLANNER_TARGET_FILE.unlink(missing_ok=True)
        PLANNER_MIN_QUEUE_FILE.unlink(missing_ok=True)
        print(f"Target set to {args.count} agents (planner advisories cleared).")
    elif args.command == "finish":
        cmd_finish(config, args)
    elif args.command == "kill":
        cmd_kill(config, args)
    elif args.command == "once":
        cmd_once(config, args)
    elif args.command == "status":
        cmd_status(config, args)
    elif args.command == "cleanup":
        cmd_cleanup(config, args)
    elif args.command == "log":
        cmd_log(config, args)
    elif args.command == "config":
        cmd_config(config, args)
    elif args.command == "accounts":
        cmd_accounts(config, args)
    elif args.command == "gh-stats":
        cmd_gh_stats(config, args)
    elif args.command == "_check-provenance":
        cmd_check_provenance(config, args)
    elif args.command == "_filter-trusted-issues":
        cmd_filter_trusted_issues(config, args)


if __name__ == "__main__":
    main()
