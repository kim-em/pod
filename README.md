# pod

Multi-agent manager for Claude Code. Launches and monitors concurrent
autonomous Claude sessions, coordinating via GitHub issues, labels, and PRs.

## Quick Start

```bash
# Install
uv tool install git+https://github.com/kim-em/pod.git

# Bootstrap a project
cd your-project
pod init

# Now write your long-term PLAN.md

# Launch the TUI
pod

# Or use CLI commands
pod add 3        # launch 3 agents
pod status       # queue depth, agent count, cost
pod list         # show running agents
```

## How It Works

Pod manages a pool of autonomous Claude Code agents, each running in its
own git worktree on its own branch. Agents coordinate through GitHub
issues and PRs:

- **Planners** create scoped work items as GitHub issues, then exit
- **Workers** claim issues, implement changes, and open PRs
- Auto-merge handles the rest

## Requirements

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`)
- [GitHub CLI](https://cli.github.com/) (`gh`), authenticated
- Git

## Commands

| Command | Description |
|---------|-------------|
| `pod` | Interactive TUI |
| `pod init [--force]` | Bootstrap `.pod/` in current git repo |
| `pod update` | Re-populate agent config from installed package |
| `pod add [N]` | Launch N agents (default 1) |
| `pod list` | Show running agents |
| `pod finish [ID\|all]` | Signal agent(s) to finish after current work |
| `pod kill [ID\|all]` | Kill agent(s) immediately |
| `pod status` | Queue depth, agent count, total cost |
| `pod log [ID]` | Tail agent's session output |
| `pod config [--edit]` | Show or edit configuration |
| `pod coordination ...` | Run bundled coordination script directly |

## Configuration

After `pod init`, edit `.pod/config.toml` to customize:

- **Worker types**: define agent roles (`/plan`, `/feature`, `/review`, etc.)
- **Dispatch strategy**: `queue_balance` or `round_robin`
- **Claude model**: default `opus`
- **Build cache**: directory to rsync into worktrees
- **Protected files**: files agents may not modify in PRs

Agent session config (commands, skills) lives in `.pod/claude-config/`
and is managed by pod -- run `pod update` after upgrading pod to get
the latest agent prompts.

## License

Apache 2.0
