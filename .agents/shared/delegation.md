# Delegation

Shared by every workflow prompt. `AGENTS.md`「Agent Harness」owns *which* role
handles what; this file owns *how* a child is started.

## Mechanisms

The harness is runtime-agnostic; a workflow names the role, not the transport.
Use whichever delegation mechanism the current runtime provides:

| runtime | delegation |
|---|---|
| Claude Code | `Agent` tool (subagent types `Explore` / `general-purpose` / `Plan` / `claude`), plus `SendMessage` / `ListAgents` / `TaskStop` |
| Codex | `collaboration.spawn_agent`, plus `followup_task` / `wait_agent` / `list_agents` / `send_message` / `interrupt_agent` |
| dsh / other | `delegate_to_agent(agent_type, task, working_dir) -> task_id` and `get_delegation_status([task_id, ...], wait_ms)` when the runtime exposes that pair |

Fan out by passing several tasks to one dispatch; collect as soon as one reaches a
terminal state. Do not set a model override; children inherit the current model.
If delegation is unavailable, do the work locally and say so (Rule 12) — never
quietly drop a verification step that the workflow promised to a different party.

## Role files are not wired into any runtime

No runtime auto-discovers `.agents/agents/*.md` as subagent definitions. The
orchestrator reads `.agents/agents/<name>.md` and inlines the role text into the
`task`, together with the concrete task and the repository context.

Three consequences, because all of them bite in practice:

- A child starts cold. It sees no session context, no earlier turn, no file you
  have open. Anything the role needs must be inside the `task` string.
- The `tools:` line in a role file (`read, grep, find, ls, bash` for the
  read-only roles) is **documentation, not enforcement**. Nothing strips write
  access from a child. State "read-only, do not modify any file" in the task,
  and in Claude pick an `Explore`-type subagent.
- A child can stop and wait for a human: a permission prompt, a sandbox denial,
  or a plan approval parks it until someone answers. Long unattended workflows
  must therefore be built out of children that need no approval — keep children
  to read-only recon, evaluation runs and tests, and do the privileged steps in
  the orchestrator, which the user is already watching.

## Why the verifier is a separate agent

`qa-inspector` exists so that the agent that wrote a change is not the agent that
judges it. Delegating a whole workflow to a single child destroys that property:
keep verification a separate delegation from implementation, even when one agent
could do both. `accuracy-analyst` is likewise separate from the developer roles:
an implementation that misreads its own accuracy result cannot be its own judge.
