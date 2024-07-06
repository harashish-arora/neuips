# Agent Handoff Protocol

> **Purpose:** Ensure seamless continuity between agent sessions. Every agent MUST read this file and the latest handoff note before doing any work.

---

## How It Works

1. **Starting a session:** Read this file, then `STATUS.md`, then the latest handoff in `handoffs/`.
2. **During a session:** Update the relevant `progress.md` in the phase folder as you complete sub-tasks.
3. **Ending a session:** Write a new handoff note to `handoffs/handoff_NNN.md` using the template below, then update `STATUS.md`.

## Handoff Note Template

```markdown
# Handoff NNN — [Date]

## Session Summary
What was accomplished in this session (bullet points).

## Current State
- Phase: X (substep Y.Z)
- Key files created/modified: [list]
- Any running processes or pending operations: [list or "none"]

## What Worked
Approaches or decisions that proved correct.

## What Didn't Work
Approaches that were tried and abandoned, with reasons.

## Unresolved Issues
Things that need investigation or decisions.

## Next Steps (Prioritized)
1. First thing the next agent should do
2. Second thing
3. ...

## Important Context
Anything the next agent needs to know that isn't captured elsewhere.
```

## Handoff Log

| # | Date | Agent | Phase | Summary |
|---|------|-------|-------|---------|
| 000 | 2026-04-14 | Setup | Pre-1 | Created planning infrastructure, folder structure, handoff protocol |
| 001 | 2026-04-14 | Main | Phase 5 | Multimodality analysis, metric justification (PS-RMSE, Z-RMSE, MAPE rejection) |
