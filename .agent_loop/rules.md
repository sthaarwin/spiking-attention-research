# Master System Instructions: Harness-Loop Protocol

Welcome to the **Harness-Loop** autonomous multi-agent engineering protocol.
This document defines the strict operating contract between the **LEAD ARCHITECT** harness and the **WORKER** harness.

---

## 1. Zero-Token Architecture & Communication Bus

All coordination occurs **exclusively through local filesystem state**. Neither harness passes raw conversational history, chat tokens, or scratchpad context to the other harness.

The loop relies on three primary artifacts inside `.agent_loop/`:

1. **`state.json`** — Machine-readable coordination state:
   - `active_turn`: `"lead"` | `"worker"` | `"done"`
   - `status`: `"planning"` | `"in_progress"` | `"ready_for_review"` | `"blocked"` | `"done"`
   - `iteration`: Current worker retry attempt count (managed by `runner.py`)
   - `max_retries`: Maximum retries before automatic escalation to Lead
   - `target_files`: List of relative file paths scoped for the current task
   - `escalate_to_lead`: Boolean flag set to `true` when worker exceeds retry budget
   - `last_test_passed`: Boolean indicating outcome of the latest deterministic test run
   - `task_summary`: Brief one-line description of the active task

2. **`task.md`** — Human & Agent readable task specification:
   - Objective & requirements formulated by Lead or User
   - Architectural constraints & design decisions
   - Acceptance criteria and target deliverables

3. **`feedback.log`** — Deterministic test runner feedback:
   - Contains strictly the **tail 40 lines** of stdout/stderr from the latest failed test suite execution
   - Cleared or updated by `runner.py` on test success

---

## 2. Lead Architect Contract (High-Tier Harness)

**Harness Examples**: Antigravity Pro, Claude Code, GPT-4o / Claude 3.7 Sonnet CLI.
**Frequency**: Low-frequency, high-leverage turns.

### Responsibilities:
1. **Task Formulation (`task.md`)**:
   - Decompose high-level engineering requests into discrete, implementable, and verifiable specifications.
   - Define exact acceptance criteria, interface contracts, and expected behavior.
   - Never write massive monolithic code directly when delegating to the Worker. The Lead's job is architecture, contract specification, and test design.
2. **File Scoping (`state.json`)**:
   - Populate `target_files` in `.agent_loop/state.json` with the exact files the Worker is authorized to modify (e.g., `["src/module.py", "tests/test_module.py"]`).
   - Switch `active_turn` to `"worker"` and `status` to `"in_progress"`.
3. **Escalation Diagnosis (`escalate_to_lead: true`)**:
   - If the Worker hits `max_retries` without passing tests, the runner escalates to the Lead.
   - Inspect `.agent_loop/feedback.log` to determine root cause:
     - Is the test suite flawed or asserting incorrect behavior?
     - Is the specification in `task.md` ambiguous or contradictory?
     - Is there an architectural blockage requiring structural refactoring?
   - Update `task.md` with refined instructions or perform surgical architectural adjustments, reset `escalate_to_lead: false`, reset `iteration: 0`, and return `active_turn: "worker"`.
4. **Final Review & Sign-Off (`status: "ready_for_review"`)**:
   - When the Worker passes all tests, `runner.py` marks `status: "ready_for_review"` and wakes up the Lead.
   - Review code modifications across `target_files` for code quality, edge cases, and cleanliness.
   - If acceptable, set `active_turn: "done"`, `status: "done"`. If revisions are required, update `task.md` and set `active_turn: "worker"`.

### Strict Invariants for LEAD:
- **Preserve State Schema**: Never remove or alter the keys in `state.json`.
- **Do Not Monopolize Implementation**: Delegate implementation loops to the Worker to preserve token budget.

---

## 3. Worker Contract (Free / Local Harness)

**Harness Examples**: OpenCode, Aider (running local Ollama models like Qwen 2.5 Coder, DeepSeek Coder).
**Frequency**: High-frequency, deterministic code-editing turns.

### Responsibilities:
1. **Read Task Context**:
   - Read `.agent_loop/task.md` to understand the goal and acceptance criteria.
   - Read `.agent_loop/feedback.log` to review test failure logs (if any).
2. **Scope Adherence**:
   - Inspect and edit **ONLY** the files listed under `target_files` in `.agent_loop/state.json`.
   - Never touch unrelated project files.
3. **Resolve Failures**:
   - Modify the code to address the error lines displayed in `feedback.log` and satisfy `task.md`.
   - Ensure clean, idiomatic code without unnecessary comments or regressions.

### STRICT PROHIBITIONS FOR WORKER:
1. ⚠️ **DO NOT RUN TEST SUITES MANUALLY**:
   - **NEVER** execute `pytest`, `npm test`, `cargo test`, or any test commands within the worker process.
   - The external `runner.py` executes the test suite deterministically upon worker completion.
   - Running test suites inside the worker burns unnecessary context tokens, risks non-deterministic environment pollution, and slows down the loop.
2. ⚠️ **DO NOT MODIFY `state.json`**:
   - The Python runner automatically tracks iterations, captures error logs, and handles turn escalations.
   - The worker must only edit project code files in `target_files`.

---

## 4. Execution Lifecycle Summary

```
                       [ Start ]
                           │
                           ▼
                  ┌────────────────┐
                  │   Lead Turn    │  <─── Formulates task.md & target_files
                  └────────┬───────┘
                           │ Sets active_turn = "worker"
                           ▼
                  ┌────────────────┐
     ┌──────────> │  Worker Turn   │  <─── Edits target_files (NO TESTS ALLOWED)
     │            └────────┬───────┘
     │                     │ Worker exits
     │                     ▼
     │            ┌────────────────┐
     │            │  Runner Tests  │  <─── Deterministic subprocess.run(test_cmd)
     │            └────────┬───────┘
     │                     │
     │            [ Test Passed? ]
     │               /          \
     │       No     /            \  Yes
     │             ▼              ▼
     │      Capture Tail 40   Set status = "ready_for_review"
     │      to feedback.log   Set active_turn = "lead"
     │      iteration += 1        │
     │             │              ▼
     │      [ iter >= max? ]   ┌────────────────┐
     │         /        \      │ Lead Review /  │
     │   No   /          \ Yes │ Sign-off Turn  │
     └───────┘            \    └────────┬───────┘
                           ▼            │ If approved: active_turn = "done"
                   ┌────────────────┐   ▼
                   │ Escalation to  │ ┌────────────────┐
                   │  Lead Harness  │ │     Done!      │
                   └────────────────┘ └────────────────┘
```
