#!/usr/bin/env python3
"""
harness-loop: Autonomous Multi-Agent Software Engineering Loop
Deterministic, zero-token coordination engine between Worker and Lead harnesses.
Standard Library Python 3 only (no external dependencies required).
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Base paths
ROOT_DIR = Path(__file__).resolve().parent
LOOP_DIR = ROOT_DIR / ".agent_loop"
CONFIG_FILE = LOOP_DIR / "config.json"
CONFIG_EXAMPLE_FILE = LOOP_DIR / "config.example.json"
STATE_FILE = LOOP_DIR / "state.json"
TASK_FILE = LOOP_DIR / "task.md"
FEEDBACK_FILE = LOOP_DIR / "feedback.log"
RULES_FILE = LOOP_DIR / "rules.md"
CURRENT_PROMPT_FILE = LOOP_DIR / ".current_prompt.txt"

# Default state structure
DEFAULT_STATE: Dict[str, Any] = {
    "active_turn": "lead",
    "status": "planning",
    "iteration": 0,
    "max_retries": 3,
    "target_files": [],
    "escalate_to_lead": False,
    "last_test_passed": False,
    "task_summary": "",
}

# Default models for known harnesses when 'none' or default is selected
DEFAULT_HARNESS_MODELS: Dict[str, str] = {
    "opencode": "deepseek-coder:6.7b",
    "antigravity pro": "gemini-3.8-flash",
    "antigravity": "gemini-3.8-flash",
    "agy": "gemini-3.8-flash",
    "claude code": "claude-3-7-sonnet",
    "claude": "claude-3-7-sonnet",
    "aider": "ollama/qwen2.5-coder:7b",
}


def resolve_model_name(
    user_input: Optional[str],
    preset_default: str = "",
    previous_value: Optional[str] = None,
    harness_name: Optional[str] = None,
) -> str:
    """
    Resolve model name. When user inputs 'none', 'None', '', 'null', or 'default',
    it falls back to:
    1. Previous value from existing config (if valid and not 'none')
    2. Preset default model (if valid and not 'none')
    3. Known default model for that harness
    """
    raw = (str(user_input) if user_input is not None else "").strip()
    is_none = not raw or raw.lower() in ("none", "null", "default", "nil", "unset")

    if not is_none:
        return raw

    # 1. Fallback to previously used model
    if previous_value:
        prev_clean = str(previous_value).strip()
        if prev_clean and prev_clean.lower() not in ("none", "null", "default", "nil", "unset"):
            return prev_clean

    # 2. Fallback to preset default
    if preset_default:
        preset_clean = str(preset_default).strip()
        if preset_clean and preset_clean.lower() not in ("none", "null", "default", "nil", "unset"):
            return preset_clean

    # 3. Fallback to harness default based on name
    if harness_name:
        h_lower = harness_name.lower()
        for k, v in DEFAULT_HARNESS_MODELS.items():
            if k in h_lower:
                return v

    return "default"


# ANSI Colors for terminal output
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        cls.CYAN = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.MAGENTA = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RESET = ""

if not sys.stdout.isatty():
    Colors.disable()


def log_header(title: str, color: str = Colors.CYAN) -> None:
    line = "━" * 68
    print(f"\n{color}{Colors.BOLD}{line}")
    print(f"  {title}")
    print(f"{line}{Colors.RESET}")


def log_info(msg: str) -> None:
    print(f"{Colors.CYAN}[INFO]{Colors.RESET} {msg}")


def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {msg}")


def log_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")


def log_error(msg: str) -> None:
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")


def atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Safely write JSON using atomic replace to prevent race conditions or partial reads."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile("w", dir=file_path.parent, delete=False, suffix=".tmp", encoding="utf-8")
    try:
        json.dump(data, temp_file, indent=2)
        temp_file.write("\n")
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        Path(temp_file.name).replace(file_path)
    except Exception:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise


def atomic_write_text(file_path: Path, content: str) -> None:
    """Safely write text using atomic replace."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile("w", dir=file_path.parent, delete=False, suffix=".tmp", encoding="utf-8")
    try:
        temp_file.write(content)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        Path(temp_file.name).replace(file_path)
    except Exception:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise


def load_config() -> Dict[str, Any]:
    """Load config.json, or fallback to config.example.json if config.json is absent."""
    if not CONFIG_FILE.exists():
        if CONFIG_EXAMPLE_FILE.exists():
            log_warn(f"Configuration file not found at {CONFIG_FILE}. Run `python runner.py --setup` to configure.")
        else:
            log_error(f"Missing configuration and example file at {CONFIG_FILE}")
        sys.exit(1)

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Failed to read {CONFIG_FILE}: {e}")
        sys.exit(1)


def load_state() -> Dict[str, Any]:
    """Load or initialize state.json atomically."""
    if not STATE_FILE.exists():
        state = DEFAULT_STATE.copy()
        atomic_write_json(STATE_FILE, state)
        return state

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure default keys exist
        for k, v in DEFAULT_STATE.items():
            if k not in data:
                data[k] = v
        return data
    except Exception as e:
        log_warn(f"Corrupt state.json detected ({e}). Resetting to default state.")
        state = DEFAULT_STATE.copy()
        atomic_write_json(STATE_FILE, state)
        return state


def update_state(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Apply partial updates to state.json atomically."""
    state = load_state()
    state.update(updates)
    atomic_write_json(STATE_FILE, state)
    return state


def detect_test_command() -> str:
    """Heuristic helper to auto-detect the default test runner in the repository."""
    if (ROOT_DIR / "Cargo.toml").exists():
        return "cargo test"
    if (ROOT_DIR / "package.json").exists():
        return "npm test"
    if (ROOT_DIR / "pytest.ini").exists() or (ROOT_DIR / "pyproject.toml").exists() or (ROOT_DIR / "tests").exists():
        return "pytest"
    return "pytest"


def build_command_args(
    template: str,
    variables: Dict[str, str],
) -> Tuple[List[str], bool, str]:
    """
    Safely resolve command template placeholders: {prompt}, {session}, {model}, {dir}.
    Returns (argv_list, use_shell, raw_command_str).
    If shell piping or redirection is detected, use_shell is set to True.
    """
    # Replace non-prompt variables first
    resolved_template = template
    for var in ("model", "session", "dir"):
        resolved_template = resolved_template.replace(f"{{{var}}}", variables.get(var, ""))

    prompt_value = variables.get("prompt", "")

    # Check for shell operators in template
    shell_operators = ["|", "&&", "||", ">", "<", ";", "$("]
    needs_shell = any(op in resolved_template for op in shell_operators)

    if needs_shell:
        # If shell is required, safely escape prompt and variables using shlex.quote
        quoted_vars = {k: shlex.quote(v) for k, v in variables.items()}
        cmd_str = template
        for k, qv in quoted_vars.items():
            cmd_str = cmd_str.replace(f"{{{k}}}", qv)
        return [], True, cmd_str

    # Safe tokenization via shlex.split
    # Split the template structure first
    try:
        tokens = shlex.split(resolved_template)
    except ValueError:
        # Fallback to direct split if quoting in template is irregular
        tokens = resolved_template.split()

    argv: List[str] = []
    for token in tokens:
        if "{prompt}" in token:
            argv.append(token.replace("{prompt}", prompt_value))
        else:
            argv.append(token)

    return argv, False, " ".join(shlex.quote(arg) for arg in argv)


def execute_harness(
    role: str,
    harness_config: Dict[str, Any],
    prompt: str,
    session_id: str,
    project_dir: Path,
) -> int:
    """
    Execute a harness command (Lead or Worker) with safe argument passing.
    Logs prompt to .agent_loop/.current_prompt.txt for zero-token auditing.
    """
    name = harness_config.get("name", role.capitalize())
    cmd_template = harness_config.get("command", "")
    raw_model = harness_config.get("model", "")
    model = resolve_model_name(raw_model, harness_name=name)
    timeout = harness_config.get("timeout_seconds", 1200)

    if not cmd_template:
        log_error(f"No command template defined for role '{role}' in config.json")
        return 1

    # Audit log of the prompt
    atomic_write_text(CURRENT_PROMPT_FILE, prompt)

    variables = {
        "prompt": prompt,
        "model": model,
        "session": session_id,
        "dir": str(project_dir),
    }

    argv, use_shell, cmd_str = build_command_args(cmd_template, variables)

    log_info(f"Triggering {Colors.BOLD}{name}{Colors.RESET} ({role.upper()})")
    log_info(f"Target directory: {project_dir}")
    if model:
        log_info(f"Engine model: {model}")

    start_time = time.time()
    try:
        if use_shell:
            process = subprocess.run(
                cmd_str,
                shell=True,
                cwd=str(project_dir),
                timeout=timeout,
            )
        else:
            process = subprocess.run(
                argv,
                shell=False,
                cwd=str(project_dir),
                timeout=timeout,
            )
        elapsed = time.time() - start_time
        log_info(f"{name} exited with code {process.returncode} in {elapsed:.1f}s")
        return process.returncode
    except FileNotFoundError as e:
        log_error(f"Harness executable not found: {e}")
        log_error(f"Please ensure {name} is installed and available in your PATH.")
        return 127
    except subprocess.TimeoutExpired:
        log_error(f"Harness {name} timed out after {timeout} seconds.")
        return 124
    except Exception as e:
        log_error(f"Execution failed for {name}: {e}")
        return 1


def run_deterministic_tests(
    test_command: str,
    project_dir: Path,
    tail_lines_count: int = 40,
) -> Tuple[bool, str]:
    """
    Execute project test suite deterministically.
    On failure: strictly capture the tail N lines of combined output into feedback.log.
    On pass: record success and clear error feedback.
    """
    log_header("RUNNER DETERMINISTIC TEST EXECUTION", Colors.MAGENTA)
    log_info(f"Executing test command: {Colors.BOLD}{test_command}{Colors.RESET}")

    start_time = time.time()
    result = subprocess.run(
        test_command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(project_dir),
    )
    elapsed = time.time() - start_time

    combined_output = (result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")
    lines = combined_output.rstrip().splitlines()

    if result.returncode == 0:
        log_success(f"All tests passed successfully in {elapsed:.2f}s!")
        feedback_content = f"TEST SUITE PASSED ({elapsed:.2f}s)\n"
        atomic_write_text(FEEDBACK_FILE, feedback_content)
        return True, ""
    else:
        # Extract strictly tail N lines
        tail_slice = lines[-tail_lines_count:] if len(lines) > tail_lines_count else lines
        tail_text = "\n".join(tail_slice)
        log_error(f"Tests failed with exit code {result.returncode} in {elapsed:.2f}s.")
        log_warn(f"Clipping strictly tail {len(tail_slice)} error lines into {FEEDBACK_FILE}")

        feedback_header = (
            f"=== TEST RUNNER FAILURE LOG (Tail {len(tail_slice)} lines) ===\n"
            f"Command: {test_command}\n"
            f"Exit Code: {result.returncode}\n"
            f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"===========================================================\n"
        )
        atomic_write_text(FEEDBACK_FILE, feedback_header + tail_text + "\n")
        return False, tail_text


def construct_lead_prompt(state: Dict[str, Any]) -> str:
    """Construct minimal zero-token prompt guiding the Lead harness."""
    status = state.get("status", "planning")
    escalated = state.get("escalate_to_lead", False)
    target_files = state.get("target_files", [])

    prompt_lines = [
        "You are the LEAD ARCHITECT in the harness-loop multi-agent engineering system.",
        "Read .agent_loop/rules.md, .agent_loop/task.md, and .agent_loop/state.json.",
    ]

    if escalated:
        prompt_lines.extend([
            "ESCALATION ALERT: The worker harness exceeded its retry budget without passing tests.",
            "Read .agent_loop/feedback.log to inspect the test failures.",
            "Diagnose the root cause: refine .agent_loop/task.md, adjust architecture, or correct the test contract.",
            "In .agent_loop/state.json, reset 'escalate_to_lead': false, set 'iteration': 0, and set 'active_turn': 'worker'.",
        ])
    elif status == "ready_for_review":
        prompt_lines.extend([
            "REVIEW REQUIRED: All automated tests have passed!",
            f"Inspect the target files: {target_files}",
            "Verify code quality, edge cases, and adherence to acceptance criteria in .agent_loop/task.md.",
            "If complete and satisfactory, set 'active_turn': 'done', 'status': 'done' in .agent_loop/state.json.",
            "If revisions are needed, update .agent_loop/task.md, and set 'active_turn': 'worker', 'status': 'in_progress'.",
        ])
    else:
        prompt_lines.extend([
            "Formulate or refine the task specification in .agent_loop/task.md with clear acceptance criteria.",
            "Populate 'target_files' in .agent_loop/state.json with the files authorized for modification.",
            "Set 'active_turn': 'worker' and 'status': 'in_progress' in .agent_loop/state.json.",
            "DO NOT write large monolithic implementation code yourself; delegate implementation to the Worker.",
        ])

    return "\n".join(prompt_lines)


def construct_worker_prompt(state: Dict[str, Any]) -> str:
    """Construct minimal prompt guiding the Worker harness."""
    iteration = state.get("iteration", 0)
    max_retries = state.get("max_retries", 3)
    target_files = state.get("target_files", [])

    prompt_lines = [
        "You are the WORKER in the harness-loop multi-agent engineering system.",
        f"Current cycle: Iteration {iteration + 1} of {max_retries}.",
        "Read .agent_loop/rules.md and .agent_loop/task.md.",
        f"You are authorized to modify ONLY these target files: {target_files}",
    ]

    if FEEDBACK_FILE.exists() and FEEDBACK_FILE.stat().st_size > 0:
        prompt_lines.extend([
            "Read .agent_loop/feedback.log for the latest test failure errors.",
            "Edit the target files to fix the reported errors and satisfy task.md.",
        ])
    else:
        prompt_lines.append("Implement the code specified in .agent_loop/task.md.")

    prompt_lines.extend([
        "CRITICAL RULE: DO NOT RUN TEST SUITES (e.g. pytest, npm test, cargo test) MANUALLY.",
        "The external runner executes tests deterministically after your turn concludes.",
        "DO NOT modify .agent_loop/state.json.",
    ])

    return "\n".join(prompt_lines)


def ensure_workspace_initialized() -> None:
    """Ensure directory structure and initial files exist."""
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "src").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "tests").mkdir(parents=True, exist_ok=True)

    if not TASK_FILE.exists():
        default_task = (
            "# Task Specification\n\n"
            "## Objective\n"
            "Define the objective or feature to implement here.\n\n"
            "## Acceptance Criteria\n"
            "- [ ] Unit tests pass cleanly\n"
            "- [ ] Code adheres to project conventions\n"
        )
        atomic_write_text(TASK_FILE, default_task)


def step_cycle(config: Dict[str, Any]) -> bool:
    """
    Execute a single turn in the harness loop.
    Returns True if the loop should continue, False if done or halted.
    """
    state = load_state()
    active_turn = state.get("active_turn", "lead")
    status = state.get("status", "planning")
    iteration = state.get("iteration", 0)
    runner_cfg = config.get("runner", {})
    max_free_retries = runner_cfg.get("max_free_retries", 3)
    test_command = runner_cfg.get("test_command", "pytest")
    tail_lines = runner_cfg.get("error_tail_lines", 40)
    session_id = f"loop-{int(time.time())}"

    # Check termination condition
    if active_turn == "done" or status == "done":
        log_header("HARNESS LOOP COMPLETE", Colors.GREEN)
        log_success("Active turn is marked 'done'. All requirements satisfied!")
        return False

    # Lead Turn
    if active_turn == "lead":
        lead_cfg = config.get("lead", {})
        log_header(f"LEAD TURN [{lead_cfg.get('name', 'Lead')}]", Colors.CYAN)
        prompt = construct_lead_prompt(state)
        exit_code = execute_harness("lead", lead_cfg, prompt, session_id, ROOT_DIR)
        if exit_code != 0:
            log_error(f"Lead harness exited with non-zero code {exit_code}")

        # Reload state after Lead update
        post_state = load_state()
        new_turn = post_state.get("active_turn", "lead")
        if new_turn == "lead" and post_state.get("status") == "planning":
            log_warn("Lead did not switch active_turn to 'worker'. Automatic switch enabled if target_files present.")
            if post_state.get("target_files"):
                update_state({"active_turn": "worker", "status": "in_progress"})
        return True

    # Worker Turn
    elif active_turn == "worker":
        worker_cfg = config.get("worker", {})
        log_header(
            f"WORKER TURN [{worker_cfg.get('name', 'Worker')}] — Iteration {iteration + 1}/{max_free_retries}",
            Colors.YELLOW,
        )
        prompt = construct_worker_prompt(state)
        exit_code = execute_harness("worker", worker_cfg, prompt, session_id, ROOT_DIR)
        if exit_code != 0:
            log_warn(f"Worker harness exited with code {exit_code}")

        # Deterministic Runner Test Execution
        test_passed, _ = run_deterministic_tests(test_command, ROOT_DIR, tail_lines)

        if test_passed:
            log_success("Tests passed! Updating state to 'ready_for_review' and notifying Lead.")
            update_state({
                "last_test_passed": True,
                "status": "ready_for_review",
                "active_turn": "lead",
                "escalate_to_lead": False,
                "iteration": 0,
            })
        else:
            new_iter = iteration + 1
            if new_iter >= max_free_retries:
                log_header("RETRY BUDGET EXCEEDED — ESCALATING TO LEAD", Colors.RED)
                log_warn(f"Worker reached {new_iter}/{max_free_retries} attempts without passing tests.")
                update_state({
                    "iteration": new_iter,
                    "last_test_passed": False,
                    "escalate_to_lead": True,
                    "active_turn": "lead",
                    "status": "blocked",
                })
            else:
                log_info(f"Retrying worker. Iteration {new_iter}/{max_free_retries} scheduled.")
                update_state({
                    "iteration": new_iter,
                    "last_test_passed": False,
                    "escalate_to_lead": False,
                    "active_turn": "worker",
                    "status": "in_progress",
                })
        return True

    else:
        log_error(f"Unknown active_turn: '{active_turn}' in state.json")
        return False


def run_loop(max_cycles: int = 50) -> None:
    """Run autonomous continuous loop until complete or halted."""
    config = load_config()
    ensure_workspace_initialized()

    log_header("STARTING AUTONOMOUS HARNESS-LOOP", Colors.BOLD)
    log_info(f"Worker: {config.get('worker', {}).get('name')} | Lead: {config.get('lead', {}).get('name')}")
    log_info(f"Test command: {config.get('runner', {}).get('test_command')} | Max retries: {config.get('runner', {}).get('max_free_retries')}")

    cycles = 0
    try:
        while cycles < max_cycles:
            cycles += 1
            log_info(f"--- Loop Cycle {cycles}/{max_cycles} ---")
            should_continue = step_cycle(config)
            if not should_continue:
                break
            time.sleep(1)
        if cycles >= max_cycles:
            log_warn(f"Harness loop reached safety limit of {max_cycles} cycles.")
    except KeyboardInterrupt:
        print("\n")
        log_warn("Harness loop paused by user (SIGINT). State preserved.")


def print_status() -> None:
    """Print current state overview in clean formatted output."""
    ensure_workspace_initialized()
    state = load_state()
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    log_header("HARNESS-LOOP CURRENT STATUS", Colors.CYAN)
    active_turn = state.get("active_turn", "unknown")
    status = state.get("status", "unknown")
    iteration = state.get("iteration", 0)
    max_retries = state.get("max_retries", config.get("runner", {}).get("max_free_retries", 3))
    escalated = state.get("escalate_to_lead", False)
    test_passed = state.get("last_test_passed", False)
    target_files = state.get("target_files", [])

    turn_color = Colors.CYAN if active_turn == "lead" else (Colors.YELLOW if active_turn == "worker" else Colors.GREEN)
    print(f"  {Colors.BOLD}Active Turn:{Colors.RESET}       {turn_color}{active_turn.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Loop Status:{Colors.RESET}       {status}")
    print(f"  {Colors.BOLD}Worker Iteration:{Colors.RESET}  {iteration} / {max_retries}")
    print(f"  {Colors.BOLD}Escalated to Lead:{Colors.RESET} {Colors.RED if escalated else Colors.GREEN}{escalated}{Colors.RESET}")
    print(f"  {Colors.BOLD}Last Test Passed:{Colors.RESET}  {Colors.GREEN if test_passed else Colors.RED}{test_passed}{Colors.RESET}")
    print(f"  {Colors.BOLD}Target Files:{Colors.RESET}      {', '.join(target_files) if target_files else '(none specified)'}")

    if FEEDBACK_FILE.exists():
        feedback_size = FEEDBACK_FILE.stat().st_size
        print(f"  {Colors.BOLD}Feedback Log:{Colors.RESET}      {FEEDBACK_FILE} ({feedback_size} bytes)")
    if TASK_FILE.exists():
        print(f"  {Colors.BOLD}Task Spec:{Colors.RESET}         {TASK_FILE}")
    print()


def reset_state() -> None:
    """Reset state.json and feedback.log back to initial clean state."""
    ensure_workspace_initialized()
    state = DEFAULT_STATE.copy()
    atomic_write_json(STATE_FILE, state)
    if FEEDBACK_FILE.exists():
        atomic_write_text(FEEDBACK_FILE, "")
    log_success("Loop state reset to initial planning state.")
    print_status()


def interactive_setup() -> None:
    """Interactive CLI wizard to configure harnesses, presets, and test commands."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}  _                                     _                     
 | |__   __ _ _ __ _ __   ___  ___ ___ | | ___   ___  _ __   
 | '_ \\ / _` | '__| '_ \\ / _ \\/ __/ __|| |/ _ \\ / _ \\| '_ \\  
 | | | | (_| | |  | | | |  __/\\__ \\__ \\| | (_) | (_) | |_) | 
 |_| |_|\\__,_|_|  |_| |_|\\___||___/___/|_|\\___/ \\___/| .__/  
                                                      |_|     
   Autonomous Multi-Agent Engineering Loop Setup Wizard{Colors.RESET}
"""
    print(banner)
    ensure_workspace_initialized()

    # Load presets from example file if available
    presets: Dict[str, Any] = {}
    if CONFIG_EXAMPLE_FILE.exists():
        try:
            with open(CONFIG_EXAMPLE_FILE, "r", encoding="utf-8") as f:
                ex = json.load(f)
                presets = ex.get("presets", {})
        except Exception:
            pass

    if not presets:
        presets = {
            "1": {
                "name": "OpenCode (Worker) + Antigravity Pro (Lead)",
                "worker": {
                    "name": "OpenCode",
                    "command": "opencode run --model {model} --prompt {prompt}",
                    "model": "deepseek-coder:6.7b",
                },
                "lead": {
                    "name": "Antigravity Pro",
                    "command": "agy run --model {model} {prompt}",
                    "model": "gemini-1.5-pro",
                },
            },
            "2": {
                "name": "Aider (Worker) + Claude Code (Lead)",
                "worker": {
                    "name": "Aider",
                    "command": "aider --model {model} --message {prompt} --no-auto-commits",
                    "model": "ollama/qwen2.5-coder:7b",
                },
                "lead": {
                    "name": "Claude Code",
                    "command": "claude -p {prompt}",
                    "model": "claude-3-7-sonnet",
                },
            },
            "3": {
                "name": "OpenCode (Worker) + Claude Code (Lead)",
                "worker": {
                    "name": "OpenCode",
                    "command": "opencode run --model {model} --prompt {prompt}",
                    "model": "deepseek-coder:6.7b",
                },
                "lead": {
                    "name": "Claude Code",
                    "command": "claude -p {prompt}",
                    "model": "claude-3-7-sonnet",
                },
            },
            "4": {
                "name": "Custom Harness Templates",
                "worker": {
                    "name": "Custom Worker",
                    "command": "worker-cli --prompt {prompt}",
                    "model": "default",
                },
                "lead": {
                    "name": "Custom Lead",
                    "command": "lead-cli --prompt {prompt}",
                    "model": "default",
                },
            },
        }

    print(f"{Colors.BOLD}Select a Multi-Agent Engine Preset:{Colors.RESET}\n")
    for key, p in presets.items():
        desc = p.get("description", "")
        desc_str = f" - {Colors.DIM}{desc}{Colors.RESET}" if desc else ""
        print(f"  [{Colors.CYAN}{key}{Colors.RESET}] {Colors.BOLD}{p.get('name')}{Colors.RESET}{desc_str}")

    preset_choice = input(f"\nEnter preset number [1-{len(presets)}] (default: 1): ").strip() or "1"
    selected_preset = presets.get(preset_choice, presets["1"])

    print(f"\nSelected: {Colors.GREEN}{selected_preset.get('name')}{Colors.RESET}\n")

    # Check if a config already exists to preserve previous choices
    prev_config: Dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                prev_config = json.load(f)
        except Exception:
            pass

    prev_worker_model = prev_config.get("worker", {}).get("model", "")
    prev_lead_model = prev_config.get("lead", {}).get("model", "")

    # Worker details
    worker_data = selected_preset.get("worker", {})
    w_name = worker_data.get("name", "Worker")
    if preset_choice == "4":
        w_name = input("Worker name (e.g., OpenCode, Aider, custom): ").strip() or "Worker"
        w_cmd = input("Worker command template (placeholders: {prompt}, {model}, {dir}): ").strip()
        w_default = resolve_model_name("", previous_value=prev_worker_model, harness_name=w_name)
        w_model_input = input(f"Worker model [{w_default}] (type 'none' for default): ").strip()
        w_model = resolve_model_name(w_model_input, preset_default=w_default, previous_value=prev_worker_model, harness_name=w_name)
        worker_config = {"name": w_name, "command": w_cmd, "model": w_model, "timeout_seconds": 600}
    else:
        preset_w_model = worker_data.get("model", "")
        w_default = resolve_model_name("", preset_default=preset_w_model, previous_value=prev_worker_model, harness_name=w_name)
        w_model_input = input(f"Worker model [{w_default}] (type 'none' for default): ").strip()
        custom_w_model = resolve_model_name(w_model_input, preset_default=w_default, previous_value=prev_worker_model, harness_name=w_name)
        worker_config = {
            "name": worker_data.get("name", "Worker"),
            "command": worker_data.get("command", ""),
            "model": custom_w_model,
            "timeout_seconds": 600,
        }

    # Lead details
    lead_data = selected_preset.get("lead", {})
    l_name = lead_data.get("name", "Lead")
    if preset_choice == "4":
        l_name = input("Lead name (e.g., Antigravity Pro, Claude Code): ").strip() or "Lead"
        l_cmd = input("Lead command template (placeholders: {prompt}, {model}, {dir}): ").strip()
        l_default = resolve_model_name("", previous_value=prev_lead_model, harness_name=l_name)
        l_model_input = input(f"Lead model [{l_default}] (type 'none' for default): ").strip()
        l_model = resolve_model_name(l_model_input, preset_default=l_default, previous_value=prev_lead_model, harness_name=l_name)
        lead_config = {"name": l_name, "command": l_cmd, "model": l_model, "timeout_seconds": 900}
    else:
        preset_l_model = lead_data.get("model", "")
        l_default = resolve_model_name("", preset_default=preset_l_model, previous_value=prev_lead_model, harness_name=l_name)
        l_model_input = input(f"Lead model [{l_default}] (type 'none' for default): ").strip()
        custom_l_model = resolve_model_name(l_model_input, preset_default=l_default, previous_value=prev_lead_model, harness_name=l_name)
        lead_config = {
            "name": lead_data.get("name", "Lead"),
            "command": lead_data.get("command", ""),
            "model": custom_l_model,
            "timeout_seconds": 900,
        }

    # Test runner settings
    detected_test_cmd = detect_test_command()
    test_cmd = input(f"Project test command [{detected_test_cmd}]: ").strip() or detected_test_cmd

    retries_input = input("Max worker retries before lead escalation [3]: ").strip() or "3"
    try:
        max_retries = int(retries_input)
    except ValueError:
        max_retries = 3

    # Generate config dictionary
    config_data = {
        "active_preset": preset_choice,
        "worker": worker_config,
        "lead": lead_config,
        "runner": {
            "test_command": test_cmd,
            "max_free_retries": max_retries,
            "error_tail_lines": 40,
            "project_dir": ".",
        },
        "presets": presets,
    }

    atomic_write_json(CONFIG_FILE, config_data)
    log_success(f"Configuration saved atomically to {CONFIG_FILE}")

    # Initialize state if needed
    if not STATE_FILE.exists():
        state = DEFAULT_STATE.copy()
        state["max_retries"] = max_retries
        atomic_write_json(STATE_FILE, state)

    # Prompt for task specification
    task_desc = input("\nDescribe initial engineering task (or press Enter to edit task.md manually): ").strip()
    if task_desc:
        task_content = (
            f"# Task Specification\n\n"
            f"## Objective\n{task_desc}\n\n"
            f"## Acceptance Criteria\n"
            f"- [ ] Implementation passes `{test_cmd}` cleanly\n"
            f"- [ ] Clean idiomatic code with zero regressions\n"
        )
        atomic_write_text(TASK_FILE, task_content)
        log_success(f"Initial task written to {TASK_FILE}")

    print(f"\n{Colors.GREEN}{Colors.BOLD}Setup complete!{Colors.RESET}")
    print(f"Start the autonomous loop anytime with:\n  {Colors.CYAN}{Colors.BOLD}python runner.py{Colors.RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="harness-loop: Autonomous Multi-Agent Software Engineering Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive configuration wizard for presets and test commands.",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Execute exactly one cycle/turn (Lead or Worker + tests) and exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current loop state, turn, target files, and test status.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset state.json to initial planning state.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Update or initialize the task specification in .agent_loop/task.md.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit target files for the task.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Directly run the project test command and update feedback.log without invoking harnesses.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=50,
        help="Maximum loop iterations before stopping (default: 50).",
    )

    args = parser.parse_args()

    if args.setup:
        interactive_setup()
        return

    if args.status:
        print_status()
        return

    if args.reset:
        reset_state()
        return

    if args.task:
        import re
        ensure_workspace_initialized()
        task_content = (
            f"# Task Specification\n\n"
            f"## Objective\n{args.task}\n\n"
            f"## Acceptance Criteria\n"
            f"- [ ] Tests pass cleanly\n"
            f"- [ ] Lead code review approval\n"
        )
        atomic_write_text(TASK_FILE, task_content)
        log_success(f"Updated task specification in {TASK_FILE}")

        # Extract target files from --files or auto-detect from task text
        target_files = args.files
        if not target_files:
            detected = re.findall(r'[\w\-./]+\.(?:py|yaml|yml|json|md)', args.task)
            target_files = [f for f in detected if not f.startswith(".agent_loop")]

        update_data: Dict[str, Any] = {
            "active_turn": "worker",
            "status": "in_progress",
            "iteration": 0,
            "escalate_to_lead": False,
            "last_test_passed": False,
            "task_summary": args.task[:120],
        }
        if target_files:
            update_data["target_files"] = target_files

        update_state(update_data)
        if FEEDBACK_FILE.exists():
            atomic_write_text(FEEDBACK_FILE, "")
        log_success("Set active_turn to 'worker' and initialized task in state.json.")
        print_status()
        return

    if args.test_only:
        config = load_config()
        test_cmd = config.get("runner", {}).get("test_command", "pytest")
        tail_lines = config.get("runner", {}).get("error_tail_lines", 40)
        passed, _ = run_deterministic_tests(test_cmd, ROOT_DIR, tail_lines)
        sys.exit(0 if passed else 1)

    if args.step:
        config = load_config()
        ensure_workspace_initialized()
        step_cycle(config)
        return

    # Default action: run autonomous loop
    if not CONFIG_FILE.exists():
        log_warn("No configuration file found. Launching setup wizard...")
        interactive_setup()
    else:
        run_loop(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
