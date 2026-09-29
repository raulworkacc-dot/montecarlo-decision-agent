"""PreToolUse hook: block writes to the project's protected paths.

Protected paths are read from `.claude/protected_paths.json` (a list of paths
relative to the repo root). To protect or release a path, edit that file: this
script does not need to change.

The event JSON arrives on stdin. Exit 2 blocks the action and returns the stderr
message to the agent. Exit 0 allows it.

It covers two routes:
- Editing tools (Edit, Write, MultiEdit, NotebookEdit): the file path is checked.
- Bash: a heuristic on the command text. It is not airtight: a script that writes
  to a protected path without naming it in the command is not detected.
"""

import json
import os
import re
import sys
from pathlib import Path

CONFIG_PATH = Path(".claude") / "protected_paths.json"
SHELL_OPERATORS = (">", "|", ";", "&", "\n", "\r", "$(", "`")
READ_ONLY_COMMANDS = {
    "ls", "dir", "cat", "type", "head", "tail", "wc",
    "get-childitem", "gci", "get-content", "gc", "select-string",
}  # fmt: skip


def load_protected_paths(project_dir: Path) -> list[str]:
    config = project_dir / CONFIG_PATH
    if not config.is_file():
        return []
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item.strip() for item in data if isinstance(item, str) and item.strip()]


def build_pattern(paths: list[str]) -> re.Pattern | None:
    """Combine the paths into one pattern: segments separated by / or \\."""
    alternatives = []
    for path in paths:
        segments = [re.escape(part) for part in re.split(r"[\\/]+", path.strip("/\\")) if part]
        if segments:
            alternatives.append(r"[\\/]+".join(segments))
    if not alternatives:
        return None
    return re.compile(rf"(?:{'|'.join(alternatives)})(?![\w-])", re.IGNORECASE)


def is_protected(file_path: str, project_dir: Path, protected_paths: list[str]) -> bool:
    target = Path(file_path)
    if not target.is_absolute():
        target = project_dir / target
    candidate = os.path.normcase(str(target.resolve()))
    for protected in protected_paths:
        base = os.path.normcase(str((project_dir / protected).resolve()))
        if candidate == base or candidate.startswith(base + os.sep):
            return True
    return False


def bash_touches_protected(command: str, pattern: re.Pattern) -> bool:
    """True if the command mentions a protected path and is not a simple read."""
    if not pattern.search(command):
        return False
    parts = command.strip().split()
    read_only = bool(parts) and parts[0].lower() in READ_ONLY_COMMANDS
    has_operator = any(op in command for op in SHELL_OPERATORS)
    return not read_only or has_operator


def main() -> int:
    try:
        event = json.load(sys.stdin)
        tool_input = event.get("tool_input", {})
        command = tool_input.get("command")
        file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    except (json.JSONDecodeError, AttributeError):
        print(
            "block_protected_writes: unreadable input, blocked for safety.",
            file=sys.stderr,
        )
        return 2

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))
    protected_paths = load_protected_paths(project_dir)
    if not protected_paths:
        return 0

    pattern = build_pattern(protected_paths)
    if isinstance(command, str) and pattern and bash_touches_protected(command, pattern):
        print(
            "Blocked: the command touches a protected path "
            f"({', '.join(protected_paths)}). Use simple read commands "
            "(ls, cat) or a just recipe.",
            file=sys.stderr,
        )
        return 2

    if not file_path:
        return 0
    if is_protected(file_path, project_dir, protected_paths):
        print(
            f"Blocked: attempt to write to a protected path ({file_path}). "
            "A person must edit .claude/protected_paths.json if that path should no "
            "longer be protected.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
