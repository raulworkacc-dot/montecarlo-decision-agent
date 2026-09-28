"""Hook PreToolUse: bloquea escrituras en las rutas protegidas del proyecto.

Lee las rutas protegidas de `.claude/protected_paths.json` (lista de rutas
relativas a la raiz del repo). Para proteger o liberar una ruta, edita ese
archivo: no hace falta tocar este script.

Lee el JSON del evento por stdin. Exit 2 bloquea la accion y devuelve el
mensaje de stderr al agente. Exit 0 la permite.

Cubre dos vias:
- Herramientas de edicion (Edit, Write, MultiEdit, NotebookEdit): comprueba la
  ruta del archivo.
- Bash: heuristica sobre el texto del comando. No es hermetica: un script que
  escriba en una ruta protegida sin nombrarla en el comando no se detecta.
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
    """Combina las rutas en un patron: cada segmento separado por / o \\."""
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
    """True si el comando menciona una ruta protegida y no es una lectura simple."""
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
            "block_protected_writes: entrada ilegible, se bloquea por seguridad.",
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
            "Bloqueado: el comando toca una ruta protegida "
            f"({', '.join(protected_paths)}). Usa comandos simples de lectura "
            "(ls, cat) o una receta de just.",
            file=sys.stderr,
        )
        return 2

    if not file_path:
        return 0
    if is_protected(file_path, project_dir, protected_paths):
        print(
            f"Bloqueado: intento de escribir en una ruta protegida ({file_path}). "
            "Edita .claude/protected_paths.json si esa ruta ya no debería estar "
            "protegida.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
