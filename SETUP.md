# Cómo personalizar este arnés

Esto es una plantilla, no un proyecto listo para usar. Haz fork (o copia esta
carpeta) y sigue esta lista antes de pedirle tareas reales al agente.

## 1. Nombre del proyecto y del paquete
- [ ] Renombra `src/harness/` a `src/<tu_paquete>/`.
- [ ] Cambia `name = "agent-harness-template"` en `pyproject.toml`.
- [ ] Sustituye `harness.` por `<tu_paquete>.` en: `justfile` (recetas `eval`/`evals`),
      `src/<tu_paquete>/runner.py` (import de `grader`), `tests/test_runner.py`,
      `tests/test_smoke.py`.
- [ ] `tests/test_smoke.py` importa el paquete por su nombre: actualízalo.

## 2. CLAUDE.md
- [ ] Rellena los bloques marcados con TODO: qué hace el proyecto, stack real,
      sistema operativo, CI.
- [ ] Añade las recetas `just` propias del proyecto y cualquier regla de dominio.

## 3. Rutas protegidas
- [ ] Sustituye la carpeta `protected/` de ejemplo por las rutas reales que el
      agente no debe tocar (datos crudos, migraciones ya aplicadas, secretos...).
- [ ] Edita `.claude/protected_paths.json` con esas rutas — el hook
      (`.claude/hooks/block_protected_writes.py`) y el `reviewer` ya las leen de
      ahí, no hace falta tocar código.
- [ ] Si hay archivos binarios sueltos (no una carpeta) que tampoco debe editar,
      añade una regla `deny` explícita en `.claude/settings.json`
      (ej. `"Edit(data/warehouse.duckdb)"`).
- [ ] Actualiza `evals/tasks/002_protected_path_trap.toml` (o bórralo) para que la
      trampa apunte a tu ruta protegida real.

## 4. Acceso a datos u otros recursos (opcional)
Si el proyecto necesita leer una base de datos, una API u otro recurso sensible,
sigue el patrón documentado en
`.claude/agents/EXAMPLE-domain-agent.md.example`:
- [ ] Expón el acceso solo a través de una receta `just` de solo lectura.
- [ ] Añade esa receta concreta (no `just *`) a `permissions.allow` en
      `.claude/settings.json`.
- [ ] Crea el subagente restringido a esa receta y a la carpeta donde escribe su
      informe; quita el sufijo `.example` o crea uno nuevo a partir de la plantilla.
- [ ] Documenta el patrón en `CLAUDE.md`.

## 5. CI
- [ ] Ajusta `bitbucket-pipelines.yml` (o sustitúyelo por el CI que uses) para que
      instale las dependencias reales del proyecto.

## 6. Primer `just check`
- [ ] `just setup && just check` en verde antes de la primera tarea real.
- [ ] `just evals` (requiere la CLI de Claude Code en el PATH y sesión iniciada)
      para confirmar que el arnés en sí funciona sobre las dos tareas de ejemplo.

## 7. Limpieza
- [ ] Borra este archivo, o los apartados ya completados, cuando termines.
- [ ] Deja `HARNESS_CHANGELOG.md` vacío (solo la cabecera) al empezar.
