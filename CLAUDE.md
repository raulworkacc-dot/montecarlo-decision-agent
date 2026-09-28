# {{nombre-del-proyecto}}

<!-- TODO (ver SETUP.md): describe en 2-3 líneas qué hace el proyecto, el dominio y
     el stack real. Este bloque es plantilla: no lo dejes así. -->
Arnés agéntico sobre Claude Code para TODO-describe-el-dominio. Stack: Python 3.11,
uv, just, ruff, pytest. Entorno: TODO-sistema-operativo-local y CI en TODO-CI.

## Comandos (siempre vía just)
- `just setup`: instala dependencias (uv sync)
- `just lint`: ruff check + ruff format --check
- `just test`: pytest
- `just check`: lint + test
- `just eval evals/tasks/<tarea>.toml`: ejecuta una tarea de referencia del arnés
- `just evals`: ejecuta todas las tareas de referencia
<!-- TODO: añade aquí las recetas propias del proyecto (acceso a datos, servicios
     externos...). Si una receta da acceso a un recurso sensible, expónla de una en
     una en permissions.allow (.claude/settings.json); nunca autorices `just *`. -->

Nunca ejecutes ruff o pytest directamente: usa `just`.

## Definición de "hecho"
Una tarea está terminada solo cuando `just check` sale en verde. No declares una
tarea completada sin haberlo ejecutado y sin mostrar su resultado.
Si `just check` falla, corrige la causa; no desactives reglas ni borres tests.
Antes de dar por terminada una tarea que cambie código, invoca al subagente `reviewer`.

## Reglas
- Todo script o hook se escribe en Python. Nunca en bash.
- Todo debe funcionar en Windows y en Linux: rutas con `pathlib`, sin comandos
  específicos de un shell.
- Las rutas protegidas (ver abajo) son de solo lectura: no crear, editar ni borrar
  nada ahí.
- Código en `src/<paquete>/`, tests en `tests/`. Todo código nuevo lleva su test.
- Cambios mínimos: no añadas dependencias ni refactorices lo que no pide la tarea.
- Si falta información para decidir, pregunta en vez de asumir.
<!-- TODO: añade aquí reglas propias del dominio del proyecto. -->

## Rutas protegidas
Las rutas de solo lectura están listadas en `.claude/protected_paths.json` (por
defecto, la carpeta de ejemplo `protected/`). El hook `block_protected_writes.py` y
el subagente `reviewer` las respetan; para proteger o liberar una ruta basta con
editar ese archivo, no hace falta tocar código.
<!-- TODO: si el proyecto tiene una base de datos, una API u otro recurso sensible al
     que el agente solo debe acceder de forma controlada, documenta aquí la vía
     permitida (p. ej. una receta `just query` de solo lectura) y el subagente que
     delega en ella. Ver .claude/agents/EXAMPLE-domain-agent.md.example. -->

## Fallos del arnés
Si el agente comete un error real (incumple una regla, da por hecho algo sin
verificar), se documenta en `HARNESS_CHANGELOG.md` junto al cambio del arnés que
lo corrige.
