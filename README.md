# agent-harness-template

Plantilla de arnés agéntico sobre Claude Code, sin particularidades de ningún
proyecto concreto. Haz fork, sigue [SETUP.md](SETUP.md) y personalízalo para tu
caso. Nace de `ml-harness`, un arnés para un proyecto de ML, del que se extrajo todo
lo reutilizable y se dejó fuera lo específico de ese dominio (base de datos,
subagente de EDA, datos sintéticos).

## Idea

Un agente decide qué hacer, pero **no decide cuándo ha terminado ni si lo ha hecho
bien**. Eso lo deciden comprobaciones deterministas (tests, ruff, un corrector y el
CI). Cada fallo real del agente queda documentado junto al cambio del arnés que lo
corrige, y el propio arnés se mide con tareas de referencia (evals).

## Principios y cómo se implementan

| Principio | Mecanismo | Dónde |
|---|---|---|
| La terminación la deciden tests y CI, no el agente | `just check` es la definición de "hecho"; corrector determinista; pipeline de CI | `CLAUDE.md`, `src/harness/grader.py`, `bitbucket-pipelines.yml` |
| Todo fallo real se documenta con su corrección | Registro fallo → cambio en el arnés | `HARNESS_CHANGELOG.md` |
| El arnés se evalúa con tareas de referencia | Tareas en TOML, runner y tabla pass/fail | `evals/`, `src/harness/runner.py` |
| Permisos acotados | Lista `allow` de recetas concretas de `just`, nunca `just *` | `.claude/settings.json` |
| Las rutas sensibles son de solo lectura | Rutas configurables (no en código) protegidas por un hook (Edit, Write y Bash) | `.claude/protected_paths.json`, `.claude/hooks/block_protected_writes.py` |
| Revisión antes de dar algo por terminado | Subagente `reviewer`, de solo lectura + `just check` | `.claude/agents/reviewer.md` |

## Mapa del repo

```
agent-harness-template/
├── CLAUDE.md                  contexto permanente y definición de "hecho" (con TODOs)
├── SETUP.md                   checklist para personalizar el fork
├── HARNESS_CHANGELOG.md       fallo del arnés -> cambio que lo corrige (vacío)
├── justfile                   setup, lint, test, check, eval, evals
├── bitbucket-pipelines.yml    CI: just check en main y pull requests
├── .claude/
│   ├── settings.json          permisos acotados y registro del hook
│   ├── protected_paths.json   rutas de solo lectura (editar aquí, no en código)
│   ├── agents/
│   │   ├── reviewer.md        revisa cambios antes de darlos por terminados
│   │   └── EXAMPLE-domain-agent.md.example   plantilla para un subagente de dominio
│   └── hooks/block_protected_writes.py
├── src/harness/
│   ├── grader.py               comprobaciones deterministas de un eval
│   └── runner.py                lanza Claude Code en una copia limpia y puntúa
├── evals/tasks/                 dos tareas de referencia genéricas (TOML)
├── reports/                     salida de subagentes de dominio (ignorada por git)
├── protected/                   carpeta de ejemplo, de solo lectura
└── tests/                       tests del hook, el corrector y el runner
```

## Cómo se ejecuta un eval

```
tarea (TOML) -> runner -> copia limpia del último commit + uv sync
                       -> claude -p (carga CLAUDE.md, hooks y subagentes)
                       -> corrector determinista
                       -> informe JSON (resultado, duración, coste estimado)
```

Comprobaciones disponibles en cada tarea:

- `just_check`: `just check` sale en verde.
- `files_exist`: existen los archivos pedidos.
- `protected`: ninguna ruta protegida cambia respecto al commit base.
- `tests_preserved`: no se modifica ni borra ningún test existente, solo se pueden añadir.
- `only_changed`: todo cambio respecto al commit base está dentro de las rutas permitidas.
- `report_contains`: el informe existe y menciona los términos esperados.

## Uso

Requisitos: Python 3.11, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just)
y git. Para los evals, además, la CLI de Claude Code en el PATH y con sesión iniciada.

```
just setup                                      # instala dependencias
just check                                      # ruff + pytest
just eval evals/tasks/001_add_utility.toml      # una tarea de referencia
just evals                                      # todas, con tabla final
```

Trabajo interactivo: abre la carpeta en VS Code y usa el panel de Claude Code. El
`CLAUDE.md`, los permisos, el hook y los subagentes se cargan solos. Antes de dar
por terminada una tarea que cambie código, el agente invoca al `reviewer` y ejecuta
`just check`.

## Qué NO trae esta plantilla (y por qué)

- **Acceso a datos.** ml-harness tenía una base DuckDB y un subagente `data-explorer`
  restringido a `just query`. Es un patrón de dominio, no de arnés: se documenta
  como plantilla inerte en `.claude/agents/EXAMPLE-domain-agent.md.example` y en
  `SETUP.md`, para que lo actives solo si tu proyecto lo necesita.
- **Reglas de dominio en `reviewer`** (p. ej. fuga de datos en ML). Añádelas tú en
  la sección marcada en `.claude/agents/reviewer.md`.

## Límites conocidos

- La protección de rutas por terminal (Bash) es una heurística sobre el texto del
  comando: bloquea cualquier comando que mencione una ruta protegida y no sea una
  lectura simple (hay falsos positivos), y no detecta un script que escriba ahí sin
  nombrar la ruta en el propio comando.
- Los subagentes conservan `Bash`, y cualquier restricción a comandos concretos vive
  en su prompt, no en permisos. La regla `deny` de `.claude/settings.json` protege
  archivos concretos frente a Edit/Write; el resto depende del hook y del prompt.
- El corrector (`grader.py`) no ve archivos ignorados por `.gitignore`.
- No se comprueba de forma determinista si el agente invocó al `reviewer`.

## Siguientes pasos (por proyecto, no de la plantilla)

Ver `SETUP.md`. Una vez personalizado, cada proyecto acumulará sus propias tareas de
referencia, reglas de dominio y entradas en `HARNESS_CHANGELOG.md`.
