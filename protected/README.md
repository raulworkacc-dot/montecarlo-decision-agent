# protected/

Carpeta de ejemplo, de solo lectura para el agente. Demuestra el mecanismo de rutas
protegidas: está listada en `.claude/protected_paths.json`, así que el hook
`block_protected_writes.py` bloquea cualquier intento de crear, editar o borrar algo
aquí (por herramientas de edición o por Bash), y la tarea de referencia
`evals/tasks/002_protected_path_trap.toml` comprueba que el agente respeta esto
incluso cuando la especificación se lo pide.

Sustitúyela por las rutas reales de tu proyecto (datos crudos, migraciones ya
aplicadas, secretos...) y actualiza `.claude/protected_paths.json`. Ver `SETUP.md`.
