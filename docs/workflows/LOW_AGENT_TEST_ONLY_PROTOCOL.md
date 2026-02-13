# Protocolo Estricto: Agente Solo Ejecuta Tests (Sin Modificar Archivos)

Este protocolo es para un agente de menor capacidad.

## Reglas obligatorias

1. No editar ningun archivo (`.py`, `.md`, `.ps1`, etc.).
2. No crear ni borrar archivos.
3. Solo ejecutar comandos de test.
4. Si un test falla, reportar salida y detenerse.
5. No correr `pytest -q` completo salvo instruccion explicita.

## Entorno

```powershell
cd D:\Procastrinar\LMS
.\.venv\Scripts\Activate.ps1
```

## Bloque A (sanidad minima, siempre)

```powershell
python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument
python -m pytest -q tests/test_lms_live_collector_bot.py
```

## Bloque B (entrenamiento por repeticion, 20 corridas)

Objetivo: reforzar estabilidad de parser/flujo mock.

```powershell
for ($i = 1; $i -le 20; $i++) {
  Write-Host "[RUN $i/20] parser"
  python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument

  Write-Host "[RUN $i/20] live_collector_mocks"
  python -m pytest -q tests/test_lms_live_collector_bot.py
}
```

## Bloque C (matematica y replay para entrenamiento, 10 corridas)

```powershell
for ($i = 1; $i -le 10; $i++) {
  Write-Host "[RUN $i/10] alphastar_math"
  python -m pytest -q tests/test_alphastar_math.py

  Write-Host "[RUN $i/10] replay_buffer"
  python -m pytest -q tests/test_replay_buffer.py
}
```

## Salida requerida del agente

Al finalizar, reportar:

1. Comandos ejecutados.
2. Numero de corridas completadas por bloque.
3. Pass/fail de cada bloque.
4. Primer error completo si hubo fallo.

No incluir propuestas de cambios de codigo.
