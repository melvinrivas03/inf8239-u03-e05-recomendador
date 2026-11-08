# Declaración de uso de inteligencia artificial

**Herramienta:** Claude (Anthropic), modelo Claude Opus 5.5, en la aplicación Claude con un
entorno de ejecución en la nube (Python 3.12). Fecha: 8 de noviembre de 2026.

**Responsabilidad:** el estudiante revisó, ejecutó y es capaz de explicar todo el código, los
resultados y las conclusiones entregadas. La IA no sustituye esa responsabilidad.

## Para qué se usó

| Tarea | Aporte de la IA | Qué hizo el estudiante |
|---|---|---|
| Arquitectura del repositorio | Propuso la estructura `src/ scripts/ tests/ notebooks/ reports/ docs/` siguiendo LAB08 | Comparó con el proyecto base del curso |
| Código de modelos y métricas | Redactó popularidad, TF-IDF, MF-SGD, híbrido, métricas y pipeline | Leyó cada función y ejecutó las pruebas |
| Pruebas | Redactó 36 pruebas `pytest` con datos sintéticos | Comprobó que fallan si se rompe el contrato (p. ej. rating fuera de rango) |
| Diagnóstico | Detectó y corrigió los errores listados abajo | Validó las correcciones |
| Redacción | Borrador de README, Dataset Card, System Card y PDF | Revisó la interpretación y la adaptó |

## Prompts relevantes

1. Enunciado completo del Ejercicio 05 (propósito, mandato, rúbrica) junto con los PDF de
   LAB08 y LAB09, pidiendo consolidarlos en un producto ejecutable y defendible.
2. Entrega del archivo `ml-latest-small.zip` descargado manualmente de grouplens.org (el
   entorno de la IA no tenía acceso de red a GroupLens).

## Verificaciones realizadas

* `pytest`: 36 pruebas pasan (sin fuga temporal, partición exacta, exclusión de ítems vistos,
  fallback declarado, métricas con casos calculados a mano, regla de Pareto).
* Auditoría del dataset: columnas, nulos, claves foráneas, rango de ratings y timestamps.
  El SHA-256 del zip se registra en el manifiesto.
* Reproducibilidad: el notebook se ejecutó dos veces de principio a fin y las métricas
  coincidieron exactamente (sólo cambian los tiempos de reloj). Además, la MF reentrenada con la
  misma semilla reproduce parámetros idénticos.
* Comparación pareada (bootstrap + McNemar) para no afirmar diferencias que son ruido.
* Las cifras del README y del PDF se copiaron de `reports/*.csv`, no se escribieron de memoria.
* La única referencia bibliográfica (Harper y Konstan, 2015) se comprobó contra el README
  oficial incluido en el zip.

## Errores detectados y corregidos (trazabilidad)

1. **Baselines perdidos al agregar**: `groupby` descartaba las filas con `alpha = NaN`
   (los baselines no tienen alpha), así que no aparecían en la tabla de Pareto. Se detectó
   por un `KeyError` y se corrigió con `dropna=False`.
2. **Objetivo de usuario frío poco realista**: la primera versión evaluaba la *última*
   valoración del usuario frío, a veces años después de su llegada. Se cambió a la valoración
   inmediatamente posterior al onboarding.
3. **Regla de decisión sesgada (protocolo v1)**: la v1 sólo dejaba competir a híbridos y MF y,
   al no haber ninguno dentro de 1 EE, elegía «el de mayor HitRate». Así seleccionó un híbrido
   con HitRate@10 de prueba 0,010 frente a 0,043 de la popularidad. Se rediseñó (v2): compiten
   todos los modelos, se añade el prior de popularidad `gamma` y se decide por Pareto + 1 EE +
   cobertura. **Limitación honesta**: el cambio se hizo *después* de ver los resultados de
   prueba de la v1, así que la prueba no está totalmente intacta. Los resultados v1 se
   conservan en `reports/v1_protocolo_inicial/`.
5. **Regla v2a premiaba el modelo más grande**: la primera regla v2 elegía la mayor cobertura
   entre los equivalentes y usaba segundos de reloj como costo. Seleccionó MF de 50 factores y
   25 épocas, 10× más cómputo para +0,2 puntos de cobertura, y su decisión dependía de la carga
   de la máquina. Se cambió a un costo determinista (factores × épocas × valoraciones), con
   tolerancia de cobertura del 10 % y elección de la más barata. El cambio sólo usa validación,
   pero también se hizo tras ver la prueba de v2a (archivada en `reports/v2a_regla_max_cobertura/`).
6. **Empates del contenido**: el usuario de historial amplio recibía la misma lista con α = 0,25
   y α = 0,75. Se investigó como posible error y resultó ser un comportamiento real: 1 053
   dramas empatan con similitud máxima. Se documentó en el notebook (sección 8.1).
4. **Fallback elegido por ruido**: en la v1, la popularidad del fallback se elegía en validación,
   donde la suavizada superaba al conteo; en prueba el orden se invirtió (diferencia menor que
   1 EE). En la v2 se fija el conteo por diseño: el objetivo es predecir *consumo*, no calidad.
   El peso `beta` del fallback se elige con una muestra fría disjunta.
