# U03.E05 · Motor recomendador reproducible (MovieLens)

**INF-8239 Ciencia de Datos II** · Unidad 03 · Algoritmos y características generativas
Estudiante: _[Nombre y matrícula]_ · Docente: Edwin Ramón José Nolasco

Este repositorio consolida LAB08 y LAB09 en un solo producto:
* baselines de popularidad (conteo y media bayesiana);
* recomendador por contenido (TF-IDF de géneros);
* factorización matricial con sesgos (SGD);
* híbrido con fallback para usuarios fríos;
* corte temporal y métricas de rating y de ranking;
* decisión Pareto con criterio de costo (Green AI).

## Resultado principal (prueba, 610 usuarios, media de 3 semillas)

| Modelo | RMSE | HitRate@10 ± EE | NDCG@10 | Cobertura | Cuota cabeza | Novedad | Entrenamiento |
|---|---|---|---|---|---|---|---|
| Aleatorio | — | 0,001 ± 0,001 | 0,000 | 46,3 % | 9 % | 7,47 | 0 s |
| Media global | 1,117 | — | — | — | — | — | 0 s |
| Popularidad (conteo) | — | **0,043 ± 0,008** | **0,019** | 1,3 % | 100 % | 1,42 | <0,01 s |
| Popularidad suavizada | 1,058 | 0,034 ± 0,007 | 0,018 | 0,8 % | 100 % | 1,76 | <0,01 s |
| Sesgos usuario+ítem | 0,973 | 0,023 ± 0,006 | 0,012 | 0,7 % | 100 % | 2,18 | <0,1 s |
| Contenido (TF-IDF géneros) | — | 0,008 ± 0,004 | 0,005 | **22,3 %** | 48 % | 5,18 | <0,1 s |
| MF (10 factores, 12 épocas) | **0,964** | 0,011 ± 0,004 | 0,006 | 0,9 % | 62 % | 3,92 | ≈ 8–9 s |
| Híbrido LAB09 (α = 0,25, γ = 0) | — | 0,009 ± 0,004 | 0,005 | 14,4 % | 41 % | 4,93 | ≈ 8–9 s |
| **Híbrido final (α = 0,5, γ = 0,25)** | — | 0,031 ± 0,007 | 0,016 | 3,8 % | 93 % | 2,30 | ≈ 8–9 s |

*Cuota cabeza* = fracción de posiciones recomendadas ocupadas por el 10 % de películas más
valoradas. *Novedad* = media de −log₂(fracción de usuarios que vio el ítem). EE = error
estándar binomial entre usuarios. Fuente: `reports/test_results.csv`.

**Decisión.** Para usuarios con historial se usa el **híbrido final**. Su acierto es
estadísticamente indistinguible del de la popularidad: en la comparación pareada usuario a
usuario el IC 95 % incluye el 0 (`reports/paired_tests.csv`). A cambio, triplica la cobertura,
reduce la concentración y personaliza. Si el único objetivo fuera HitRate@10, la popularidad
por conteo sería la elección; también está en la frontera de Pareto. Para usuarios nuevos se
usa popularidad, declarada como **no personalizada**. El razonamiento completo está en
`notebooks/E05_motor_recomendador.ipynb` (sección 10).

## Inicio rápido

```bash
# 1) entorno (elija uno)
uv sync --extra dev                                   # con uv (usa uv.lock)
python -m venv .venv && . .venv/bin/activate && \
  pip install -r requirements.txt && pip install -e . --no-deps   # con pip

# 2) datos: NO están en el repositorio
python scripts/download_data.py                       # descarga oficial de GroupLens (~1 MB)
#   o, si su red no llega a GroupLens, descárguelo a mano desde
#   https://grouplens.org/datasets/movielens/latest/ y:
python scripts/download_data.py --zip ruta/ml-latest-small.zip
python scripts/audit_data.py                          # contratos + hash SHA-256

# 3) pruebas y experimento
python -m pytest                                      # 36 pruebas, ~5 s, sin MovieLens
python scripts/run_experiments.py                     # protocolo completo, ~11 min en CPU
python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.75   # experimento individual LAB09

# 4) notebook ejecutado (reproduce todo)
jupyter nbconvert --to notebook --execute notebooks/E05_motor_recomendador.ipynb --output E05_rerun.ipynb
```

Con uv, anteponga `uv run` a cada comando. Hash esperado del zip:
`696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.

## Estructura

```
src/inf8239_u03/   data.py (descarga, hash, auditoría, datos sintéticos) · split.py (corte temporal, usuarios fríos)
                   popularity.py · content.py · matrix_factorization.py · hybrid.py
                   metrics.py · evaluation.py · pipeline.py (protocolo completo)
scripts/           download_data.py · audit_data.py · run_experiments.py · lab09_hybrid.py
tests/             test_data · test_content · test_matrix_factorization · test_metrics · test_pipeline
notebooks/         E05_motor_recomendador.ipynb (ejecutado)
reports/           tablas CSV, summary.json, figures/; v1_protocolo_inicial/ y v2a_regla_max_cobertura/ (trazabilidad)
docs/              dataset_card.md · system_card.md · declaracion_ia.md · E05_entrega.pdf
data/raw/          vacío en Git (MovieLens se descarga aquí)
```

## Datos y licencia

MovieLens Latest Small, de GroupLens Research (University of Minnesota). Se usa con fines
académicos, no comerciales, según el README oficial del dataset. **No se redistribuye**:
`data/raw/` está en `.gitignore`. Es un dataset de *desarrollo* que puede cambiar, por eso se
registran la fecha, la URL y el SHA-256. No se declara respaldo de la Universidad de Minnesota
ni de GroupLens. Detalles: `docs/dataset_card.md`.

> Harper, F. M. y Konstan, J. A. (2015). The MovieLens Datasets: History and Context.
> *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19. https://doi.org/10.1145/2827872

## Baselines

* **Aleatorio**: piso de referencia.
* **Media global**: piso de RMSE.
* **Popularidad por conteo**: lo más visto.
* **Popularidad suavizada**: media bayesiana `n/(n+m)·media + m/(n+m)·media_global`,
  con m = 27 (p90). Sin suavizar, el Top-10 lo ocupan películas con 1 o 2 votos de 5 estrellas.
* **Sesgos usuario+ítem**: μ + b_u + b_i regularizados.

## Particiones

* **Leave-last-out temporal por usuario.** Para cada usuario se ordenan sus valoraciones por
  timestamp: la última va a **prueba**, la penúltima a **validación** y el resto a
  entrenamiento.
* Toda selección (factores, épocas, α, γ) se hace en validación con 3 semillas
  (42, 7, 2025). Después se reentrena con entrenamiento + validación y se evalúa en prueba.
* **Ranking sobre todo el catálogo** (9 742 películas, menos las ya vistas), sin muestreo
  de negativos.
* **Usuarios fríos.** El 10 % de los usuarios (61) se retira por completo. Se evalúa la
  valoración siguiente a sus primeras 0 o 3 valoraciones. β se elige con otra muestra fría
  disjunta.

## Decisión Pareto

Se usa sólo validación:
1. **Frontera de Pareto**: maximizar HitRate@10 y cobertura; minimizar cómputo (factores ×
   épocas × valoraciones, determinista) y parámetros.
2. **Equivalencia en acierto**: a ≤ 1 EE de la mejor HitRate@10.
3. **Equivalencia en cobertura**: ≥ 90 % de la mayor cobertura del grupo anterior.
4. Entre esas, **la más barata**.

Resultado: MF de 10 factores y 12 épocas, α = 0,5, γ = 0,25. La alternativa de 50 factores y
25 épocas tenía el mismo acierto con 10× más cómputo (`reports/figures/pareto_validacion.png`).

## Errores y límites (resumen)

* **Ratings ≠ ranking.** La MF tiene el mejor RMSE y aun así es mala en el Top-10.
* **El híbrido de LAB09 (sin γ) no supera a la popularidad.** El prior γ se añadió *después*
  de ver la prueba de la primera versión del protocolo; está documentado y archivado en
  `reports/v1_protocolo_inicial/`, así que la prueba no está totalmente intacta.
* **Fallback con 3 valoraciones.** Con β = 0,5 queda por debajo de la popularidad pura
  (0,109 vs 0,148, con 61 usuarios). Es el siguiente experimento.
* **Contenido con 20 géneros.** Produce empates masivos y sobre-especialización.
* **Usuarios con historial muy amplio.** Casi no se aciertan (HitRate@10 de 0,005).
* **Otros límites**: sin demografía, sesgo de selección, corte temporal por usuario y no
  global, y n = 610 (EE ≈ 0,008).

Más detalle en `docs/system_card.md`.

## Uso de IA

Se usó Claude (Anthropic) para código, pruebas, diagnóstico y redacción. Las verificaciones y
correcciones están en `docs/declaracion_ia.md`. El estudiante responde por todo lo entregado.
