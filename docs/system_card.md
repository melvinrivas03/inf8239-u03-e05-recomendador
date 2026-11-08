# System Card · Motor recomendador U03.E05

## Propósito y alcance

Ordenar películas del catálogo MovieLens Latest Small para una persona, a partir de sus
valoraciones previas. Es un **prototipo académico de evaluación offline**: no se ha probado con
usuarios reales y no debe usarse con fines comerciales (licencia GroupLens).

Un Top-10 **no** es una verdad sobre los gustos de nadie. Es el resultado de unos datos
observados, una función de puntuación y una política de candidatos (todo el catálogo no visto).

## Componentes

| Componente | Señal | Uso |
|---|---|---|
| Popularidad por conteo | nº de valoraciones en entrenamiento | prior del híbrido; fallback de usuario nuevo |
| Popularidad suavizada | media bayesiana (m = p90 de conteos = 27) | baseline de LAB08 y de RMSE |
| Contenido | TF-IDF de 20 géneros + coseno; perfil ponderado por (rating − media del usuario) | señal del híbrido; onboarding |
| MF-SGD con sesgos | μ + b_u + b_i + p_u·q_i, lr 0,01, reg 0,05 | señal colaborativa |
| **Híbrido final** | (1 − γ)·[α·MF + (1 − α)·contenido] + γ·popularidad, todo min-max por usuario | **α = 0,5, γ = 0,25, MF 10 factores, 12 épocas** |

## Política por tipo de usuario

1. Con historial y vector colaborativo: híbrido. La lista excluye lo ya visto.
2. Sin vector colaborativo pero con valoraciones (onboarding):
   β·contenido + (1 − β)·popularidad, con β = 0,5.
3. Sin valoraciones: popularidad por conteo. La salida se etiqueta
   **«popularidad (NO personalizada)»**.

Empates: se rompen por popularidad y luego por posición en el catálogo (determinista).

## Desempeño (prueba, leave-last-out, 610 usuarios, media de 3 semillas)

| Modelo | RMSE | HitRate@10 | NDCG@10 | Cobertura | Cuota cabeza |
|---|---|---|---|---|---|
| Popularidad (conteo) | — | 0,043 | 0,019 | 1,25 % | 100 % |
| MF f=10 ep=12 | 0,964 | 0,011 | 0,006 | 0,9 % | 62 % |
| **Híbrido final** | — | **0,031** | **0,016** | **3,8 %** | **93 %** |

Usuarios fríos (61 por semilla): con 0 valoraciones, HitRate@10 = 0,142 (popularidad); con 3
valoraciones, 0,109 (contenido+popularidad), frente a 0,148 de la popularidad pura.

## Limitaciones conocidas

* **Ratings ≠ ranking.** La MF optimiza el error en estrellas. Por eso tiene el mejor RMSE y,
  aun así, es mala ordenando lo que la persona verá después. El prior γ compensa en parte.
* **El acierto no supera a la popularidad.** La diferencia no es significativa (prueba
  pareada), pero la estimación puntual del híbrido es menor.
* **Contenido pobre.** Con 20 géneros hay empates masivos: para el usuario 414 (2 697
  valoraciones), 1 053 películas etiquetadas sólo como *Drama* (10,8 % del catálogo) empatan
  con similitud máxima. El contenido actúa como filtro de género y por eso α = 0,25 y α = 0,75
  le dan el mismo Top-10. Ejemplo de sobre-especialización: el Top-10 por contenido de *Silence of the Lambs*
  son diez títulos Crime|Horror|Thriller, tres de ellos de la saga *Saw*.
* **Heavy users.** En el tercil de historial amplio (127 a 2 697 valoraciones), HitRate@10 =
  0,005, frente a 0,034 en el tercil pequeño. Su siguiente película suele ser poco popular.
* **Ítems nuevos.** El 3,8 % de los ítems de prueba no tiene valoraciones de entrenamiento.
  La MF no puede recomendarlos; sólo el contenido puede.
* **Evaluación.** Un único ítem relevante por usuario y n = 610 dan un EE ≈ 0,008:
  las diferencias pequeñas son ruido. El corte es temporal *por usuario*, no global.
* **Protocolo modificado tras ver resultados.** Ver `declaracion_ia.md` y
  `reports/v1_protocolo_inicial/`.

## Riesgos y mitigaciones

| Riesgo | Evidencia | Mitigación |
|---|---|---|
| Popularidad dominante | 93 % de posiciones en el 10 % más popular; Gini de exposición 0,99 | γ acotado; monitorizar cobertura y cuota de cabeza |
| Burbuja de filtro | ILD del contenido = 0,08 (listas casi idénticas en género) | α ≥ 0,5; añadir diversificación (MMR) como siguiente paso |
| Sesgo de selección | sólo se observan películas que la persona eligió ver | no interpretar ausencia como rechazo; evaluar con exposición aleatoria si existiera |
| Ausencia de demografía | el dataset no tiene atributos de personas | no se afirma nada sobre equidad por grupos |
| Retroalimentación | lo recomendado se ve más y entrena el siguiente modelo | posiciones de exploración; re-entrenar con corte temporal y vigilar la deriva de cobertura |

## Costo

Entrenamiento de ≈ 8–9 s en CPU (Python puro), 114 k parámetros MF + 185 k valores TF-IDF.
El protocolo completo (93 configuraciones × 3 semillas en validación, más prueba y frío) tarda ≈ 11 min.
