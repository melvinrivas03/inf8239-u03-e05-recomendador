# Dataset Card · MovieLens Latest Small (ml-latest-small)

| Campo | Valor |
|---|---|
| Fuente | GroupLens Research, University of Minnesota |
| URL oficial | https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |
| Términos de uso | README oficial incluido en el zip (`ml-latest-small/README.txt`), sección *Usage License* |
| Versión | Generada por GroupLens el 26-09-2018 (dataset de *desarrollo*: puede cambiar) |
| SHA-256 del zip usado | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` (978 202 bytes) |
| SHA-256 `ratings.csv` | `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646` |
| Registro de obtención | `data/raw/download_manifest.json` (fecha UTC, URL, hash de cada CSV); no se versiona |
| Referencia | Harper, F. M. y Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM TiiS*, 5(4), 19. https://doi.org/10.1145/2827872 |

## Contenido observado (salida de `scripts/audit_data.py`)

| Medida | Valor |
|---|---|
| Usuarios | 610 |
| Películas en catálogo / con al menos una valoración | 9 742 / 9 724 |
| Valoraciones | 100 836 (densidad 1,70 %) |
| Escala | 0,5 a 5 estrellas en pasos de 0,5; media 3,50 |
| Valoraciones por usuario | mínimo 20, mediana 70,5 |
| Películas con una sola valoración | 3 446 (35 % del catálogo valorado) |
| Cuota del 10 % de películas más valoradas | 60 % de todas las valoraciones |
| Periodo | 29-03-1996 a 24-09-2018 |
| Películas sin género | 34 |

Comprobaciones de contrato (todas pasan): columnas esperadas, sin nulos, `movieId` único,
sin pares usuario–película duplicados, claves foráneas `ratings → movies`, rango y paso del
rating, timestamps positivos.

## Uso en este proyecto y condiciones

* Uso **académico, no comercial**, conforme al README de GroupLens.
* **No se redistribuye**: `data/raw/` está en `.gitignore`. Cada persona obtiene el zip con
  `python scripts/download_data.py` (o lo descarga en grouplens.org y lo instala con `--zip`).
  El hash permite comprobar que se usan los mismos bytes.
* No se declara ni insinúa respaldo de la Universidad de Minnesota ni de GroupLens.
* Se cita Harper y Konstan (2015).
* GroupLens advierte que es un dataset de desarrollo y no una versión archivada para
  resultados científicos compartidos. Por eso los números de este repositorio valen para
  **este hash** y no deben compararse sin más con artículos publicados.

## Limitaciones relevantes

* **Selección**: sólo usuarios con ≥ 20 valoraciones, elegidos al azar. No hay usuarios
  realmente nuevos; el arranque en frío se **simula** (ver System Card).
* **Sesgo de selección**: sólo se observan valoraciones de películas que la persona eligió
  ver y valorar. Que falte una valoración no significa rechazo.
* **Sin demografía**: no puede evaluarse equidad por grupos poblacionales.
* **Timestamps**: indican cuándo se *valoró*, no cuándo se vio la película. Muchas cuentas
  valoran decenas de películas en el mismo segundo (onboarding de MovieLens), así que el orden
  dentro de un mismo timestamp se desempata por `movieId`.
* **Géneros**: 20 etiquetas amplias. Son una descripción pobre del contenido (ver System Card).
