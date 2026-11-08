"""Genera un RESUMEN de 2 páginas (docs/E05_resumen.pdf) a partir de reports/.

El informe formal de entrega es docs/E05_entrega.pdf (fuente editable: docs/E05_entrega.docx).

    python scripts/make_report_pdf.py https://github.com/USUARIO/REPO
"""
import json, sys
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = str(ROOT / "reports")
REPO = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/[usuario]/inf8239-u03-e05-recomendador"
OUT = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "docs" / "E05_resumen.pdf")
import matplotlib  # DejaVu viene con matplotlib: fuente portable con ≤ ≥ × → ₂
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB")
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontName="DV", fontSize=8.6, leading=11.2, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=B, fontName="DVB", fontSize=14, leading=17, spaceAfter=2, textColor=colors.HexColor("#0b3d91"))
H2 = ParagraphStyle("h2", parent=B, fontName="DVB", fontSize=10, leading=13, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#0b3d91"))
SM = ParagraphStyle("sm", parent=B, fontSize=7.2, leading=9, textColor=colors.HexColor("#444444"))
TC = ParagraphStyle("tc", parent=B, fontSize=7.3, leading=8.8, spaceAfter=0)
TCB = ParagraphStyle("tcb", parent=TC, fontName="DVB")

def f3(x): return "—" if pd.isna(x) else f"{x:.3f}".replace(".", ",")
def pct(x): return "—" if pd.isna(x) else f"{100*x:.1f} %".replace(".", ",")

t = pd.read_csv(f"{R}/test_results.csv")
s = json.load(open(f"{R}/summary.json"))
pr = pd.read_csv(f"{R}/paired_tests.csv")
cold = pd.read_csv(f"{R}/cold_start.csv")
d = s["decision"]

def table(rows, widths, bold_row=None, head_bg="#0b3d91"):
    data = [[Paragraph(str(c), TCB if i == 0 or i == bold_row else TC) for c in r] for i, r in enumerate(rows)]
    tb = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(head_bg)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d0da")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if bold_row is not None:
        st.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#e6efff")))
    for r in range(1, len(rows)):  # encabezado en blanco
        pass
    tb.setStyle(TableStyle(st))
    data[0] = [Paragraph(f'<font color="white">{c}</font>', TCB) for c in rows[0]]
    tb._cellvalues[0] = data[0]
    return tb

story = []
story.append(Paragraph("U03.E05 · Motor recomendador reproducible", H1))
story.append(Paragraph("INF-8239 Ciencia de Datos II · Unidad 03 · Docente: Edwin Ramón José Nolasco · Universidad Autónoma de Santo Domingo", SM))
story.append(Paragraph("Estudiante: <b>Melvin H. Rivas Genao</b> · Matrícula 100279298 · Fecha: 8 de noviembre de 2026", SM))
story.append(Spacer(1, 4))
story.append(Paragraph(f'<b>Repositorio:</b> <link href="{REPO}"><font color="#0b3d91"><u>{REPO}</u></font></link>', B))
story.append(Paragraph("Contiene el notebook ejecutado, <i>src</i>, <i>tests</i> (36 pruebas), <i>reports</i>, <i>requirements.txt</i>, <i>uv.lock</i>, README, Dataset Card, System Card y la declaración de IA. MovieLens <b>no</b> se redistribuye: se descarga con <i>scripts/download_data.py</i> y se verifica con SHA-256 <font size=6.5>696d65a3…c9e436</font>.", B))

story.append(Paragraph("1. Datos, particiones y protocolo", H2))
story.append(Paragraph(
    "MovieLens Latest Small (GroupLens, uso académico según su README; Harper y Konstan, 2015, doi:10.1145/2827872): 610 usuarios, 9 742 películas, "
    "100 836 valoraciones (densidad 1,7 %). El 10 % de películas más valoradas concentra el 60 % de las valoraciones. La auditoría de columnas, claves foráneas, "
    "rango de rating y timestamps pasa completa. <b>Corte temporal leave-last-out por usuario</b>: última valoración → prueba, penúltima → validación, resto → entrenamiento "
    "(sin fuga temporal por usuario, verificado con pruebas). Ranking sobre todo el catálogo no visto. Toda selección se hace en validación con 3 semillas (42, 7, 2025); "
    "después se reentrena con entrenamiento + validación y se evalúa una vez en prueba. <b>Usuarios fríos</b>: 10 % de los usuarios (61) se retiran del entrenamiento y se "
    "evalúa la valoración siguiente a sus primeras 0 o 3.", B))

story.append(Paragraph("2. Tabla principal de resultados (prueba, 610 usuarios, media de 3 semillas)", H2))
order = ["Aleatorio", "Media global", "Popularidad (conteo)", "Popularidad suavizada", "Sesgos usuario+ítem",
         "Contenido (TF-IDF géneros)"]
mf_name = [m for m in t.modelo if m.startswith("MF")][0]
lab = [m for m in t.modelo if m.startswith("Híbrido LAB09")][0]
fin = [m for m in t.modelo if m.startswith("Híbrido FINAL")][0]
rows = [["Modelo", "RMSE", "HitRate@10 ± EE", "NDCG@10", "Cobertura", "Cuota cabeza", "Novedad", "Tiempo"]]
for m in order + [mf_name, lab, fin]:
    r = t[t.modelo == m].iloc[0]
    hr = "—" if pd.isna(r["HitRate@10"]) else f'{f3(r["HitRate@10"])} ± {f3(r["HitRate@10_EE"])}'
    name = m.replace("Híbrido FINAL", "<b>Híbrido FINAL</b>")
    rows.append([name, f3(r["RMSE"]), hr, f3(r["NDCG@10"]), pct(r["Cobertura"]), pct(r["Cuota_cabeza"]),
                 "—" if pd.isna(r["Novedad"]) else f'{r["Novedad"]:.2f}'.replace(".", ","),
                 "<0,1 s" if r["seg_entrenamiento"] == 0 else f'{r["seg_entrenamiento"]:.1f} s'.replace(".", ",")])
story.append(table(rows, [4.3*cm, 1.3*cm, 2.5*cm, 1.95*cm, 2.05*cm, 2.0*cm, 1.95*cm, 1.5*cm], bold_row=len(rows)-1))
story.append(Paragraph("EE = error estándar binomial entre usuarios. Cuota cabeza = posiciones ocupadas por el 10 % de películas más valoradas. Novedad = −log₂ de la fracción de usuarios que vio el ítem. Fuente: reports/test_results.csv.", SM))

pp = pr[pr.B == "Popularidad (conteo)"]
story.append(Paragraph("3. Lectura de los resultados", H2))
story.append(Paragraph(
    "<b>Ratings ≠ ranking.</b> La MF tiene el mejor RMSE (0,964), pero su HitRate@10 (0,011) es la cuarta parte del de la popularidad por conteo (0,043): "
    "predecir estrellas no es predecir qué se verá después. El híbrido de LAB09 (MF + contenido) tampoco la supera. "
    f"Al añadir un prior de popularidad (γ) el híbrido final llega a 0,031. En la <b>comparación pareada</b> usuario a usuario frente a la popularidad, la diferencia media es "
    f"{f3(pp.dif_HitRate.mean())}, con IC 95 % entre {f3(pp.IC95_inf.min())} y {f3(pp.IC95_sup.max())} según la semilla (McNemar p ≥ {pp.p_McNemar.min():.2f}): "
    "no es significativa. A cambio, el híbrido final <b>triplica la cobertura</b> (3,8 % vs 1,25 %) y eleva la novedad.", B))
c0 = cold[(cold.escenario == "0 valoraciones") & (cold.modelo.str.startswith("Híbrido"))].iloc[0]
c3 = cold[(cold.escenario == "3 valoraciones") & (cold.modelo.str.startswith("Híbrido"))].iloc[0]
c3p = cold[(cold.escenario == "3 valoraciones") & (cold.modelo == "Popularidad (conteo)")].iloc[0]
story.append(Paragraph(
    f"<b>Usuarios fríos.</b> La MF no tiene vector para ninguno de los 61. Con 0 valoraciones, el fallback de popularidad acierta en el {pct(c0['HitRate@10'])} de los casos y se "
    f"declara «NO personalizado». Con 3 valoraciones, contenido + popularidad (β = {d['beta_frio']:g}) logra {pct(c3['HitRate@10'])}, por debajo de la popularidad pura "
    f"({pct(c3p['HitRate@10'])}): el β elegido no se sostuvo en prueba. <b>Perfiles</b>: el usuario de historial amplio (2 697 valoraciones) recibe dramas clásicos "
    "personalizados; el de historial pequeño (19), clásicos populares; el nuevo, el Top-10 de popularidad. Ninguna lista repite títulos ya vistos.", B))

h4 = Paragraph("4. Híbrido y decisión Pareto", H2)
img = Image(f"{R}/figures/pareto_validacion.png", width=10.4*cm, height=6.2*cm)
txt = Paragraph(
    "Regla aplicada sólo en validación: (1) frontera de Pareto, maximizando HitRate@10 y cobertura y minimizando cómputo (factores × épocas × valoraciones) y parámetros; "
    f"(2) acierto a ≤ 1 EE de la mejor HitRate@10 ({f3(d['mejor_val_HitRate'])} − {f3(d['1EE'])}); (3) cobertura ≥ 90 % del máximo de ese grupo; (4) <b>la más barata</b>. "
    f"<br/><b>Elegida:</b> MF de {d['factores']} factores y {d['epocas']} épocas, α = {d['alpha']:g} (peso colaborativo frente a contenido), γ = {d['gamma']:g} (prior de popularidad). "
    "Con 50 factores y 25 épocas se obtenía el mismo acierto con 10× más cómputo (Green AI). La popularidad también es no dominada: si el único objetivo fuera "
    "HitRate@10, sería la opción. <b>Costo</b>: ≈ 8–9 s de entrenamiento en CPU y ≈114 k parámetros de la MF.", B)
story.append(KeepTogether([h4, Table([[img, txt]], colWidths=[10.6*cm, 7.6*cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")])]))

story.append(Paragraph("5. Errores, límites y decisión final", H2))
story.append(Paragraph(
    "<b>Errores corregidos</b>: (a) un <i>groupby</i> descartaba los baselines (alpha = NaN) de la tabla de Pareto; "
    "(b) el usuario frío se evaluaba en su última valoración, años después, en lugar de la siguiente al onboarding; "
    "(c) la regla v1 sólo dejaba competir a híbridos y eligió uno con HitRate de prueba de 0,010 frente a 0,043 de la popularidad. "
    "<b>Límite metodológico declarado</b>: γ y la regla v2 se introdujeron tras ver la prueba de la v1 (archivada en reports/v1_protocolo_inicial/), "
    "así que la prueba no está totalmente intacta. <b>Otros límites</b>: géneros pobres (20 etiquetas, empates masivos, sobre-especialización); "
    "heavy users casi sin aciertos (0,5 %); sesgo de selección; sin demografía; corte temporal por usuario y no global; n = 610 (EE ≈ 0,008). "
    "<b>Riesgo principal</b>: dominio de la popularidad y retroalimentación (Gini de exposición 0,99). Se mitiga con γ acotado, monitorización de cobertura y "
    "posiciones de exploración.", B))
story.append(Paragraph(
    "<b>Decisión</b>: usuarios con historial → híbrido final (acierto equivalente a la popularidad, cobertura ×3, personalizado). Usuarios nuevos → popularidad declarada "
    "como no personalizada. Siguiente experimento: β del onboarding con más usuarios fríos y diversificación del Top-10.", B))

story.append(Paragraph("6. Declaración de uso de IA", H2))
story.append(Paragraph(
    "Herramienta: Claude (Anthropic, modelo Claude Opus 5.5) para redactar código, pruebas y documentación y para diagnosticar errores. "
    "Prompts relevantes: el enunciado del Ejercicio 05 con los PDF de LAB08 y LAB09, y la entrega del zip de MovieLens descargado manualmente. "
    "Verificaciones: 36 pruebas <i>pytest</i>, auditoría del dataset con hash, determinismo con semilla, re-ejecución completa del notebook y cifras copiadas de "
    "reports/*.csv. Única referencia (Harper y Konstan, 2015) contrastada con el README oficial. Correcciones: las de la sección 5. "
    "El estudiante revisó, ejecutó y responde por todo lo entregado (detalle en docs/declaracion_ia.md).", B))

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.3*cm, bottomMargin=1.2*cm,
                        title="U03.E05 Motor recomendador reproducible", author="Melvin H. Rivas Genao")
doc.build(story)
print("ok", OUT)
