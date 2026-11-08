"""Experimento completo del Ejercicio 05 (LAB08 + LAB09 consolidados) · protocolo v2.

1. Corte temporal leave-last-out: train / validación (penúltima) / prueba (última).
2. Selección en VALIDACIÓN, 3 semillas: factores × épocas de la MF, alpha (CF vs contenido)
   y gamma (prior de popularidad) del híbrido.
3. Decisión: frontera de Pareto (HitRate@10 y cobertura ↑; segundos y parámetros ↓) +
   equivalencia a 1 EE binomial de la mejor HitRate@10 → la de mayor cobertura.
   Compiten todas las configuraciones, incluidos los baselines.
4. Reentrenar con train+validación y evaluar UNA vez en prueba todos los modelos (3 semillas).
5. Usuarios fríos: 10 % de usuarios tratados como nuevos; se evalúa la valoración siguiente
   al onboarding (0 o 3 valoraciones). beta se elige con otra muestra fría disjunta.
6. Tres perfiles, riesgos de exposición y figuras.

v1 (sin gamma, regla «la más barata entre híbridos») está archivado en reports/v1_protocolo_inicial/.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .base import History, ItemIndex, histories
from .content import ContentRecommender
from .data import MovieLens, audit
from .evaluation import evaluate_ranking, evaluate_ratings, hits_by_user, item_counts, paired_comparison
from .hybrid import HybridRecommender
from .matrix_factorization import BiasBaseline, MatrixFactorization
from .metrics import pareto_mask
from .popularity import (GlobalMean, PopularityCount, PopularitySmoothed, RandomRecommender,
                         popularity_table)
from .split import cold_user_split, leave_last_out, merge_train_valid

FULL_GRID = {"factors": [10, 20, 50], "epochs": [12, 25], "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
             "gamma": [0.0, 0.25, 0.5]}
QUICK_GRID = {"factors": [5, 10], "epochs": [5], "alpha": [0.0, 0.5, 1.0], "gamma": [0.0, 0.5]}
CONTENT_QUERIES = ["Toy Story (1995)", "Silence of the Lambs, The (1991)", "Notebook, The (2004)"]


@dataclass
class Context:
    ml: MovieLens
    index: ItemIndex
    content: ContentRecommender
    out: Path
    k: int = config.K


def _save(df: pd.DataFrame, out: Path, name: str) -> pd.DataFrame:
    df.to_csv(out / name, index=False)
    return df


def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Media y desviación estándar sobre semillas."""
    num = [c for c in df.columns if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, sort=False, dropna=False)[num]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    std.columns = [f"{c}_sd" for c in std.columns]
    return pd.concat([mean, std], axis=1).reset_index()


# --------------------------------------------------------------------------- LAB08
def lab08(ctx: Context) -> dict:
    r, m = ctx.ml.ratings, ctx.ml.movies
    aud = audit(ctx.ml)
    (ctx.out / "audit.json").write_text(json.dumps(aud, indent=2, ensure_ascii=False), encoding="utf-8")
    pop = popularity_table(r, m)
    top = pop.head(ctx.k)[["movieId", "title", "genres", "mean", "count", "weighted_score"]].round(4)
    _save(top, ctx.out, "popularity_top10.csv")
    # comparación: media cruda vs suavizada (por qué hace falta suavizar)
    raw = pop.sort_values(["mean", "count"], ascending=False).head(ctx.k)[["title", "mean", "count", "weighted_score"]]
    _save(raw.round(4), ctx.out, "popularity_raw_mean_top10.csv")
    counts = item_counts(r, ctx.index)
    queries = [t for t in CONTENT_QUERIES if t in set(m.title)]
    if len(queries) < 3:  # datos sintéticos: usar las 3 más populares de géneros distintos
        queries = list(pop.drop_duplicates("genres").title.head(3))
    qids = [int(m.loc[m.title == t, "movieId"].iloc[0]) for t in queries]
    content_recs = pd.concat([ctx.content.similar_items(q, ctx.k, tiebreak=counts) for q in qids])
    _save(content_recs, ctx.out, "content_recommendations.csv")
    return {"audit": aud, "popularity_top10": top, "popularity_raw": raw,
            "content_recs": content_recs, "pop_m": pop.attrs["m"]}


# --------------------------------------------------------------------------- validación
def _label(a: float, g: float) -> str:
    if a == 1.0 and g == 0.0:
        return "MF"
    return "Híbrido"


def validation_grid(ctx: Context, split, seeds=config.SEEDS, grid=FULL_GRID, log=print) -> pd.DataFrame:
    tr, va = split.train, split.valid
    rows = []
    movies = ctx.ml.movies
    base_cfg = {"factores": 0, "epocas": 0, "alpha": np.nan, "gamma": np.nan}
    for name, model, fit in [
        ("Popularidad (conteo)", PopularityCount(), lambda mdl: mdl.fit(tr, ctx.index)),
        ("Popularidad suavizada", PopularitySmoothed(), lambda mdl: mdl.fit(tr, ctx.index)),
        ("Contenido", ContentRecommender(), lambda mdl: mdl.fit(None, ctx.index, movies)),
    ]:
        t0 = time.perf_counter()
        fit(model)
        fit_s = time.perf_counter() - t0
        res = evaluate_ranking(model, tr, va, ctx.index, ctx.content.X, ctx.k)
        for s in seeds:  # deterministas: se repite el valor para promediar igual que el resto
            cost = model.n_params() if name == "Contenido" else len(tr)  # una pasada por los datos
            rows.append({"modelo": name, **base_cfg, "seed": s, "seg_entrenamiento": fit_s, "costo_computo": cost,
                         "n_parametros": model.n_params(), "RMSE": evaluate_ratings(model, va)["RMSE"], **res})
    pop = PopularityCount().fit(tr, ctx.index)
    for f in grid["factors"]:
        for e in grid["epochs"]:
            for s in seeds:
                mf = MatrixFactorization(factors=f, epochs=e, seed=s).fit(tr, ctx.index)
                rm = evaluate_ratings(mf, va)["RMSE"]
                log(f"  MF f={f:>3} ep={e:>3} seed={s:<5} RMSE_val={rm:.4f}  {mf.fit_seconds_:.1f}s")
                for g in grid["gamma"]:
                    for a in grid["alpha"]:
                        h = HybridRecommender(mf, ctx.content, pop, alpha=a, gamma=g)
                        res = evaluate_ranking(h, tr, va, ctx.index, ctx.content.X, ctx.k)
                        rows.append({"modelo": _label(a, g), "factores": f, "epocas": e, "alpha": a, "gamma": g,
                                     "seed": s, "seg_entrenamiento": mf.fit_seconds_,
                                     "costo_computo": f * e * len(tr),  # actualizaciones SGD × dimensión
                                     "n_parametros": mf.n_params() if _label(a, g) == "MF" else h.n_params() + pop.n_params(),
                                     "RMSE": rm, **res})
    return pd.DataFrame(rows)


def choose_config(val_agg: pd.DataFrame, k: int, n_users: int, cov_tol: float = 0.10) -> tuple[pd.Series, pd.DataFrame]:
    """Regla v2 (Pareto + 1 EE + cobertura + Green AI). Sólo usa VALIDACIÓN.

    1. Frontera de Pareto: maximizar HitRate@K y Cobertura; minimizar costo de cómputo
       (determinista: factores × épocas × valoraciones) y número de parámetros.
    2. Equivalencia en acierto: HitRate@K ≥ mejor − 1 EE binomial (n = usuarios de validación).
    3. Equivalencia en cobertura: ≥ (1 − cov_tol) × la mayor cobertura del grupo anterior.
    4. Entre esas, la MÁS BARATA. Compiten todas las configuraciones, incluida la popularidad.
    El tiempo de reloj se informa pero no decide (depende de la carga de la máquina).
    """
    hr = f"HitRate@{k}"
    df = val_agg.copy()
    df["pareto"] = pareto_mask(df, maximize=[hr, "Cobertura"], minimize=["costo_computo", "n_parametros"])
    best = df[hr].max()
    se = np.sqrt(best * (1 - best) / max(1, n_users))
    df["dentro_1EE"] = df[hr] >= best - se
    cand = df[df.pareto & df.dentro_1EE]
    df["cobertura_equivalente"] = df.index.isin(cand.index) & (df["Cobertura"] >= (1 - cov_tol) * cand["Cobertura"].max())
    cand = df[df.cobertura_equivalente]
    chosen = cand.sort_values(["costo_computo", "n_parametros", hr], ascending=[True, True, False]).iloc[0]
    df["elegida"] = df.index == chosen.name
    df.attrs["best_hr"], df.attrs["se"] = best, se
    return chosen, df


# --------------------------------------------------------------------------- prueba
def test_phase(ctx: Context, split, f: int, e: int, alpha: float, gamma: float, lab_alpha: float,
               seeds=config.SEEDS, log=print):
    full_tr, te = merge_train_valid(split), split.test
    rows = []
    pops = PopularitySmoothed().fit(full_tr, ctx.index)
    popc = PopularityCount().fit(full_tr, ctx.index)
    for s in seeds:
        mf = MatrixFactorization(factors=f, epochs=e, seed=s).fit(full_tr, ctx.index)
        models = [
            ("Aleatorio", RandomRecommender(seed=s).fit(full_tr, ctx.index), 0.0),
            ("Media global", GlobalMean().fit(full_tr, ctx.index), 0.0),
            ("Popularidad (conteo)", popc, 0.0),
            ("Popularidad suavizada", pops, 0.0),
            ("Sesgos usuario+ítem", BiasBaseline().fit(full_tr, ctx.index), 0.0),
            ("Contenido (TF-IDF géneros)", ctx.content, 0.0),
            (f"MF f={f} ep={e}", mf, mf.fit_seconds_),
            (f"Híbrido LAB09 α={lab_alpha:g} γ=0", HybridRecommender(mf, ctx.content, popc, alpha=lab_alpha), mf.fit_seconds_),
            (f"Híbrido FINAL α={alpha:g} γ={gamma:g}",
             HybridRecommender(mf, ctx.content, popc, alpha=alpha, gamma=gamma), mf.fit_seconds_),
        ]
        for name, model, fit_s in models:
            res = evaluate_ranking(model, full_tr, te, ctx.index, ctx.content.X, ctx.k)
            rows.append({"modelo": name, "seed": s, **evaluate_ratings(model, te), **res,
                         "seg_entrenamiento": fit_s, "n_parametros": model.n_params()})
        log(f"  prueba seed={s} listo")
    raw = pd.DataFrame(rows)
    agg = _agg(raw, ["modelo"])
    rank_cols = [c for c in agg.columns if c.startswith(("HitRate", "Precision", "NDCG", "MRR", "Cobertura",
                                                          "Novedad", "Cuota", "Diversidad", "Gini"))]
    agg.loc[agg.modelo == "Media global", rank_cols] = np.nan  # sólo es referencia de RMSE
    return raw, agg, mf, popc


def history_buckets(ctx: Context, split, model) -> pd.DataFrame:
    """HitRate@K del modelo elegido por tamaño de historial (usuarios con historial pequeño vs amplio)."""
    full_tr = merge_train_valid(split)
    hits = hits_by_user(model, full_tr, split.test, ctx.index, ctx.k)
    n = full_tr.groupby("userId").size().reindex(hits.index)
    bins = pd.qcut(n, q=3, labels=["pequeño (tercil 1)", "medio (tercil 2)", "amplio (tercil 3)"])
    out = pd.DataFrame({"hit": hits, "n_hist": n, "grupo": bins}).groupby("grupo", observed=True).agg(
        usuarios=("hit", "size"), min_hist=("n_hist", "min"), max_hist=("n_hist", "max"),
        **{f"HitRate@{ctx.k}": ("hit", "mean")}).reset_index()
    return out


# --------------------------------------------------------------------------- usuarios fríos
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def choose_beta(ctx: Context, exclude, seed: int = 99, n_onb: int = 3) -> tuple[float, pd.DataFrame]:
    """Elige beta (peso del contenido en el fallback) con usuarios fríos DISJUNTOS de los de prueba."""
    cs = cold_user_split(ctx.ml.ratings, frac=0.10, n_onboarding=n_onb, seed=seed, exclude=exclude)
    popc = PopularityCount().fit(cs.train, ctx.index)
    dummy = MatrixFactorization(factors=1, epochs=0).fit(cs.train.head(1), ctx.index)
    rows = []
    for b in BETAS:
        h = HybridRecommender(dummy, ctx.content, popc, beta=b)
        res = evaluate_ranking(h, cs.train, cs.test, ctx.index, ctx.content.X, ctx.k, extra_history=cs.onboarding)
        rows.append({"beta": b, **res})
    df = pd.DataFrame(rows)
    best = df[f"HitRate@{ctx.k}"].max()
    se = np.sqrt(best * (1 - best) / max(1, len(cs.test)))
    ok = df[df[f"HitRate@{ctx.k}"] >= best - se]
    beta = float(ok.sort_values(["Cobertura"], ascending=False).beta.iloc[0])  # misma regla: 1 EE + cobertura
    return beta, df


def cold_start(ctx: Context, f: int, e: int, alpha: float, gamma: float, beta: float,
               seeds=config.SEEDS, n_onb: int = 3, log=print):
    rows = []
    for s in seeds:
        cs = cold_user_split(ctx.ml.ratings, frac=0.10, n_onboarding=n_onb, seed=s)
        pops = PopularitySmoothed().fit(cs.train, ctx.index)
        popc = PopularityCount().fit(cs.train, ctx.index)
        mf = MatrixFactorization(factors=f, epochs=e, seed=s).fit(cs.train, ctx.index)
        hyb = HybridRecommender(mf, ctx.content, popc, alpha=alpha, gamma=gamma, beta=beta)
        empty = cs.onboarding.iloc[0:0]
        scen = [
            ("0 valoraciones", "Aleatorio", RandomRecommender(seed=s).fit(cs.train, ctx.index), empty),
            ("0 valoraciones", "Popularidad (conteo)", popc, empty),
            ("0 valoraciones", "Popularidad suavizada", pops, empty),
            ("0 valoraciones", "Híbrido FINAL → fallback", hyb, empty),
            (f"{n_onb} valoraciones", "Popularidad (conteo)", popc, cs.onboarding),
            (f"{n_onb} valoraciones", "Contenido", ctx.content, cs.onboarding),
            (f"{n_onb} valoraciones", "Híbrido FINAL → fallback", hyb, cs.onboarding),
        ]
        for esc, name, model, onb in scen:
            res = evaluate_ranking(model, cs.train, cs.test, ctx.index, ctx.content.X, ctx.k,
                                   extra_history=onb, return_lists=True)
            src = pd.Series(res.pop("_sources")).dropna()
            res.pop("_lists")
            rows.append({"escenario": esc, "modelo": name, "seed": s, **res,
                         "fuente": src.mode().iloc[0] if len(src) else "—"})
        rows.append({"escenario": "cualquiera", "modelo": "MF sola", "seed": s, "usuarios_eval": len(cs.test),
                     "fuente": f"sin vector para {sum(not mf.knows_user(u) for u in cs.cold_users)}/{len(cs.cold_users)} usuarios"})
        log(f"  frío seed={s}: {len(cs.cold_users)} usuarios fríos")
    raw = pd.DataFrame(rows)
    agg = _agg(raw.drop(columns=["fuente"]), ["escenario", "modelo"])
    agg = agg.merge(raw.groupby(["escenario", "modelo"], sort=False, dropna=False).fuente.first().reset_index(),
                    on=["escenario", "modelo"])
    return raw, agg


# --------------------------------------------------------------------------- perfiles
def profiles(ctx: Context, split, mf, pop, alpha: float, gamma: float, beta: float) -> pd.DataFrame:
    full_tr = merge_train_valid(split)
    hist = histories(full_tr, ctx.index)
    n = full_tr.groupby("userId").size()
    counts = item_counts(full_tr, ctx.index)
    u_big, u_small = int(n.idxmax()), int(n.idxmin())
    cases = [
        ("Historial amplio", u_big, hist[u_big], alpha, gamma),
        ("Historial amplio", u_big, hist[u_big], 0.25, 0.0),
        ("Historial amplio", u_big, hist[u_big], 0.75, 0.0),
        ("Historial pequeño", u_small, hist[u_small], alpha, gamma),
        ("Usuario nuevo", None, History(), alpha, gamma),
    ]
    out = []
    for perfil, uid, h, a, g in cases:
        hyb = HybridRecommender(mf, ctx.content, pop, alpha=a, gamma=g, beta=beta)
        recs = hyb.recommend(uid, h, k=ctx.k, tiebreak=counts)
        seen = set(h.items.tolist())
        for rank, p in enumerate(recs, 1):
            out.append({"perfil": perfil, "userId": uid if uid is not None else "nuevo",
                        "n_historial": len(h), "alpha": a, "gamma": g, "fuente": hyb.last_source,
                        "personalizada": hyb.personalized, "rank": rank,
                        "movieId": int(ctx.index.movie_ids[p]), "title": ctx.index.titles[p],
                        "genres": ctx.index.genres[p], "visto_antes": p in seen})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- figuras
def figures(ctx: Context, val_df: pd.DataFrame, test_agg: pd.DataFrame, mf, chosen) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors  # noqa: F401
    import matplotlib.pyplot as plt

    figdir = ctx.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    hr = f"HitRate@{ctx.k}"
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Pareto (validación): acierto vs cobertura; tamaño/color = costo de cómputo
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    d = val_df.copy()
    costs = np.sort(d["costo_computo"].unique())
    cmap = plt.get_cmap("viridis_r")
    cnorm = matplotlib.colors.LogNorm(vmin=costs.min(), vmax=costs.max())
    dom = d[~d.pareto]
    ax.scatter(dom["Cobertura"], dom[hr], s=14, c="#c9d1d9", zorder=2, label="dominadas")
    nd = d[d.pareto]
    sc = ax.scatter(nd["Cobertura"], nd[hr], s=34, c=nd["costo_computo"], cmap=cmap, norm=cnorm,
                    edgecolors="black", linewidths=0.4, zorder=3, label="no dominadas (Pareto)")
    eq = d[d.cobertura_equivalente]
    ax.scatter(eq["Cobertura"], eq[hr], s=110, facecolors="none", edgecolors="#2da44e", lw=1.2, zorder=4,
               label="equivalentes (≤1 EE y cobertura ≥90 %)")
    ax.scatter([chosen["Cobertura"]], [chosen[hr]], s=230, facecolors="none", edgecolors="#cf222e", lw=2, zorder=5,
               label="elegida (la más barata)")
    ax.axhline(d.attrs["best_hr"] - d.attrs["se"], color="#d29922", ls="--", lw=1, label="mejor HitRate − 1 EE")
    def lab(r):
        return r["modelo"] if r["factores"] == 0 else f"MF f{int(r['factores'])}/e{int(r['epocas'])} α{r['alpha']:g} γ{r['gamma']:g}"
    notes = list(d[d.factores == 0].index) + [chosen.name]
    big = eq.sort_values("costo_computo").index
    if len(big) > 1:
        notes.append(big[-1])
    for i in notes:
        r = d.loc[i]
        ax.annotate(lab(r), (r["Cobertura"], r[hr]), fontsize=6.5, xytext=(5, -9 if i == chosen.name else 4),
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlim(right=d["Cobertura"].max() * 1.6)
    ax.set_xlabel("Cobertura del catálogo (log)")
    ax.set_ylabel(f"{hr} (validación, media de 3 semillas)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("costo: factores × épocas × valoraciones", fontsize=7)
    ax.set_title("Validación: acierto, cobertura y costo de 93 configuraciones", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout(); fig.savefig(figdir / "pareto_validacion.png"); plt.close(fig)

    # 2. Efecto de alpha y gamma (configuración de MF elegida)
    sub = d[(d.factores == chosen["factores"]) & (d.epocas == chosen["epocas"])].sort_values("alpha")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for g, col in zip(sorted(sub.gamma.dropna().unique()), ["#1f6feb", "#8250df", "#d29922", "#2da44e"]):
        q = sub[sub.gamma == g]
        ax1.errorbar(q.alpha, q[hr], yerr=q[f"{hr}_sd"], marker="o", ms=3, color=col, capsize=2, label=f"γ={g:g}")
        ax2.plot(q.alpha, q["Cobertura"], marker="o", ms=3, color=col, label=f"γ={g:g}")
    pc = d[d.modelo == "Popularidad (conteo)"].iloc[0]
    ax1.axhline(pc[hr], color="gray", ls=":", lw=1, label="popularidad (conteo)")
    ax1.set_xlabel("alpha (peso colaborativo)"); ax1.set_ylabel(f"{hr} validación"); ax1.legend(fontsize=6, frameon=False)
    ax2.set_xlabel("alpha (peso colaborativo)"); ax2.set_ylabel("Cobertura del catálogo")
    fig.suptitle(f"Híbrido: acierto y cobertura según alpha y gamma (MF f={int(chosen['factores'])}, ep={int(chosen['epocas'])})", fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "alpha_tradeoff.png"); plt.close(fig)

    # 3. Curva de entrenamiento MF
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(mf.history_) + 1), mf.history_, marker=".", color="#1f6feb")
    ax.set_xlabel("época"); ax.set_ylabel("RMSE entrenamiento"); ax.set_title("Convergencia MF-SGD (modelo final)")
    fig.tight_layout(); fig.savefig(figdir / "mf_training_curve.png"); plt.close(fig)

    # 4. Cola larga
    c = np.sort(ctx.ml.ratings.groupby("movieId").size().to_numpy())[::-1]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(c) + 1), c, color="#1f6feb")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rango de la película (log)"); ax.set_ylabel("nº valoraciones (log)")
    ax.set_title("Cola larga de popularidad")
    fig.tight_layout(); fig.savefig(figdir / "long_tail.png"); plt.close(fig)

    # 5. Prueba: HitRate vs cobertura por modelo
    t = test_agg.dropna(subset=[hr])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(t["Cobertura"], t[hr], yerr=t[f"{hr}_EE"], fmt="o", color="#1f6feb", capsize=2, zorder=3)
    ax.set_xscale("log")
    for _, r in t.iterrows():
        ax.annotate(r["modelo"], (r["Cobertura"], r[hr]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Cobertura del catálogo (log)"); ax.set_ylabel(f"{hr} (prueba, ± 1 EE)")
    ax.set_title("Prueba: acierto vs cobertura")
    fig.tight_layout(); fig.savefig(figdir / "test_hit_vs_coverage.png"); plt.close(fig)


# --------------------------------------------------------------------------- orquestador
def run_all(ml: MovieLens, out: Path = config.REPORTS, quick: bool = False, seeds=config.SEEDS,
            log=print) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    index = ItemIndex(ml.movies)
    content = ContentRecommender().fit(None, index, ml.movies)
    ctx = Context(ml, index, content, out)
    k = ctx.k
    grid = QUICK_GRID if quick else FULL_GRID

    log("[1/6] LAB08: auditoría, popularidad y contenido")
    l8 = lab08(ctx)

    log("[2/6] Corte temporal leave-last-out")
    split = leave_last_out(ml.ratings)
    split_info = pd.DataFrame([
        {"particion": "entrenamiento", "valoraciones": len(split.train), "usuarios": split.train.userId.nunique()},
        {"particion": "validación (penúltima)", "valoraciones": len(split.valid), "usuarios": split.valid.userId.nunique()},
        {"particion": "prueba (última)", "valoraciones": len(split.test), "usuarios": split.test.userId.nunique()},
    ])
    _save(split_info, out, "split_summary.csv")
    leak = (split.test.merge(split.train.groupby("userId").timestamp.max().rename("tmax"), on="userId")
            .eval("timestamp >= tmax").all())
    test_items_unseen = float((~split.test.movieId.isin(merge_train_valid(split).movieId)).mean())

    log("[3/6] Selección en validación (3 semillas)")
    val_raw = validation_grid(ctx, split, seeds, grid, log)
    _save(val_raw, out, "validation_runs.csv")
    val_agg = _agg(val_raw, ["modelo", "factores", "epocas", "alpha", "gamma"])
    chosen, val_agg = choose_config(val_agg, k, split.valid.userId.nunique())
    _save(val_agg.round(5), out, "validation_pareto.csv")
    if chosen.factores == 0:  # ganó un baseline: se documenta y se sigue evaluando el mejor híbrido
        log(f"      ATENCIÓN: la regla elige el baseline {chosen.modelo}")
        chosen = val_agg[val_agg.factores > 0].sort_values(f"HitRate@{k}", ascending=False).iloc[0]
    f, e, a, g = int(chosen.factores), int(chosen.epocas), float(chosen.alpha), float(chosen.gamma)
    lab = val_agg[(val_agg.factores == f) & (val_agg.epocas == e) & (val_agg.gamma == 0) & (val_agg.alpha < 1)]
    lab_alpha = float(lab.sort_values(f"HitRate@{k}", ascending=False).alpha.iloc[0])
    log(f"      elegida: {chosen.modelo} factores={f} épocas={e} alpha={a:g} gamma={g:g}; híbrido LAB09 alpha={lab_alpha:g}")

    log("[4/6] Evaluación única en prueba")
    test_raw, test_agg, mf, pop = test_phase(ctx, split, f, e, a, g, lab_alpha, seeds, log)
    _save(test_raw, out, "test_runs.csv")
    _save(test_agg.round(5), out, "test_results.csv")

    # comparación pareada del modelo final contra los baselines más fuertes (3 semillas)
    full_tr = merge_train_valid(split)
    pairs = []
    for s_ in seeds:
        mf_s = MatrixFactorization(factors=f, epochs=e, seed=s_).fit(full_tr, index)
        h_fin = hits_by_user(HybridRecommender(mf_s, content, pop, alpha=a, gamma=g), full_tr, split.test, index, k)
        for nm, mdl in [("Popularidad (conteo)", pop), ("MF sola", mf_s),
                        (f"Híbrido LAB09 α={lab_alpha:g} γ=0", HybridRecommender(mf_s, content, pop, alpha=lab_alpha))]:
            pairs.append({"A": "Híbrido FINAL", "B": nm, "seed": s_,
                          **paired_comparison(h_fin, hits_by_user(mdl, full_tr, split.test, index, k))})
    pairs = pd.DataFrame(pairs)
    _save(pairs.round(5), out, "paired_tests.csv")

    log("[5/6] Usuarios fríos")
    test_cold = set()
    for s_ in seeds:
        test_cold |= set(cold_user_split(ml.ratings, frac=0.10, seed=s_).cold_users.tolist())
    beta, beta_df = choose_beta(ctx, exclude=test_cold)
    _save(beta_df.round(5), out, "cold_beta_validation.csv")
    log(f"      beta elegido en validación fría disjunta: {beta:g}")
    hyb = HybridRecommender(mf, content, pop, alpha=a, gamma=g, beta=beta)
    buckets = history_buckets(ctx, split, hyb)
    _save(buckets.round(5), out, "history_size_buckets.csv")
    cold_raw, cold_agg = cold_start(ctx, f, e, a, g, beta, seeds, log=log)
    _save(cold_raw, out, "cold_start_runs.csv")
    _save(cold_agg.round(5), out, "cold_start.csv")

    log("[6/6] Perfiles, figuras y resumen")
    prof = profiles(ctx, split, mf, pop, a, g, beta)
    _save(prof, out, "profiles.csv")
    figures(ctx, val_agg, test_agg, mf, chosen)

    summary = {
        "protocolo": "v2 (gamma + regla Pareto/1EE/cobertura; ver reports/v1_protocolo_inicial)",
        "decision": {"modelo": str(chosen.modelo), "factores": f, "epocas": e, "alpha": a, "gamma": g,
                     "beta_frio": beta, "alpha_hibrido_lab09": lab_alpha,
                     f"val_HitRate@{k}": float(chosen[f"HitRate@{k}"]),
                     "mejor_val_HitRate": float(val_agg.attrs["best_hr"]), "1EE": float(val_agg.attrs["se"]),
                     "regla": "Pareto (HR@10, cobertura vs cómputo, parámetros) + 1 EE en HR + cobertura ≥ 90 % del máximo → la más barata",
                     "popularidad_en_hibrido_y_fallback": "Popularidad (conteo)"},
        "split": {"sin_fuga_temporal_por_usuario": bool(leak),
                  "pct_items_prueba_no_vistos_en_train": test_items_unseen},
        "seeds": list(seeds), "k": k, "quick": quick,
        "pop_m": l8["pop_m"],
        "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "segundos_totales": round(time.perf_counter() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Listo en {summary['segundos_totales']} s → {out}")
    return {"lab08": l8, "split": split, "split_info": split_info, "val": val_agg, "chosen": chosen,
            "test": test_agg, "test_raw": test_raw, "pairs": pairs, "buckets": buckets, "cold": cold_agg, "beta": beta_df,
            "profiles": prof, "summary": summary, "mf": mf}
