import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re
import plotly.graph_objects as go

st.set_page_config(page_title="Valencia Dashboard", layout="wide")
st.title("Valencia CF — Análisis estadístico")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Numeric basics
    for col in ["puntos", "goles_valencia", "goles_rival", "diferencia_goles"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize strings
    if "temporada" in df.columns:
        df["temporada"] = df["temporada"].astype(str)

    for col in ["rival", "condicion", "franja"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fix mojibake en 'franja'
    if "franja" in df.columns:
        df["franja"] = df["franja"].replace({
            "MediodÃ­a": "Mediodía",
            "MEDIODÃ�A": "Mediodía",
            "mediodÃ­a": "Mediodía"
        })
    return df

DATA_PATH = "valencia_partidos_enriquecido.csv"
if not Path(DATA_PATH).exists():
    st.error(f"No se encuentra el archivo '{DATA_PATH}' en el directorio actual.")
    st.stop()

df_full = load_data(DATA_PATH)

# ---------- Helpers ----------
def temporada_start_year(s: str) -> int | None:
    s = str(s)
    m1 = re.fullmatch(r"(\d{4})[-/](\d{4})", s)
    if m1:
        return int(m1.group(1))
    m2 = re.fullmatch(r"(\d{2})[-/](\d{2})", s)
    if m2:
        a = int(m2.group(1))
        return 1900 + a if a >= 70 else 2000 + a
    try:
        return int(s[:4])
    except Exception:
        return None

def sort_temporadas(temporadas: list[str]) -> list[str]:
    return sorted(temporadas, key=lambda t: (temporada_start_year(t) or 0, t))

def infer_result_letter_from_row(row) -> str:
    # V/E/D desde resultado_valencia o puntos
    if "resultado_valencia" in row and pd.notna(row["resultado_valencia"]):
        s = str(row["resultado_valencia"]).lower()
        if s.startswith("vic"): return "V"
        if s.startswith("emp"): return "E"
        if s.startswith("der"): return "D"
    if "puntos" in row and pd.notna(row["puntos"]):
        if row["puntos"] == 3: return "V"
        if row["puntos"] == 1: return "E"
        if row["puntos"] == 0: return "D"
    return "?"

def infer_result_series(df: pd.DataFrame) -> pd.Series:
    if "resultado_valencia" in df.columns and df["resultado_valencia"].notna().any():
        s = df["resultado_valencia"].astype(str).str.strip().str.lower()
        res = np.select(
            [s.str.startswith("vic"), s.str.startswith("emp"), s.str.startswith("der")],
            ["V", "E", "D"],
            default=None
        )
    else:
        res = np.array([None] * len(df))
    if "puntos" in df.columns:
        pts = pd.to_numeric(df["puntos"], errors="coerce")
        res = np.where(
            res == None,
            np.where(pts == 3, "V", np.where(pts == 1, "E", np.where(pts == 0, "D", "?"))),
            res
        )
    return pd.Series(res, index=df.index, name="resultado_simplificado")

def format_score_with_letter(row) -> str:
    gf = int(row.get("goles_valencia", 0)) if pd.notna(row.get("goles_valencia")) else 0
    gc = int(row.get("goles_rival", 0)) if pd.notna(row.get("goles_rival")) else 0
    letter = infer_result_letter_from_row(row)
    return f"{gf}–{gc} ({letter})"

def compute_kpis(df: pd.DataFrame) -> dict:
    pj = int(len(df))
    pts = float(df["puntos"].sum()) if "puntos" in df.columns else np.nan
    ppg = (pts / pj) if pj > 0 else np.nan
    gf = float(df["goles_valencia"].sum()) if "goles_valencia" in df.columns else np.nan
    gc = float(df["goles_rival"].sum()) if "goles_rival" in df.columns else np.nan
    dg = (gf - gc) if (not np.isnan(gf) and not np.isnan(gc)) else np.nan
    res = infer_result_series(df)
    v = int((res == "V").sum())
    e = int((res == "E").sum())
    d = int((res == "D").sum())
    pv = (v / pj * 100) if pj > 0 else np.nan
    return {
        "PJ": pj, "Puntos": pts, "PPG": ppg, "%V": pv,
        "GF": gf, "GC": gc, "DG": dg, "V": v, "E": e, "D": d
    }
# ---- derivar 'vuelta' desde 'fase_temporada' (1=Ida, 2=Vuelta) ----
if "fase_temporada" in df_full.columns:
    df_full["vuelta"] = (
        pd.to_numeric(df_full["fase_temporada"], errors="coerce")
        .map({1: "Ida", 2: "Vuelta"})
    )

# ---------- Filtros (sidebar) ----------
st.sidebar.header("Filtros")

# Construir la lista de temporadas (ordenada)
temporadas = (
    sort_temporadas(df_full["temporada"].dropna().unique().tolist())
    if "temporada" in df_full.columns else []
)

sel_temporadas = st.sidebar.multiselect("Temporada(s)", temporadas, default=[])


condiciones = sorted(df_full["condicion"].dropna().unique().tolist()) if "condicion" in df_full.columns else []
sel_condicion = st.sidebar.multiselect("Condición", condiciones, default=[])

# --- Ida / Vuelta (usar columna ya creada) ---
vueltas = sorted(df_full["vuelta"].dropna().unique().tolist()) if "vuelta" in df_full.columns else []
sel_vuelta = st.sidebar.multiselect("Ida/Vuelta", vueltas, default=[])


rivales = sorted(df_full["rival"].dropna().unique().tolist()) if "rival" in df_full.columns else []
sel_rivales = st.sidebar.multiselect("Rival(es)", rivales, default=[])

franjas_raw = sorted(df_full.get("franja", pd.Series(dtype=str)).dropna().unique().tolist()) if "franja" in df_full.columns else []
franjas = [f for f in franjas_raw if f not in ("Desconocida", "nan")]
sel_franjas = st.sidebar.multiselect("Franja horaria (desde 2019)", franjas, default=[]) if franjas else []

use_dates = False
if "fecha" in df_full.columns and df_full["fecha"].notna().any():
    use_dates = st.sidebar.checkbox("Filtrar por fechas", value=False)
    if use_dates:
        min_date = df_full["fecha"].min().date()
        max_date = df_full["fecha"].max().date()
        date_range = st.sidebar.date_input("Rango de fechas", (min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            f_ini, f_fin = date_range
        else:
            f_ini, f_fin = min_date, max_date
    else:
        f_ini = f_fin = None
else:
    f_ini = f_fin = None

# Aplicar filtros
df = df_full.copy()
if sel_temporadas:
    df = df[df["temporada"].isin(sel_temporadas)]
if sel_condicion:
    df = df[df["condicion"].isin(sel_condicion)]
if sel_rivales:
    df = df[df["rival"].isin(sel_rivales)]
if sel_franjas and "franja" in df.columns:
    df = df[df["franja"].isin(sel_franjas)]
if use_dates and f_ini and f_fin and "fecha" in df.columns:
    df = df[(df["fecha"] >= pd.to_datetime(f_ini)) & (df["fecha"] <= pd.to_datetime(f_fin))]

if "vuelta" in df.columns and len(sel_vuelta) > 0:
    df = df[df["vuelta"].isin(sel_vuelta)]

# ------- tabs -------
tab_resumen, tab_rivales, tab_records = st.tabs(["Resumen estadístico", "Enfrentamientos", "Datos destacados"])

# ======= Resumen =======
with tab_resumen:
    st.subheader("Indicadores generales")
    k = compute_kpis(df)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Partidos", f"{k['PJ']}")
    c2.metric("Puntos", f"{k['Puntos']:.0f}" if pd.notna(k['Puntos']) else "—")
    c3.metric("Puntos por partido", f"{k['PPG']:.2f}" if pd.notna(k['PPG']) else "—")
    c4.metric("% de victorias", f"{k['%V']:.1f}%" if pd.notna(k['%V']) else "—")
    c5.metric("Goles a favor", f"{k['GF']:.0f}" if pd.notna(k['GF']) else "—")
    c6.metric("Goles en contra", f"{k['GC']:.0f}" if pd.notna(k['GC']) else "—")
    c7.metric("Diferencia de goles", f"{k['DG']:.0f}" if pd.notna(k['DG']) else "—")

    st.divider()

    # 1) Puntos totales por temporada
    st.markdown("### Puntos totales por temporada")
    if len(df) > 0 and "puntos" in df.columns and "temporada" in df.columns:
        by_temp_pts = df.groupby("temporada", as_index=False).agg(Puntos=("puntos", "sum"))
        by_temp_pts = by_temp_pts.sort_values(by="temporada", key=lambda s: s.map(temporada_start_year).fillna(0))
        fig_pts = px.bar(by_temp_pts, x="temporada", y="Puntos",
                         labels={"temporada": "Temporada", "Puntos": "Puntos"})
        fig_pts.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_pts, use_container_width=True)
    else:
        st.info("No hay datos suficientes para calcular puntos por temporada.")

    # 2) V/E/D por temporada (recuento)
    st.markdown("### Victorias, empates y derrotas por temporada")
    if len(df) > 0 and "temporada" in df.columns:
        res = infer_result_series(df)
        temp_res = df.assign(Res=res).groupby(["temporada", "Res"]).size().reset_index(name="Partidos")
        temp_res["Res"] = temp_res["Res"].map({"V": "Victoria", "E": "Empate", "D": "Derrota", "?": "Desconocido"})
        temp_res = temp_res.sort_values(by="temporada", key=lambda s: s.map(temporada_start_year).fillna(0))
        fig_bar = px.bar(temp_res, x="temporada", y="Partidos", color="Res",
                         labels={"temporada": "Temporada", "Partidos": "Partidos", "Res": "Resultado"})
        fig_bar.update_layout(barmode="stack", margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-45, height=460)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hay datos suficientes para calcular resultados por temporada.")

    # 3) Matriz
    st.markdown("### Matriz de marcadores (Goles VCF × Goles Rival)")
    if "goles_valencia" in df.columns and "goles_rival" in df.columns and len(df) > 0:
        mat = pd.crosstab(df["goles_valencia"], df["goles_rival"])
        if mat.size == 0:
            st.info("No hay datos suficientes para la matriz.")
        else:
            fig_heat = px.imshow(mat, text_auto=True, aspect="equal",
                                 labels=dict(x="Goles Rival", y="Goles Valencia", color="Partidos"))
            fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=500)
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No hay datos suficientes para generar la matriz de marcadores.")

    st.divider()
    st.markdown("### Lista de partidos según filtros")
    if len(df) > 0:
        show_cols_pref = [
            "fecha", "temporada", "competicion", "jornada_num", "condicion",
            "rival", "goles_valencia", "goles_rival", "puntos",
            "resultado_valencia", "franja", "hora"
        ]
        show_cols = [c for c in show_cols_pref if c in df.columns] + [c for c in df.columns if c in ["partido_id"]]
        st.dataframe(
            df[show_cols].sort_values(by=["fecha", "temporada"] if "fecha" in df.columns else ["temporada"]),
            use_container_width=True, height=420
        )
        csv_bytes = df[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV filtrado", data=csv_bytes, file_name="partidos_filtrados.csv", mime="text/csv")
    else:
        st.info("No hay partidos con los filtros actuales.")

# ======= Rivales =======
with tab_rivales:
    st.subheader("Resumen por rival")
    if "rival" not in df_full.columns:
        st.warning("No existe columna 'rival' en el dataset.")
        st.stop()

    df_rank = df.copy()
    res_s = infer_result_series(df_rank)
    df_rank = df_rank.assign(Res=res_s)

    agg = df_rank.groupby("rival").agg(
        PJ=("puntos", "count"),
        Puntos=("puntos", "sum"),
        GF=("goles_valencia", "sum"),
        GC=("goles_rival", "sum"),
        V=("Res", lambda s: (s == "V").sum()),
        E=("Res", lambda s: (s == "E").sum()),
        D=("Res", lambda s: (s == "D").sum()),
    ).reset_index()
    agg["PPG"] = agg["Puntos"] / agg["PJ"]
    agg["DG"] = agg["GF"] - agg["GC"]
    agg["%V"] = np.where(agg["PJ"] > 0, agg["V"] / agg["PJ"] * 100, np.nan)
    agg = agg.sort_values(by=["PJ", "PPG"], ascending=[False, False])

    st.dataframe(agg, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Descargar ranking rivales",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name="ranking_rivales.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Detalles del rival seleccionado")

    rivals_list = agg["rival"].tolist()
    if len(rivals_list) == 0:
        st.info("No hay rivales para mostrar con los filtros actuales.")
        st.stop()

    rival_sel = st.selectbox("Elige un rival", rivals_list, index=0)
    df_rival = df_rank[df_rank["rival"] == rival_sel]

    k_rival = compute_kpis(df_rival)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Partidos", f"{k_rival['PJ']}")
    c2.metric("Puntos por partido", f"{k_rival['PPG']:.2f}" if pd.notna(k_rival['PPG']) else "—")
    c3.metric("% de victorias", f"{k_rival['%V']:.1f}%" if pd.notna(k_rival['%V']) else "—")
    c4.metric("Victorias", f"{k_rival['V']}")
    c5.metric("Empates", f"{k_rival['E']}")
    c6.metric("Derrotas", f"{k_rival['D']}")
    c7.metric("Diferencia de goles", f"{k_rival['DG']:.0f}" if pd.notna(k_rival['DG']) else "—")

    st.markdown("#### Rendimiento como local y visitante")
    if "condicion" in df_rival.columns:
        split = df_rival.groupby("condicion").agg(
            PJ=("puntos", "count"),
            Puntos=("puntos", "sum"),
            GF=("goles_valencia", "sum"),
            GC=("goles_rival", "sum"),
        ).reset_index()
        split["Puntos por partido"] = split["Puntos"] / split["PJ"]
        split["Diferencia de goles"] = split["GF"] - split["GC"]
        st.dataframe(split, use_container_width=True, height=180)
    else:
        st.info("No existe columna 'condicion' para mostrar el split.")

    st.markdown("#### Evolución del rendimiento frente al rival (PPG) — comparación directa")
    
    if "temporada" in df_rival.columns and "puntos" in df_rival.columns and len(df_rival) > 0:
        tmp = df_rival.copy()
    
        # Calcular puntos del rival (desde el punto de vista del Valencia)
        tmp["puntos_rival"] = np.select(
            [
                tmp["puntos"] == 3,
                tmp["puntos"] == 1,
                tmp["puntos"] == 0
            ],
            [0, 1, 3],
            default=np.nan
        )
    
        # Agrupar por temporada
        agg = tmp.groupby("temporada", as_index=False).agg(
            Puntos_VCF=("puntos", "sum"),
            Puntos_Rival=("puntos_rival", "sum"),
            PJ=("puntos", "count")
        )
        agg["PPG_VCF"] = agg["Puntos_VCF"] / agg["PJ"]
        agg["PPG_Rival"] = agg["Puntos_Rival"] / agg["PJ"]
    
        # Ordenar cronológicamente
        agg = agg.sort_values(by="temporada", key=lambda s: s.map(temporada_start_year).fillna(0))
        x = agg["temporada"]
    
        # Crear gráfico
        fig = go.Figure()
    
        # Línea Valencia
        fig.add_trace(go.Scatter(
            x=x, y=agg["PPG_VCF"],
            mode="lines+markers",
            name="Valencia",
            line=dict(color="#003366", width=3),
            marker=dict(size=6),
            hovertemplate="Temporada: %{x}<br>Valencia PPG: %{y:.2f}<extra></extra>"
        ))
    
        # Línea rival
        fig.add_trace(go.Scatter(
            x=x, y=agg["PPG_Rival"],
            mode="lines+markers",
            name=rival_sel,
            line=dict(color="#1F77B4", width=3, dash="dot"),
            marker=dict(size=6),
            hovertemplate=f"Temporada: %{{x}}<br>{rival_sel} PPG: %{{y:.2f}}<extra></extra>"
        ))
    
        # Añadir sombreado donde Valencia supera al rival
        fig.add_trace(go.Scatter(
            x=x,
            y=np.maximum(agg["PPG_VCF"], agg["PPG_Rival"]),
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=np.minimum(agg["PPG_VCF"], agg["PPG_Rival"]),
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor="rgba(0,102,204,0.15)",
            showlegend=False,
            hoverinfo='skip'
        ))
    
        # Diseño del gráfico
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Temporada",
            yaxis_title="Puntos por partido (PPG)",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template="plotly_white"
        )
    
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No hay datos suficientes para calcular PPG por temporada frente a este rival.")


    st.markdown("#### Resultados frente a este rival (matriz)")
    if "goles_valencia" in df_rival.columns and "goles_rival" in df_rival.columns and len(df_rival) > 0:
        mat_r = pd.crosstab(df_rival["goles_valencia"], df_rival["goles_rival"])
        if mat_r.size == 0:
            st.info("No hay datos suficientes para la matriz vs rival.")
        else:
            fig_heat_r = px.imshow(mat_r, text_auto=True, aspect="equal",
                                   labels=dict(x="Goles Rival", y="Goles Valencia", color="Partidos"))
            fig_heat_r.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
            st.plotly_chart(fig_heat_r, use_container_width=True)
    else:
        st.info("No hay suficientes datos para la matriz de marcadores vs el rival seleccionado.")

# ======= Datos destacados =======
with tab_records:
    st.subheader("")

    # ---------- Goles y resultados extremos (tabla completa) ----------
    st.markdown("### Goles y resultados extremos")

    def get_row_dict(idx):
        try:
            return df.loc[idx].to_dict()
        except Exception:
            return {}

    records_rows = []

    if len(df) > 0:
        # 1) Máximo goles a favor
        if "goles_valencia" in df.columns:
            idx = df["goles_valencia"].idxmax()
            r = get_row_dict(idx)
            records_rows.append({
                "Categoría": "Partido con más goles a favor del Valencia",
                "Resultado": format_score_with_letter(r),
                "Rival": r.get("rival", ""),
                "Temporada": r.get("temporada", ""),
                "Condición": r.get("condicion", ""),
                "Información adicional": ""
            })
        # 2) Máximo goles en contra
        if "goles_rival" in df.columns:
            idx = df["goles_rival"].idxmax()
            r = get_row_dict(idx)
            records_rows.append({
                "Categoría": "Partido con más goles en contra",
                "Resultado": format_score_with_letter(r),
                "Rival": r.get("rival", ""),
                "Temporada": r.get("temporada", ""),
                "Condición": r.get("condicion", ""),
                "Información adicional": ""
            })
        # 3) Más goles totales
        if "goles_valencia" in df.columns and "goles_rival" in df.columns:
            tmp = df.assign(_tot=df["goles_valencia"] + df["goles_rival"])
            idx = tmp["_tot"].idxmax()
            r = get_row_dict(idx)
            records_rows.append({
                "Categoría": "Partido con más goles totales",
                "Resultado": format_score_with_letter(r),
                "Rival": r.get("rival", ""),
                "Temporada": r.get("temporada", ""),
                "Condición": r.get("condicion", ""),
                "Información adicional": f"Goles totales: {int(r.get('goles_valencia',0)+r.get('goles_rival',0))}"
            })
        # 4) Mayor victoria
        if "diferencia_goles" in df.columns:
            idx = df["diferencia_goles"].idxmax()
            r = get_row_dict(idx)
            records_rows.append({
                "Categoría": "Mayor victoria del Valencia",
                "Resultado": format_score_with_letter(r),
                "Rival": r.get("rival", ""),
                "Temporada": r.get("temporada", ""),
                "Condición": r.get("condicion", ""),
                "Información adicional": f"Diferencia de goles: +{int(r.get('diferencia_goles',0))}"
            })
            # 5) Derrota más abultada
            idx2 = df["diferencia_goles"].idxmin()
            r2 = get_row_dict(idx2)
            records_rows.append({
                "Categoría": "Derrota más abultada",
                "Resultado": format_score_with_letter(r2),
                "Rival": r2.get("rival", ""),
                "Temporada": r2.get("temporada", ""),
                "Condición": r2.get("condicion", ""),
                "Información adicional": f"Diferencia de goles: {int(r2.get('diferencia_goles',0))}"
            })

    if records_rows:
        df_table = pd.DataFrame(records_rows, columns=["Categoría","Resultado","Rival","Temporada","Condición","Información adicional"])
        st.dataframe(df_table, use_container_width=True)

    st.divider()

    # ---------- Remontadas y giros de marcador ----------
    st.markdown("### Remontadas y giros de marcador")

    def compute_remontada_df(dfin: pd.DataFrame):
        needed = {"goles_descanso_valencia", "goles_descanso_rival", "goles_valencia", "goles_rival"}
        if not needed.issubset(set(dfin.columns)):
            return None
        dfu = dfin.copy()
        for c in list(needed):
            dfu[c] = pd.to_numeric(dfu[c], errors="coerce")
        dfu["delta_final"] = dfu["goles_valencia"] - dfu["goles_rival"]
        dfu["delta_descanso"] = dfu["goles_descanso_valencia"] - dfu["goles_descanso_rival"]
        dfu["cambio"] = dfu["delta_final"] - dfu["delta_descanso"]
        return dfu

    df_rem = compute_remontada_df(df)

    # Tabla para 2 categorías + texto para la 3ª
    rem_rows = []
    if df_rem is not None and len(df_rem) > 0:
        # Mayor remontada a favor
        idx_max = df_rem["cambio"].idxmax()
        r = df_rem.loc[idx_max].to_dict()
        row_full = df.loc[idx_max].to_dict()
        rem_rows.append({
            "Categoría": "Mayor remontada a favor (del descanso al final)",
            "Resultado": format_score_with_letter(row_full),
            "Rival": row_full.get("rival",""),
            "Temporada": row_full.get("temporada",""),
            "Condición": row_full.get("condicion",""),
            "Información adicional": f"Descanso {int(r['goles_descanso_valencia'])}-{int(r['goles_descanso_rival'])} → Final {int(r['goles_valencia'])}-{int(r['goles_rival'])} · Cambio: +{int(r['cambio'])}"
        })
        # Mayor remontada sufrida
        idx_min = df_rem["cambio"].idxmin()
        r2 = df_rem.loc[idx_min].to_dict()
        row_full2 = df.loc[idx_min].to_dict()
        rem_rows.append({
            "Categoría": "Mayor remontada sufrida (del descanso al final)",
            "Resultado": format_score_with_letter(row_full2),
            "Rival": row_full2.get("rival",""),
            "Temporada": row_full2.get("temporada",""),
            "Condición": row_full2.get("condicion",""),
            "Información adicional": f"Descanso {int(r2['goles_descanso_valencia'])}-{int(r2['goles_descanso_rival'])} → Final {int(r2['goles_valencia'])}-{int(r2['goles_rival'])} · Cambio: {int(r2['cambio'])}"
        })
        st.dataframe(pd.DataFrame(rem_rows, columns=["Categoría","Resultado","Rival","Temporada","Condición","Información adicional"]), use_container_width=True)

        # Texto: temporada con más remontadas
        tmp = df_rem.assign(remonto=(df_rem["cambio"] > 0))
        byt = tmp.groupby("temporada")["remonto"].sum().reset_index(name="Remontadas")
        if len(byt) > 0:
            top = byt.loc[byt["Remontadas"].idxmax()]
            st.markdown(f"Temporada con más remontadas a favor: **{top['temporada']} · Remontadas: {int(top['Remontadas'])}**")
    else:
        st.info("No hay datos de goles al descanso para calcular remontadas.")

    st.divider()

    # ---------- Contexto temporal ----------
    # Ocultar este bloque si no hay franja en el subset
    valid_franjas = {"Mediodía", "Tarde", "Vespertina", "Noche"}
    has_franja = ("franja" in df.columns) and df["franja"].isin(valid_franjas).any()
    has_hora = ("hora" in df.columns) and pd.to_datetime(df["hora"], errors="coerce").notna().any()

    if has_franja and has_hora:
        st.markdown("### Contexto temporal")

        # Tabla para 2 primeras
        ctx_rows = []
        if "hora" in df.columns and df["hora"].notna().any():
            hparsed = pd.to_datetime(df["hora"], errors="coerce").dt.time
            dfh = df.assign(_hora=hparsed).dropna(subset=["_hora"])

            def to_minutes(t):
                return t.hour * 60 + t.minute if t is not None else None
            dfh = dfh.assign(_mins=[to_minutes(t) for t in dfh["_hora"]])

            if len(dfh) > 0:
                idx_e = dfh["_mins"].idxmin()
                re = dfh.loc[idx_e].to_dict()
                ro = df.loc[idx_e].to_dict()
                ctx_rows.append({
                    "Categoría": "Partido más temprano",
                    "Resultado": format_score_with_letter(ro),
                    "Rival": ro.get("rival",""),
                    "Temporada": ro.get("temporada",""),
                    "Condición": ro.get("condicion",""),
                    "Información adicional": f"Hora: {re['_hora']} · Franja: {ro.get('franja','')}"
                })
                idx_l = dfh["_mins"].idxmax()
                rl = dfh.loc[idx_l].to_dict()
                ro2 = df.loc[idx_l].to_dict()
                ctx_rows.append({
                    "Categoría": "Partido más tardío",
                    "Resultado": format_score_with_letter(ro2),
                    "Rival": ro2.get("rival",""),
                    "Temporada": ro2.get("temporada",""),
                    "Condición": ro2.get("condicion",""),
                    "Información adicional": f"Hora: {rl['_hora']} · Franja: {ro2.get('franja','')}"
                })
        if ctx_rows:
            st.dataframe(pd.DataFrame(ctx_rows, columns=["Categoría","Resultado","Rival","Temporada","Condición","Información adicional"]), use_container_width=True)

        # Texto para las 2 últimas
        if "puntos" in df.columns:
            res_v = (pd.to_numeric(df["puntos"], errors="coerce") == 3)
            t = df.assign(V=res_v).groupby("franja").agg(
                PJ=("V", "count"),
                V=("V", "sum")
            ).reset_index()
            t["%Victorias"] = np.where(t["PJ"] > 0, t["V"] / t["PJ"] * 100, np.nan)
            t = t.sort_values("%Victorias", ascending=False)

            if len(t) > 0 and pd.notna(t.iloc[0]["%Victorias"]):
                st.markdown(
                    f"Franja con mayor % de victorias: **{t.iloc[0]['franja']} · "
                    f"{t.iloc[0]['%Victorias']:.1f}% ·  Partidos jugados: {int(t.iloc[0]['PJ'])}**"
                )
    else:
        st.info("Bloque de 'Contexto temporal' oculto: el filtro actual no contiene datos de franja.")

        # ---------- Rival (todo en texto con el formato dado) ----------
    st.markdown("### Rival")
    
    if "rival" in df.columns and len(df) > 0:
        agg_r = df.groupby("rival").agg(
             PJ=("puntos", "count"),
            GF=("goles_valencia", "mean"),
            GC=("goles_rival", "mean"),
            PPG=("puntos", "mean")
        ).reset_index()
    
            # Mínimo de partidos para evitar outliers
        agg_r = agg_r[agg_r["PJ"] >= 3] if len(agg_r) else agg_r

        if len(agg_r) > 0:
            # Rival que más goles nos hace (promedio GC)
            r_gc = agg_r.loc[agg_r["GC"].idxmax()]
            st.markdown(
                f"Rival que más goles nos hace (promedio Goles a Favor): "
                f"**{r_gc['rival']} · Goles a favor medio: {r_gc['GC']:.2f} (Partidos Jugados: {int(r_gc['PJ'])})**"
            )

            # Rival más goleado por el VCF (promedio GF del VCF)
            r_gf = agg_r.loc[agg_r["GF"].idxmax()]
            st.markdown(
                f"Rival más goleado (promedio Goles a Favor del VCF): "
                f"**{r_gf['rival']} · Goles a favor medio: {r_gf['GF']:.2f} (Partidos Jugados: {int(r_gf['PJ'])})**"
            )

            # Mejor PPG contra un rival
            r_ppg = agg_r.loc[agg_r["PPG"].idxmax()]
            st.markdown(
                f"Mejor Puntos por partido (PPG) contra un rival: "
                f"**{r_ppg['rival']} · PPG: {r_ppg['PPG']:.2f} (Partidos Jugados: {int(r_ppg['PJ'])})**"
            )
        else:
            st.info("No hay suficientes partidos por rival para calcular promedios (mín. 3 PJ).")
    else:
        st.info("No hay datos de rivales para este filtro.")

        st.divider()

    # ---------- Curiosidades adicionales (todo texto) ----------
    st.markdown("### Curiosidades adicionales")

    # Temporada con mejor PPG
    if "temporada" in df.columns and "puntos" in df.columns and len(df) > 0:
        by_t = df.groupby("temporada").agg(Puntos=("puntos", "sum"), PJ=("puntos", "count")).reset_index()
        by_t["PPG"] = by_t["Puntos"] / by_t["PJ"]
        best_ppg = by_t.loc[by_t["PPG"].idxmax()]
        st.markdown(
            f"Temporada con mejor Puntos por partido (PPG): "
            f"**{best_ppg['temporada']} · PPG: {best_ppg['PPG']:.2f}**"
        )

    # Temporada con más derrotas
    # Temporada con más derrotas (oculta si el máximo es 0 o NaN)
    res_der = None
    if "resultado_valencia" in df.columns and df["resultado_valencia"].notna().any():
        s = df["resultado_valencia"].astype(str).str.lower().str.strip()
        res_der = s.str.startswith("der")
    elif "puntos" in df.columns:
        res_der = (pd.to_numeric(df["puntos"], errors="coerce") == 0)
    
    if res_der is not None and "temporada" in df.columns:
        t_der = df.assign(D=res_der).groupby("temporada")["D"].sum().reset_index(name="Derrotas")
        if len(t_der) > 0:
            worst = t_der.loc[t_der["Derrotas"].idxmax()]
            if pd.notna(worst["Derrotas"]) and int(worst["Derrotas"]) > 0:
                st.markdown(f"Temporada con más derrotas: **{worst['temporada']} · Derrotas: {int(worst['Derrotas'])}**")
            # si el máximo es 0, no mostramos nada

    # Empates 0-0
    if "goles_valencia" in df.columns and "goles_rival" in df.columns:
        emp00 = df[
            (pd.to_numeric(df["goles_valencia"], errors="coerce") == 0) &
            (pd.to_numeric(df["goles_rival"], errors="coerce") == 0)
        ]
        st.markdown(f"Empates 0-0: **{len(emp00)}**")

    # Porterías a cero (rival no marca)
    if "goles_rival" in df.columns:
        porterias_cero = (pd.to_numeric(df["goles_rival"], errors="coerce") == 0).sum()
        st.markdown(f"Porterías a cero (rival no marca): **{int(porterias_cero)}**")

    # Victorias por 3+ goles de diferencia
    if "diferencia_goles" in df.columns:
        goleadas = (pd.to_numeric(df["diferencia_goles"], errors="coerce") >= 3).sum()
        st.markdown(f"Victorias por 3+ goles de diferencia: **{int(goleadas)}**")

    # Temporada más goleadora (goles a favor)
    if "temporada" in df.columns and "goles_valencia" in df.columns and len(df) > 0:
        tmp = df.groupby("temporada")["goles_valencia"].sum().reset_index()
        top = tmp.loc[tmp["goles_valencia"].idxmax()]
        st.markdown(f"Temporada más goleadora (goles a favor): **{top['temporada']} · Goles: {int(top['goles_valencia'])}**")
    

    
           
