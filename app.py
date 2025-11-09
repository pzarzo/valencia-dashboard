import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Valencia Dashboard", layout="wide")
st.title("📊 Valencia CF — Dashboard (MVP)")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # fechas
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    # num
    for col in ["puntos","goles_valencia","goles_rival","diferencia_goles"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # texto
    if "temporada" in df.columns:
        df["temporada"] = df["temporada"].astype(str)
    for col in ["rival","condicion","franja"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # fix Mediodía
    if "franja" in df.columns:
        df["franja"] = df["franja"].replace({
            "MediodÃ­a":"Mediodía","MEDIODÃ�A":"Mediodía","mediodÃ­a":"Mediodía"
        })
    return df

DATA_PATH = "valencia_partidos_enriquecido.csv"
if not Path(DATA_PATH).exists():
    st.error(f"No se encuentra el archivo '{DATA_PATH}' en el directorio actual.")
    st.stop()

df_full = load_data(DATA_PATH)

# ------- helpers -------
def temporada_start_year(s: str) -> int | None:
    s = str(s)
    import re
    m1 = re.fullmatch(r"(\d{4})[-/](\d{4})", s)
    if m1: return int(m1.group(1))
    m2 = re.fullmatch(r"(\d{2})[-/](\d{2})", s)
    if m2:
        a = int(m2.group(1))
        return 1900 + a if a >= 70 else 2000 + a
    try: return int(s[:4])
    except: return None

def sort_temporadas(lst: list[str]) -> list[str]:
    return sorted(lst, key=lambda t: (temporada_start_year(t) or 0, t))

def infer_result_series(df: pd.DataFrame) -> pd.Series:
    if "resultado_valencia" in df.columns and df["resultado_valencia"].notna().any():
        s = df["resultado_valencia"].astype(str).str.strip().str.lower()
        res = np.select(
            [s.str.startswith("vic"), s.str.startswith("emp"), s.str.startswith("der")],
            ["V","E","D"],
            default=None
        )
    else:
        res = np.array([None]*len(df))
    if "puntos" in df.columns:
        pts = pd.to_numeric(df["puntos"], errors="coerce")
        res = np.where(res==None, np.where(pts==3,"V", np.where(pts==1,"E", np.where(pts==0,"D","?"))), res)
    return pd.Series(res, index=df.index, name="resultado_simplificado")

def compute_kpis(df: pd.DataFrame) -> dict:
    pj = len(df)
    pts = float(df["puntos"].sum()) if "puntos" in df.columns else np.nan
    ppg = (pts/pj) if pj>0 else np.nan
    gf = float(df["goles_valencia"].sum()) if "goles_valencia" in df.columns else np.nan
    gc = float(df["goles_rival"].sum()) if "goles_rival" in df.columns else np.nan
    dg = (gf-gc) if (not np.isnan(gf) and not np.isnan(gc)) else np.nan
    res = infer_result_series(df)
    v = (res=="V").sum(); e = (res=="E").sum(); d = (res=="D").sum()
    pv = (v/pj*100) if pj>0 else np.nan
    return {"PJ":pj,"Puntos":pts,"PPG":ppg,"%V":pv,"GF":gf,"GC":gc,"DG":dg,"V":v,"E":e,"D":d}

# ------- sidebar filtros -------
st.sidebar.header("⚙️ Filtros")

temporadas = sort_temporadas(df_full["temporada"].dropna().unique().tolist()) if "temporada" in df_full.columns else []
temporadas_opts = ["Todas"] + temporadas
sel_temporadas = st.sidebar.multiselect("Temporada(s)", temporadas_opts, default=[])
if "Todas" in sel_temporadas:
    sel_temporadas = temporadas

condiciones = sorted(df_full["condicion"].dropna().unique().tolist()) if "condicion" in df_full.columns else []
sel_condicion = st.sidebar.multiselect("Condición", condiciones, default=[])

rivales = sorted(df_full["rival"].dropna().unique().tolist()) if "rival" in df_full.columns else []
sel_rivales = st.sidebar.multiselect("Rival(es)", rivales, default=[])

franjas = sorted(df_full.get("franja", pd.Series(dtype=str)).dropna().unique().tolist()) if "franja" in df_full.columns else []
sel_franjas = st.sidebar.multiselect("Franja (desde 2019)", franjas, default=[]) if franjas else []

use_dates = False
if "fecha" in df_full.columns and df_full["fecha"].notna().any():
    use_dates = st.sidebar.checkbox("Filtrar por fechas", value=False)
    if use_dates:
        min_date = df_full["fecha"].min().date()
        max_date = df_full["fecha"].max().date()
        date_range = st.sidebar.date_input("Rango de fechas", (min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range)==2:
            f_ini, f_fin = date_range
        else:
            f_ini, f_fin = min_date, max_date
    else:
        f_ini = f_fin = None
else:
    f_ini = f_fin = None

# aplicar filtros
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


# ------- tabs -------
tab_resumen, tab_rivales, tab_records = st.tabs(["📌 Resumen", "🤝 Rivales (Head-to-Head)", "📈 Datos destacados"])

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
    st.markdown("### 📈 Puntos totales por temporada")
    if len(df)>0 and "puntos" in df.columns and "temporada" in df.columns:
        by_temp_pts = df.groupby("temporada", as_index=False).agg(Puntos=("puntos","sum"))
        by_temp_pts = by_temp_pts.sort_values(by="temporada", key=lambda s: s.map(temporada_start_year).fillna(0))
        fig_pts = px.bar(by_temp_pts, x="temporada", y="Puntos",
                         labels={"temporada":"Temporada","Puntos":"Puntos"})
        fig_pts.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_pts, use_container_width=True)
    else:
        st.info("No hay datos suficientes para calcular puntos por temporada.")

    # 2) Victorias/Empates/Derrotas por temporada (recuento)
    st.markdown("### 🟩 Victorias, empates y derrotas por temporada")
    if len(df)>0 and "temporada" in df.columns:
        res = infer_result_series(df)
        temp_res = df.assign(Res=res).groupby(["temporada","Res"]).size().reset_index(name="Partidos")
        temp_res["Res"] = temp_res["Res"].map({"V":"Victoria","E":"Empate","D":"Derrota","?":"Desconocido"})
        temp_res = temp_res.sort_values(by="temporada", key=lambda s: s.map(temporada_start_year).fillna(0))
        fig_bar = px.bar(temp_res, x="temporada", y="Partidos", color="Res",
                         labels={"temporada":"Temporada","Partidos":"Partidos","Res":"Resultado"})
        fig_bar.update_layout(barmode="stack", margin=dict(l=10,r=10,t=10,b=10), xaxis_tickangle=-45, height=460)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hay datos suficientes para calcular resultados por temporada.")

    # 3) Matriz de marcadores
    st.markdown("### 🔥 Matriz de marcadores (Goles VCF × Goles Rival)")
    if "goles_valencia" in df.columns and "goles_rival" in df.columns and len(df)>0:
        mat = pd.crosstab(df["goles_valencia"], df["goles_rival"])
        if mat.size == 0:
            st.info("No hay datos suficientes para la matriz.")
        else:
            fig_heat = px.imshow(mat, text_auto=True, aspect="equal",
                                 labels=dict(x="Goles Rival", y="Goles Valencia", color="Partidos"))
            fig_heat.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=500)
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No hay datos suficientes para generar la matriz de marcadores.")

    st.divider()
    st.markdown("### 🗂️ Lista de partidos según filtros")
    if len(df)>0:
        show_cols_pref = ["fecha","temporada","competicion","jornada_num","condicion","rival","goles_valencia","goles_rival","puntos","resultado_valencia","franja","hora"]
        show_cols = [c for c in show_cols_pref if c in df.columns] + [c for c in df.columns if c in ["partido_id"]]
        st.dataframe(df[show_cols].sort_values(by=["fecha","temporada"] if "fecha" in df.columns else ["temporada"]), use_container_width=True, height=420)
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
        PJ=("puntos","count"),
        Puntos=("puntos","sum"),
        GF=("goles_valencia","sum"),
        GC=("goles_rival","sum"),
        V=("Res", lambda s: (s=="V").sum()),
        E=("Res", lambda s: (s=="E").sum()),
        D=("Res", lambda s: (s=="D").sum()),
    ).reset_index()
    agg["PPG"] = agg["Puntos"]/agg["PJ"]
    agg["DG"] = agg["GF"]-agg["GC"]
    agg["%V"] = np.where(agg["PJ"]>0, agg["V"]/agg["PJ"]*100, np.nan)
    agg = agg.sort_values(by=["PJ","PPG"], ascending=[False, False])

    st.dataframe(agg, use_container_width=True, height=420)
    st.download_button("⬇️ Descargar ranking rivales", data=agg.to_csv(index=False).encode("utf-8"),
                       file_name="ranking_rivales.csv", mime="text/csv")

    st.divider()
    st.subheader("Detalles del rival seleccionado")

    rivals_list = agg["rival"].tolist()
    if len(rivals_list)==0:
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
            PJ=("puntos","count"),
            Puntos=("puntos","sum"),
            GF=("goles_valencia","sum"),
            GC=("goles_rival","sum"),
        ).reset_index()
        split["Puntos por partido"] = split["Puntos"]/split["PJ"]
        split["Diferencia de goles"] = split["GF"] - split["GC"]
        st.dataframe(split, use_container_width=True, height=180)
    else:
        st.info("No existe columna 'condicion' para mostrar el split.")

    st.markdown("#### Evolución del rendimiento frente al rival (PPG)")
    if "temporada" in df_rival.columns and len(df_rival)>0:
        by_temp_r = df_rival.groupby("temporada", as_index=False).agg(Puntos=("puntos","sum"), PJ=("puntos","count"))
        by_temp_r["PPG"] = by_temp_r["Puntos"]/by_temp_r["PJ"]
        by_temp_r = by_temp_r.sort_values(by="temporada", key=lambda s: s.map(temporada_start_year).fillna(0))
        fig_line_r = px.line(by_temp_r, x="temporada", y="PPG", markers=True,
                             labels={"temporada":"Temporada","PPG":"Puntos por partido"})
        fig_line_r.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_tickangle=-45, height=340)
        st.plotly_chart(fig_line_r, use_container_width=True)
    else:
        st.info("No hay datos de temporadas suficientes para este rival.")

    st.markdown("#### Resultados frente a este rival (matriz)")
    if "goles_valencia" in df_rival.columns and "goles_rival" in df_rival.columns and len(df_rival)>0:
        mat_r = pd.crosstab(df_rival["goles_valencia"], df_rival["goles_rival"])
        if mat_r.size == 0:
            st.info("No hay datos suficientes para la matriz vs rival.")
        else:
            fig_heat_r = px.imshow(mat_r, text_auto=True, aspect="equal",
                                   labels=dict(x="Goles Rival", y="Goles Valencia", color="Partidos"))
            fig_heat_r.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=420)
            st.plotly_chart(fig_heat_r, use_container_width=True)
    else:
        st.info("No hay suficientes datos para la matriz de marcadores vs el rival seleccionado.")

    # ======= Datos destacados =======
    with tab_records:
        st.subheader("Goles y resultados extremos")

        def show_match_record(title: str, row):
            if row is None or len(row)==0:
                st.write(f"**{title}:** —")
                return
            vals = []
            def fmt_score(r):
                try:
                    return f"{int(r.get('goles_valencia', 0))}-{int(r.get('goles_rival', 0))}"
                except Exception:
                    return f"{r.get('goles_valencia')} - {r.get('goles_rival')}"
            vals.append(f"**{title}:** {row.get('temporada','')} · {row.get('fecha','')} · {row.get('condicion','')} vs {row.get('rival','')} · Resultado: {fmt_score(row)}")
            extra = []
            if 'competicion' in row: extra.append(f"Competición: {row['competicion']}")
            if 'resultado_valencia' in row and pd.notna(row['resultado_valencia']): extra.append(f"Resultado: {row['resultado_valencia']}")
            if extra:
                vals.append(" · ".join(extra))
            st.write("  
".join(vals))

        df_rec = df.copy()

        # 1) Más goles a favor
        row_max_gf = None
        if "goles_valencia" in df_rec.columns and len(df_rec)>0:
            row_max_gf = df_rec.loc[df_rec["goles_valencia"].idxmax()].to_dict()
        show_match_record("Partido con más goles a favor del Valencia", row_max_gf)

        # 2) Más goles en contra
        row_max_gc = None
        if "goles_rival" in df_rec.columns and len(df_rec)>0:
            row_max_gc = df_rec.loc[df_rec["goles_rival"].idxmax()].to_dict()
        show_match_record("Partido con más goles en contra", row_max_gc)

        # 3) Más goles totales
        row_max_tot = None
        if "goles_valencia" in df_rec.columns and "goles_rival" in df_rec.columns and len(df_rec)>0:
            tmp = df_rec.assign(goles_totales = pd.to_numeric(df_rec["goles_valencia"], errors="coerce") + pd.to_numeric(df_rec["goles_rival"], errors="coerce"))
            row_max_tot = tmp.loc[tmp["goles_totales"].idxmax()].to_dict()
        show_match_record("Partido con más goles totales", row_max_tot)

        # 4) Mayor victoria (DG max)
        row_max_dg = None
        if "diferencia_goles" in df_rec.columns and len(df_rec)>0:
            row_max_dg = df_rec.loc[df_rec["diferencia_goles"].idxmax()].to_dict()
        show_match_record("Mayor victoria del Valencia", row_max_dg)

        # 5) Derrota más abultada (DG min)
        row_min_dg = None
        if "diferencia_goles" in df_rec.columns and len(df_rec)>0:
            row_min_dg = df_rec.loc[df_rec["diferencia_goles"].idxmin()].to_dict()
        show_match_record("Derrota más abultada", row_min_dg)

        st.divider()
        st.subheader("Remontadas y giros de marcador")

        # Remontadas usando goles al descanso si existen
        def compute_remontada_df(dfin: pd.DataFrame):
            needed = {"goles_descanso_valencia","goles_descanso_rival","goles_valencia","goles_rival"}
            if not needed.issubset(set(dfin.columns)):
                return None
            dfu = dfin.copy()
            for c in list(needed):
                dfu[c] = pd.to_numeric(dfu[c], errors="coerce")
            dfu["delta_final"] = dfu["goles_valencia"] - dfu["goles_rival"]
            dfu["delta_descanso"] = dfu["goles_descanso_valencia"] - dfu["goles_descanso_rival"]
            dfu["cambio"] = dfu["delta_final"] - dfu["delta_descanso"]
            return dfu

        df_rem = compute_remontada_df(df_rec)

        row_max_remonta = None
        if df_rem is not None and len(df_rem)>0:
            row_max_remonta = df_rem.loc[df_rem["cambio"].idxmax()].to_dict()
        show_match_record("Mayor remontada a favor (del descanso al final)", row_max_remonta)

        row_max_remontado = None
        if df_rem is not None and len(df_rem)>0:
            row_max_remontado = df_rem.loc[df_rem["cambio"].idxmin()].to_dict()
        show_match_record("Mayor remontada sufrida (del descanso al final)", row_max_remontado)

        if df_rem is not None and "temporada" in df_rem.columns:
            tmp = df_rem.assign(remonto = (df_rem["cambio"]>0))
            byt = tmp.groupby("temporada")["remonto"].sum().reset_index(name="Remontadas")
            if len(byt)>0:
                top_row = byt.loc[byt["Remontadas"].idxmax()]
                st.write(f"**Temporada con más remontadas a favor:** {top_row['temporada']} · Remontadas: {int(top_row['Remontadas'])}")

        st.divider()
        st.subheader("Contexto temporal")

        # horas
        if "hora" in df_rec.columns and df_rec["hora"].notna().any():
            # intentar parseo hora -> datetime.time
            hparsed = pd.to_datetime(df_rec["hora"], errors="coerce").dt.time
            dfh = df_rec.assign(_hora=hparsed).dropna(subset=["_hora"])
            if len(dfh)>0:
                row_early = dfh.loc[dfh["_hora"].astype(str).idxmin()].to_dict()
                show_match_record("Partido más temprano", row_early)
                row_late = dfh.loc[dfh["_hora"].astype(str).idxmax()].to_dict()
                show_match_record("Partido más tardío", row_late)

        # franja con más victorias (tasa)
        if "franja" in df_rec.columns and "puntos" in df_rec.columns and df_rec["franja"].notna().any():
            res = (pd.to_numeric(df_rec["puntos"], errors="coerce") == 3)
            t = df_rec.assign(V=res).groupby("franja").agg(PJ=("V","count"), V=("V","sum")).reset_index()
            t["%Victorias"] = np.where(t["PJ"]>0, t["V"]/t["PJ"]*100, np.nan)
            t = t.sort_values("%Victorias", ascending=False)
            if len(t)>0:
                st.write(f"**Franja con mayor % de victorias:** {t.iloc[0]['franja']} · {t.iloc[0]['%Victorias']:.1f}% (PJ: {int(t.iloc[0]['PJ'])})")

        # temporada más goleadora (GF total)
        if "temporada" in df_rec.columns and "goles_valencia" in df_rec.columns and len(df_rec)>0:
            tmp = df_rec.groupby("temporada")["goles_valencia"].sum().reset_index()
            top = tmp.loc[tmp["goles_valencia"].idxmax()]
            st.write(f"**Temporada más goleadora (goles a favor):** {top['temporada']} · Goles: {int(top['goles_valencia'])}")

        st.divider()
        st.subheader("Rival")

        if "rival" in df_rec.columns and len(df_rec)>0:
            # promedios por rival
            agg = df_rec.groupby("rival").agg(
                PJ=("puntos","count"),
                GF=("goles_valencia","mean"),
                GC=("goles_rival","mean"),
                PPG=("puntos","mean")
            ).reset_index()
            # umbral mínimo de partidos para evitar sesgos
            agg = agg[agg["PJ"]>=3] if "PJ" in agg.columns else agg

            if len(agg)>0:
                r_gf = agg.loc[agg["GF"].idxmax()]
                st.write(f"**Rival más goleado (promedio GF):** {r_gf['rival']} · GF medio: {r_gf['GF']:.2f} (PJ: {int(r_gf['PJ'])})")
                r_gc = agg.loc[agg["GC"].idxmax()]
                st.write(f"**Rival que más goles nos hace (promedio GC):** {r_gc['rival']} · GC medio: {r_gc['GC']:.2f} (PJ: {int(r_gc['PJ'])})")
                r_ppg = agg.loc[agg["PPG"].idxmax()]
                st.write(f"**Mejor Puntos por partido (PPG) contra un rival:** {r_ppg['rival']} · PPG: {r_ppg['PPG']:.2f} (PJ: {int(r_ppg['PJ'])})")

        st.divider()
        st.subheader("Curiosidades adicionales")

        # Mejor rendimiento por temporada (PPG)
        if "temporada" in df_rec.columns and "puntos" in df_rec.columns and len(df_rec)>0:
            by_t = df_rec.groupby("temporada").agg(Puntos=("puntos","sum"), PJ=("puntos","count")).reset_index()
            by_t["PPG"] = by_t["Puntos"]/by_t["PJ"]
            best = by_t.loc[by_t["PPG"].idxmax()]
            st.write(f"**Temporada con mejor Puntos por partido (PPG):** {best['temporada']} · PPG: {best['PPG']:.2f}")

        # Temporada con más derrotas
        res = None
        if "resultado_valencia" in df_rec.columns:
            res = df_rec["resultado_valencia"].astype(str).str.lower().str.startswith("der")
        elif "puntos" in df_rec.columns:
            res = (pd.to_numeric(df_rec["puntos"], errors="coerce") == 0)
        if res is not None and "temporada" in df_rec.columns:
            tmp = df_rec.assign(D=res).groupby("temporada")["D"].sum().reset_index(name="Derrotas")
            if len(tmp)>0:
                tmin = tmp.loc[tmp["Derrotas"].idxmax()]
                st.write(f"**Temporada con más derrotas:** {tmin['temporada']} · Derrotas: {int(tmin['Derrotas'])}")

        # Empates 0-0
        if "goles_valencia" in df_rec.columns and "goles_rival" in df_rec.columns:
            emp00 = df_rec[(pd.to_numeric(df_rec["goles_valencia"], errors="coerce")==0) & (pd.to_numeric(df_rec["goles_rival"], errors="coerce")==0)]
            st.write(f"**Empates 0-0:** {len(emp00)}")

        # Porterías a cero
        if "goles_rival" in df_rec.columns:
            clean = (pd.to_numeric(df_rec["goles_rival"], errors="coerce")==0).sum()
            st.write(f"**Porterías a cero (rival no marca):** {int(clean)}")

        # Victorias por goleada (3+ DG)
        if "diferencia_goles" in df_rec.columns:
            goleadas = (pd.to_numeric(df_rec['diferencia_goles'], errors="coerce")>=3).sum()
            st.write(f"**Victorias por 3+ goles de diferencia:** {int(goleadas)}")
