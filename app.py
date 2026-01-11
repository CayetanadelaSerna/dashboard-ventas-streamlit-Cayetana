import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------
# Config general
# --------------------------
st.set_page_config(
    page_title="Dashboard de Ventas | Cierre de Año",
    page_icon="📊",
    layout="wide",
)

st.title("Dashboard de Ventas - Visión Global y Análisis Detallado")
st.caption("Fuente interna: dataset corporativo (archivos parte_1.csv y parte_2.csv)")

# --------------------------
# Carga y preparación de datos
# --------------------------
@st.cache_data(show_spinner=True)
def load_data():
    # Rutas (en local y en Streamlit Cloud funcionará igual si mantienes /data)
    paths = ["data/parte_1.csv.gz", "data/parte_2.csv.gz"]

    dfs = []
    for p in paths:
        df = pd.read_csv(p, compression="gzip", low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Limpieza
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Tipos para ahorrar memoria
    cat_cols = [
        "family", "holiday_type", "locale", "locale_name", "description",
        "transferred", "city", "state", "store_type", "day_of_week"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Asegurar numéricos
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0).astype("int64")
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")

    # Columnas año/mes/semana por si hubiera nulos
    if "year" not in df.columns or df["year"].isna().any():
        df["year"] = df["date"].dt.year
    if "month" not in df.columns or df["month"].isna().any():
        df["month"] = df["date"].dt.month
    if "week" not in df.columns or df["week"].isna().any():
        df["week"] = df["date"].dt.isocalendar().week.astype("int64")

    return df

df = load_data()

# --------------------------
# Helpers
# --------------------------
def kpi_card(label: str, value):
    st.metric(label, value)

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x

# Para transacciones: evitar doble conteo (se repiten por familia)
@st.cache_data(show_spinner=False)
def transactions_base(df_):
    cols = ["date", "store_nbr", "state", "transactions", "year"]
    base = df_[cols].drop_duplicates(subset=["date", "store_nbr"])
    return base

tx_base = transactions_base(df)

# --------------------------
# Tabs (Pestañas)
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Pestaña 1 · Global", "Pestaña 2 · Tienda", "Pestaña 3 · Estado", "Pestaña 4 · Insights"])

# =========================================================
# TAB 1 - GLOBAL
# =========================================================
with tab1:
    st.subheader("1) Visión global del periodo")

    # KPIs generales
    total_stores = df["store_nbr"].nunique()
    total_products = df["family"].nunique()
    total_states = df["state"].nunique()

    # Meses disponibles (por año-mes únicos)
    months_available = df[["year", "month"]].drop_duplicates().shape[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Número total de tiendas", safe_int(total_stores))
    with c2: kpi_card("Número total de productos (familias)", safe_int(total_products))
    with c3: kpi_card("Estados", safe_int(total_states))
    with c4: kpi_card("Meses con datos", safe_int(months_available))

    st.divider()

    st.subheader("2) Análisis (rankings y distribuciones)")

    # Top 10 productos por ventas (suma)
    top_products = (
        df.groupby("family", observed=True)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_top_products = px.bar(top_products, x="sales", y="family", orientation="h", title="Top 10 productos más vendidos (ventas totales)")
    fig_top_products.update_layout(yaxis={"categoryorder": "total ascending"})
    
    # Distribución ventas por tienda (top 30 para legibilidad)
    sales_by_store = (
        df.groupby("store_nbr")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .reset_index()
    )
    fig_store_dist = px.bar(sales_by_store, x="store_nbr", y="sales", title="Distribución de ventas por tienda (Top 30 por ventas)")
    fig_store_dist.update_xaxes(type="category")

    # Top 10 tiendas con ventas en promoción
    promo_df = df[df["onpromotion"] > 0].copy()
    promo_sales_by_store = (
        promo_df.groupby("store_nbr")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_promo_stores = px.bar(promo_sales_by_store, x="sales", y="store_nbr", orientation="h",
                              title="Top 10 tiendas con ventas en productos en promoción (ventas)")
    fig_promo_stores.update_layout(yaxis={"categoryorder": "total ascending"})

    a, b = st.columns(2)
    with a:
        st.plotly_chart(fig_top_products, width='stretch')
    with b:
        st.plotly_chart(fig_promo_stores, width='stretch')

    st.plotly_chart(fig_store_dist, width='stretch')

    st.divider()
    st.subheader("3) Estacionalidad de ventas")

    # Día de la semana con más ventas por término medio
    dow_avg = (
        df.groupby("day_of_week", observed=True)["sales"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_dow = px.bar(dow_avg, x="day_of_week", y="sales", title="Ventas medias por día de la semana")
    fig_dow.update_xaxes(type="category")

    # Ventas medias por semana del año (promedio sobre años)
    week_avg = (
        df.groupby("week")["sales"]
        .mean()
        .reset_index()
        .sort_values("week")
    )
    fig_week = px.line(week_avg, x="week", y="sales", title="Ventas medias por semana del año (promedio de todos los años)")

    # Ventas medias por mes (promedio sobre años)
    month_avg = (
        df.groupby("month")["sales"]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    fig_month = px.line(month_avg, x="month", y="sales", title="Ventas medias por mes (promedio de todos los años)")
    fig_month.update_xaxes(type="category")

    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(fig_dow, width='stretch')
    with c2: st.plotly_chart(fig_week, width='stretch')
    with c3: st.plotly_chart(fig_month, width='stretch')

# =========================================================
# TAB 2 - TIENDA
# =========================================================
with tab2:
    st.subheader("Análisis por tienda")

    stores = sorted(df["store_nbr"].unique().tolist())
    store_selected = st.selectbox("Selecciona una tienda (store_nbr):", stores)

    df_store = df[df["store_nbr"] == store_selected].copy()

    # a) Ventas totales por año
    sales_by_year = (
        df_store.groupby("year")["sales"]
        .sum()
        .reset_index()
        .sort_values("year")
    )
    fig_sales_year = px.bar(sales_by_year, x="year", y="sales", title=f"Tienda {store_selected}: ventas totales por año")
    fig_sales_year.update_xaxes(type="category")

    # b) Nº total de productos vendidos (familias con ventas > 0)
    products_sold = df_store[df_store["sales"] > 0]["family"].nunique()

    # c) Nº total de productos vendidos que estaban en promoción
    promo_products_sold = df_store[(df_store["sales"] > 0) & (df_store["onpromotion"] > 0)]["family"].nunique()

    m1, m2, m3 = st.columns(3)
    with m1: kpi_card("Productos vendidos (familias con ventas > 0)", safe_int(products_sold))
    with m2: kpi_card("Productos vendidos en promoción (familias)", safe_int(promo_products_sold))
    with m3:
        promo_sales = df_store[df_store["onpromotion"] > 0]["sales"].sum()
        kpi_card("Ventas en promoción (suma sales)", float(promo_sales))

    st.plotly_chart(fig_sales_year, width='stretch')

    # Extra útil: Top productos de esa tienda
    top_store_products = (
        df_store.groupby("family", observed=True)["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_top_store_products = px.bar(
        top_store_products, x="sales", y="family", orientation="h",
        title=f"Tienda {store_selected}: Top 10 productos por ventas"
    )
    fig_top_store_products.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top_store_products, width='stretch')

# =========================================================
# TAB 3 - ESTADO
# =========================================================
with tab3:
    st.subheader("Análisis por estado")

    states = sorted(df["state"].dropna().unique().tolist())
    state_selected = st.selectbox("Selecciona un estado (state):", states)

    df_state = df[df["state"] == state_selected].copy()
    tx_state = tx_base[tx_base["state"] == state_selected].copy()

    # a) Transacciones por año (evitando doble conteo)
    tx_year = (
        tx_state.groupby("year")["transactions"]
        .sum(min_count=1)
        .reset_index()
        .sort_values("year")
    )
    fig_tx_year = px.bar(tx_year, x="year", y="transactions", title=f"{state_selected}: transacciones totales por año")
    fig_tx_year.update_xaxes(type="category")

    # b) Ranking de tiendas con más ventas
    sales_store_rank = (
        df_state.groupby("store_nbr")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_rank = px.bar(sales_store_rank, x="sales", y="store_nbr", orientation="h",
                      title=f"{state_selected}: Top 10 tiendas por ventas")
    fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})

    # c) Producto más vendido en la tienda (tabla clara: para cada tienda, su top producto)
    top_product_per_store = (
        df_state.groupby(["store_nbr", "family"], observed=True)["sales"]
        .sum()
        .reset_index()
        .sort_values(["store_nbr", "sales"], ascending=[True, False])
    )
    top_product_per_store = top_product_per_store.groupby("store_nbr").head(1)
    top_product_per_store = top_product_per_store.sort_values("sales", ascending=False).head(15)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig_tx_year, width='stretch')
        st.plotly_chart(fig_rank, width='stretch')

    with right:
        st.markdown("**Producto más vendido por tienda (Top 15 tiendas por ventas del producto líder)**")
        st.dataframe(
            top_product_per_store.rename(columns={"sales": "ventas_top_producto"}),
            width='stretch',
            hide_index=True
        )

# =========================================================
# TAB 4 - INSIGHTS (SORPRESA)
# =========================================================
with tab4:
    st.subheader("Insights rápidos para conclusiones (Pestaña 4)")

    st.markdown("Esta pestaña está pensada para acelerar conclusiones sobre promociones, festivos y drivers externos.")

    # 1) Impacto de promociones: % ventas en promo y comparación
    total_sales = df["sales"].sum()
    promo_sales = df[df["onpromotion"] > 0]["sales"].sum()
    promo_share = (promo_sales / total_sales) * 100 if total_sales > 0 else 0

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Ventas totales", float(total_sales))
    with c2:
        kpi_card("% ventas con promoción", round(promo_share, 2))

    promo_compare = (
        df.assign(promo_flag=np.where(df["onpromotion"] > 0, "Con promoción", "Sin promoción"))
        .groupby("promo_flag")["sales"]
        .mean()
        .reset_index()
    )
    fig_promo_compare = px.bar(promo_compare, x="promo_flag", y="sales", title="Ventas medias: con vs sin promoción")
    fig_promo_compare.update_xaxes(type="category")

    # 2) Festivos: ventas medias por tipo de festivo
    holiday_avg = (
        df.groupby("holiday_type", observed=True)["sales"]
        .mean()
        .reset_index()
        .sort_values("sales", ascending=False)
    )
    fig_holiday = px.bar(holiday_avg, x="holiday_type", y="sales", title="Ventas medias por tipo de festivo")
    fig_holiday.update_xaxes(type="category")

    # 3) Driver externo: dcoilwtico vs ventas (por mes para suavizar)
    oil_ok = df.dropna(subset=["dcoilwtico"]).copy()
    oil_month = (
        oil_ok.groupby(["year", "month"], observed=True)
        .agg(avg_oil=("dcoilwtico", "mean"), total_sales=("sales", "sum"))
        .reset_index()
    )
    oil_month["year_month"] = oil_month["year"].astype(str) + "-" + oil_month["month"].astype(str).str.zfill(2)
    fig_oil = px.scatter(oil_month, x="avg_oil", y="total_sales", title="Relación (mensual): precio petróleo vs ventas")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(fig_promo_compare, width='stretch')
        st.plotly_chart(fig_holiday, width='stretch')
    with b:
        st.plotly_chart(fig_oil, width='stretch')



