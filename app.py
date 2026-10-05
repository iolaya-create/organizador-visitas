
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

st.set_page_config(
    page_title="Organizador de Visitas",
    page_icon="📍",
    layout="wide"
)

# -----------------------------
# Utilidades
# -----------------------------
REQUIRED_COLUMNS = [
    "CUENTA", "SOLICITUD", "MUNICIPIO",
    "DIRECCION", "LONGITUD", "LATITUD"
]

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def detect_swapped_coordinates(df):
    """
    Detecta el caso típico de Colombia:
    LONGITUD contiene valores ~4.x y LATITUD ~-73.x.
    """
    lon = pd.to_numeric(df["LONGITUD"], errors="coerce").dropna()
    lat = pd.to_numeric(df["LATITUD"], errors="coerce").dropna()
    if len(lon) == 0 or len(lat) == 0:
        return False

    med_lon = float(lon.median())
    med_lat = float(lat.median())

    # Heurística general para columnas evidentemente invertidas.
    return (
        abs(med_lon) < 20 and abs(med_lat) > 20
    )

def haversine_km(center, points):
    center = np.radians(np.asarray(center, dtype=float))
    points = np.radians(np.asarray(points, dtype=float))
    dlat = points[:, 0] - center[0]
    dlon = points[:, 1] - center[1]
    h = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(center[0])
        * np.cos(points[:, 0])
        * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def choose_package_count(n, min_size, max_size):
    """
    El rango min_size-max_size es un objetivo, no una restricción absoluta.
    Se elige un número de paquetes que mantenga tamaños equilibrados.
    Los paquetes pequeños se permiten cuando la cantidad total no permite
    cumplir el rango o cuando el algoritmo geográfico los justifica.
    """
    if n <= 0:
        return 0, []

    # Para cantidades menores al objetivo, un solo paquete.
    if n <= max_size:
        return 1, [n]

    # Menor número de paquetes que respeta el máximo como objetivo.
    k_min = int(np.ceil(n / max_size))
    k_max = max(1, int(np.floor(n / min_size)))

    # Normalmente usamos el menor número de paquetes: favorece grupos grandes.
    # Si no existe una combinación exacta dentro del rango, se usa k_min.
    k = k_min

    base = n // k
    rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]

    return k, sizes

def balanced_geographical_clustering(coords, sizes, seed_trials=50):
    """
    Agrupa por coordenadas con capacidades exactas por paquete.
    Se usan múltiples inicializaciones KMeans y una asignación
    con capacidad exacta mediante Hungarian algorithm.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    k = len(sizes)

    mean_lat = np.radians(np.mean(coords[:, 0]))
    # Escala longitudinal para aproximar distancias en km.
    projected = np.column_stack([
        coords[:, 1] * np.cos(mean_lat),
        coords[:, 0]
    ])

    best = None
    trials = min(seed_trials, 80 if n <= 500 else 25)

    for seed in range(trials):
        km = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=5,
            max_iter=300
        )
        km.fit(coords)
        centers = km.cluster_centers_.copy()

        for _ in range(60):
            old_centers = centers.copy()

            centers_proj = np.column_stack([
                centers[:, 1] * np.cos(mean_lat),
                centers[:, 0]
            ])

            costs = np.empty((n, n), dtype=float)
            slot_package = []
            col = 0

            for p, size in enumerate(sizes):
                d2 = np.sum(
                    (projected - centers_proj[p]) ** 2,
                    axis=1
                )
                for _ in range(size):
                    costs[:, col] = d2
                    slot_package.append(p)
                    col += 1

            rows, cols = linear_sum_assignment(costs)
            labels = np.array([slot_package[c] for c in cols], dtype=int)

            centers = np.vstack([
                coords[labels == p].mean(axis=0)
                for p in range(k)
            ])

            if np.max(np.abs(centers - old_centers)) < 1e-10:
                break

        total_sq_km = 0.0
        max_radius = 0.0
        sum_radius = 0.0

        for p in range(k):
            pts = coords[labels == p]
            d = haversine_km(centers[p], pts)
            total_sq_km += float(np.sum(d ** 2))
            max_radius = max(max_radius, float(np.max(d)))
            sum_radius += float(np.max(d))

        # Prioriza compactación general y evita un paquete extremadamente
        # más disperso que los demás.
        score = total_sq_km + 150.0 * (max_radius ** 2) + 5.0 * sum_radius

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "labels": labels.copy(),
                "centers": centers.copy()
            }

    return best["labels"], best["centers"]

def organize_visits(df, min_size=12, max_size=15, seed_trials=50):
    df = df.copy()

    for col in ["LONGITUD", "LATITUD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Auto-corrección de coordenadas invertidas.
    swapped = detect_swapped_coordinates(df)
    if swapped:
        original_lon = df["LONGITUD"].copy()
        df["LONGITUD"] = df["LATITUD"]
        df["LATITUD"] = original_lon

    # Las visitas sin coordenadas NO bloquean el proceso.
    valid = (
        df["LATITUD"].between(-90, 90)
        & df["LONGITUD"].between(-180, 180)
        & df["LATITUD"].notna()
        & df["LONGITUD"].notna()
    )
    missing_coords = df.loc[~valid].copy()
    geo_df = df.loc[valid].copy()
    invalid_count = int((~valid).sum())

    # Si ninguna visita tiene GPS, todo se deja en un paquete especial.
    if len(geo_df) == 0:
        result = df.copy()
        result["PAQUETE"] = 1
        result["ORDEN"] = range(1, len(result) + 1)
        result["OBSERVACION"] = "REVISAR: SIN COORDENADAS"
        summary_df = pd.DataFrame([{
            "PAQUETE": 1,
            "VISITAS": len(result),
            "MUNICIPIOS": " | ".join(sorted(result["MUNICIPIO"].fillna("").astype(str).unique())),
            "DISTANCIA_PROMEDIO_KM": 0,
            "DISTANCIA_MAXIMA_KM": 0,
            "LATITUD_CENTRO": "",
            "LONGITUD_CENTRO": "",
            "ESTADO": "REVISAR",
            "ADVERTENCIA_DISTANCIA": "SIN COORDENADAS"
        }])
        control = pd.DataFrame({
            "CONTROL": ["TOTAL VISITAS", "VISITAS CON GPS", "VISITAS SIN GPS", "PAQUETES GEOGRÁFICOS", "PAQUETE ESPECIAL SIN COORDENADAS", "COORDENADAS INVERTIDAS DETECTADAS"],
            "VALOR": [len(df), 0, invalid_count, 0, 1, "SÍ" if swapped else "NO"]
        })
        return result, summary_df, control, swapped

    n = len(geo_df)
    k, sizes = choose_package_count(n, min_size, max_size)
    if k is None or k == 0:
        raise ValueError("No hay visitas con coordenadas suficientes para organizar.")

    coords = geo_df[["LATITUD", "LONGITUD"]].to_numpy(dtype=float)
    labels, centers = balanced_geographical_clustering(coords, sizes, seed_trials=seed_trials)

    package_order = sorted(range(k), key=lambda p: (centers[p, 0], centers[p, 1]))
    package_map = {old: new + 1 for new, old in enumerate(package_order)}
    geo_df["PAQUETE"] = [package_map[int(x)] for x in labels]
    geo_df["OBSERVACION"] = ""

    package_centers = {}
    for pnum in range(1, k + 1):
        pts = geo_df.loc[geo_df["PAQUETE"] == pnum, ["LATITUD", "LONGITUD"]].to_numpy(dtype=float)
        package_centers[pnum] = pts.mean(axis=0)

    distances = []
    for _, row in geo_df.iterrows():
        c = package_centers[int(row["PAQUETE"])]
        distances.append(float(haversine_km(c, np.array([[row["LATITUD"], row["LONGITUD"]]]))[0]))

    geo_df["_DISTANCIA_CENTRO_KM"] = distances
    geo_df = geo_df.sort_values(["PAQUETE", "_DISTANCIA_CENTRO_KM"]).copy()
    geo_df.insert(1, "ORDEN", geo_df.groupby("PAQUETE").cumcount() + 1)

    summary = []
    for pnum in range(1, k + 1):
        sub = geo_df[geo_df["PAQUETE"] == pnum]
        c = package_centers[pnum]
        d = sub["_DISTANCIA_CENTRO_KM"].to_numpy(dtype=float)
        summary.append({
            "PAQUETE": pnum,
            "VISITAS": len(sub),
            "MUNICIPIOS": " | ".join(sorted(sub["MUNICIPIO"].fillna("").astype(str).str.strip().replace("", "SIN MUNICIPIO").unique())),
            "DISTANCIA_PROMEDIO_KM": round(float(d.mean()), 3),
            "DISTANCIA_MAXIMA_KM": round(float(d.max()), 3),
            "LATITUD_CENTRO": round(float(c[0]), 6),
            "LONGITUD_CENTRO": round(float(c[1]), 6),
        })

    summary_df = pd.DataFrame(summary)
    summary_df["ESTADO"] = np.where(summary_df["VISITAS"] < min_size, "REDUCIDO", np.where(summary_df["VISITAS"] > max_size, "GRANDE", "NORMAL"))
    summary_df["ADVERTENCIA_DISTANCIA"] = np.where(summary_df["DISTANCIA_MAXIMA_KM"] >= 5, "REVISAR DESPLAZAMIENTO", "")

    # Todas las visitas sin coordenadas se agrupan juntas en un paquete especial.
    if invalid_count:
        special_package = k + 1
        missing_coords["PAQUETE"] = special_package
        missing_coords["ORDEN"] = range(1, len(missing_coords) + 1)
        missing_coords["OBSERVACION"] = "REVISAR: SIN COORDENADAS"
        missing_coords["_DISTANCIA_CENTRO_KM"] = 0
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "PAQUETE": special_package,
            "VISITAS": len(missing_coords),
            "MUNICIPIOS": " | ".join(sorted(missing_coords["MUNICIPIO"].fillna("").astype(str).str.strip().replace("", "SIN MUNICIPIO").unique())),
            "DISTANCIA_PROMEDIO_KM": 0,
            "DISTANCIA_MAXIMA_KM": 0,
            "LATITUD_CENTRO": "",
            "LONGITUD_CENTRO": "",
            "ESTADO": "REVISAR",
            "ADVERTENCIA_DISTANCIA": "SIN COORDENADAS"
        }])], ignore_index=True)
        result = pd.concat([geo_df.drop(columns=["_DISTANCIA_CENTRO_KM"]), missing_coords], ignore_index=True)
    else:
        result = geo_df.drop(columns=["_DISTANCIA_CENTRO_KM"])

    control = pd.DataFrame({
        "CONTROL": ["TOTAL VISITAS", "VISITAS CON GPS", "VISITAS SIN GPS", "PAQUETES GEOGRÁFICOS", "PAQUETE ESPECIAL SIN COORDENADAS", "TAMAÑO MÍNIMO OBJETIVO", "TAMAÑO MÁXIMO OBJETIVO", "COORDENADAS INVERTIDAS DETECTADAS"],
        "VALOR": [len(df), len(geo_df), invalid_count, k, (k + 1 if invalid_count else "NO"), min_size, max_size, "SÍ" if swapped else "NO"]
    })
    return result, summary_df, control, swapped

def excel_bytes(result, summary, control):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="VISITAS", index=False)
        summary.to_excel(writer, sheet_name="RESUMEN_PAQUETES", index=False)
        control.to_excel(writer, sheet_name="CONTROL", index=False)

        # Ajuste simple de anchos.
        for sheet_name in ["VISITAS", "RESUMEN_PAQUETES", "CONTROL"]:
            ws = writer.book[sheet_name]
            for column_cells in ws.columns:
                max_len = 0
                col_letter = column_cells[0].column_letter
                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = min(max(max_len, len(value)), 45)
                ws.column_dimensions[col_letter].width = max_len + 2

    return output.getvalue()

# -----------------------------
# Interfaz
# -----------------------------
st.title("📍 Organizador de Visitas")
st.write(
    "Carga un Excel y el aplicativo agrupará las visitas por proximidad "
    "geográfica. El objetivo es formar paquetes de 12–15 visitas, pero "
    "puede crear paquetes menores cuando la cantidad o la distribución "
    "geográfica lo hagan más lógico."
)

with st.sidebar:
    st.header("Configuración")
    min_size = st.number_input(
        "Mínimo de visitas por paquete",
        min_value=1,
        max_value=100,
        value=12,
        step=1
    )
    max_size = st.number_input(
        "Máximo de visitas por paquete",
        min_value=1,
        max_value=100,
        value=15,
        step=1
    )
    seed_trials = st.slider(
        "Calidad de búsqueda geográfica",
        min_value=10,
        max_value=80,
        value=50,
        step=10,
        help="Más pruebas pueden encontrar agrupaciones más compactas, pero tardan más."
    )

uploaded = st.file_uploader(
    "Sube el archivo Excel",
    type=["xlsx", "xls"],
    help="Columnas requeridas: CUENTA, SOLICITUD, MUNICIPIO, DIRECCION, LONGITUD, LATITUD"
)

if uploaded:
    try:
        df = pd.read_excel(uploaded)
        df = normalize_columns(df)

        st.subheader("Datos cargados")
        st.dataframe(df.head(10), use_container_width=True)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(
                "El archivo no contiene todas las columnas requeridas: "
                + ", ".join(missing)
            )
            st.stop()

        st.write(f"**Registros detectados:** {len(df)}")

        if st.button("🚀 Organizar visitas", type="primary"):
            with st.spinner("Analizando coordenadas y formando paquetes..."):
                try:
                    result, summary, control, swapped = organize_visits(
                        df,
                        min_size=int(min_size),
                        max_size=int(max_size),
                        seed_trials=int(seed_trials)
                    )

                    st.session_state.result = result
                    st.session_state.summary = summary
                    st.session_state.control = control
                    st.session_state.swapped = swapped
                    st.success("Organización completada.")
                except Exception as e:
                    st.error(str(e))

    except Exception as e:
        st.error(f"No se pudo leer el Excel: {e}")

if "result" in st.session_state:
    result = st.session_state.result
    summary = st.session_state.summary
    control = st.session_state.control

    if st.session_state.swapped:
        st.warning(
            "Se detectó que LONGITUD y LATITUD estaban invertidas en el archivo "
            "original. El aplicativo las corrigió automáticamente para el proceso."
        )

    st.subheader("Resumen de paquetes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Visitas", len(result))
    c2.metric("Paquetes", len(summary))
    c3.metric(
        "Visitas por paquete",
        f"{summary['VISITAS'].min()}–{summary['VISITAS'].max()}",
        delta=f"Objetivo {int(min_size)}–{int(max_size)}"
    )

    st.dataframe(summary, use_container_width=True)

    reduced = summary[summary["ESTADO"] == "REDUCIDO"]
    dispersed = summary[summary["ADVERTENCIA_DISTANCIA"] != ""]

    if len(reduced):
        paquetes = ", ".join(str(int(x)) for x in reduced["PAQUETE"])
        st.warning(
            f"Paquete(s) reducido(s): {paquetes}. "
            "Se mantienen así porque la distribución geográfica/cantidad "
            "no justifica forzar visitas lejanas dentro de otro paquete."
        )

    if len(dispersed):
        paquetes = ", ".join(str(int(x)) for x in dispersed["PAQUETE"])
        st.warning(
            f"Revisar desplazamiento en paquete(s): {paquetes}. "
            "La distancia máxima al centroide es elevada."
        )

    missing_rows = summary[summary["ADVERTENCIA_DISTANCIA"] == "SIN COORDENADAS"]
    if len(missing_rows):
        pnum = int(missing_rows.iloc[0]["PAQUETE"])
        nmissing = int(missing_rows.iloc[0]["VISITAS"])
        st.error(
            f"⚠️ Paquete {pnum}: {nmissing} visita(s) sin coordenadas. "
            "Revisar antes de programar el desplazamiento."
        )

    st.subheader("Visitas organizadas")
    st.dataframe(result, use_container_width=True)

    st.subheader("Control")
    st.dataframe(control, use_container_width=True)

    data = excel_bytes(result, summary, control)

    st.download_button(
        "⬇️ Descargar Excel organizado",
        data=data,
        file_name="visitas_organizadas_por_paquetes.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )
else:
    st.info(
        "Sube el Excel para comenzar. El aplicativo valida las coordenadas "
        "y también detecta automáticamente el caso de LONGITUD/LATITUD invertidas."
    )
