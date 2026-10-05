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

REQUIRED_COLUMNS = [
    "CUENTA", "SOLICITUD", "MUNICIPIO",
    "DIRECCION", "LONGITUD", "LATITUD"
]

# V6:
# - Primero separa por MUNICIPIO.
# - La compactación geográfica tiene prioridad sobre llenar 12–15.
# - 12–15 es objetivo, no obligación.
# - Un municipio nunca se mezcla con otro para completar paquetes.
# - Se aumenta el número de paquetes dentro de un municipio cuando
#   eso reduce la dispersión máxima.
TARGET_MAX_RADIUS_KM = 17.0
WARNING_RADIUS_KM = 17.0


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def detect_swapped_coordinates(df):
    lon = pd.to_numeric(df["LONGITUD"], errors="coerce").dropna()
    lat = pd.to_numeric(df["LATITUD"], errors="coerce").dropna()
    if len(lon) == 0 or len(lat) == 0:
        return False
    return abs(float(lon.median())) < 20 and abs(float(lat.median())) > 20


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


def balanced_geographical_clustering(coords, sizes, seed_trials=50):
    """
    Asignación geográfica con capacidades exactas.
    Se mantiene para casos donde el número de paquetes ya está decidido.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    k = len(sizes)

    if k == 1:
        return np.zeros(n, dtype=int), np.array([coords.mean(axis=0)])

    mean_lat = np.radians(np.mean(coords[:, 0]))
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
        km.fit(projected)
        centers = km.cluster_centers_.copy()

        for _ in range(60):
            old_centers = centers.copy()
            centers_proj = centers

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

            new_centers = []
            for p in range(k):
                pts = coords[labels == p]
                if len(pts):
                    new_centers.append(pts.mean(axis=0))
                else:
                    new_centers.append(centers[p])
            centers = np.asarray(new_centers)

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

        score = total_sq_km + 150.0 * (max_radius ** 2) + 5.0 * sum_radius

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "labels": labels.copy(),
                "centers": centers.copy()
            }

    return best["labels"], best["centers"]


def candidate_sizes(n, k, max_size):
    """
    Reparte n elementos en k paquetes sin superar max_size.
    No impone mínimo: la geografía puede justificar paquetes reducidos.
    """
    if k <= 0 or k > n or n > k * max_size:
        return None

    base = n // k
    rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]

    if max(sizes) > max_size:
        return None
    return sizes


def evaluate_clustering(coords, labels, centers, min_size, max_size):
    radii = []
    total_sq = 0.0
    size_penalty = 0.0

    for p in range(len(centers)):
        pts = coords[labels == p]
        if len(pts) == 0:
            continue
        d = haversine_km(centers[p], pts)
        r = float(np.max(d))
        radii.append(r)
        total_sq += float(np.sum(d ** 2))

        if len(pts) < min_size:
            # Penalización suave: permite reducidos, pero no los favorece.
            size_penalty += float((min_size - len(pts)) ** 2)

    max_radius = max(radii) if radii else 0.0
    sum_radius = sum(radii)

    return {
        "max_radius": max_radius,
        "sum_radius": sum_radius,
        "total_sq": total_sq,
        "size_penalty": size_penalty,
    }


def geographic_priority_clustering(
    coords,
    min_size=12,
    max_size=15,
    seed_trials=50,
    target_radius=TARGET_MAX_RADIUS_KM
):
    """
    V6:
    Busca el MENOR número de paquetes que consiga una compactación
    geográfica razonable. El tamaño 12–15 queda como preferencia.

    La selección se hace por:
      1. cumplir radio objetivo cuando sea posible;
      2. minimizar el radio máximo;
      3. minimizar dispersión total;
      4. evitar paquetes reducidos innecesarios;
      5. finalmente, usar menos paquetes.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)

    if n == 0:
        return np.array([], dtype=int), np.empty((0, 2)), 0

    if n <= max_size:
        return np.zeros(n, dtype=int), np.array([coords.mean(axis=0)]), 1

    k_start = int(np.ceil(n / max_size))

    # Probamos hasta n paquetes. En la práctica se detiene al encontrar
    # una solución dentro del radio objetivo.
    best_any = None
    best_target = None

    for k in range(k_start, n + 1):
        sizes = candidate_sizes(n, k, max_size)
        if sizes is None:
            continue

        labels, centers = balanced_geographical_clustering(
            coords, sizes, seed_trials=seed_trials
        )
        metrics = evaluate_clustering(
            coords, labels, centers, min_size, max_size
        )

        # Una vez que la compactación máxima es aceptable, preferimos
        # no seguir aumentando el número de paquetes.
        if metrics["max_radius"] <= target_radius:
            score = (
                metrics["max_radius"],
                metrics["sum_radius"],
                metrics["size_penalty"],
                k
            )
            if best_target is None or score < best_target["score"]:
                best_target = {
                    "score": score,
                    "labels": labels.copy(),
                    "centers": centers.copy(),
                    "k": k,
                    "metrics": metrics
                }

            # Ya encontramos el menor k que entra en el objetivo.
            # No necesitamos fragmentar más el municipio.
            break

        # Mejor solución de respaldo: primero radio máximo, luego
        # dispersión, penalización de reducidos y número de paquetes.
        score_any = (
            metrics["max_radius"],
            metrics["sum_radius"],
            metrics["size_penalty"],
            k
        )
        if best_any is None or score_any < best_any["score"]:
            best_any = {
                "score": score_any,
                "labels": labels.copy(),
                "centers": centers.copy(),
                "k": k,
                "metrics": metrics
            }

    chosen = best_target if best_target is not None else best_any

    if chosen is None:
        raise ValueError("No fue posible formar paquetes geográficos.")

    return chosen["labels"], chosen["centers"], chosen["k"]


def organize_visits(
    df,
    min_size=12,
    max_size=15,
    seed_trials=50,
    target_radius=TARGET_MAX_RADIUS_KM
):
    df = df.copy()

    for col in ["LONGITUD", "LATITUD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    swapped = detect_swapped_coordinates(df)

    if swapped:
        original_lon = df["LONGITUD"].copy()
        df["LONGITUD"] = df["LATITUD"]
        df["LATITUD"] = original_lon

    valid = (
        df["LATITUD"].between(-90, 90)
        & df["LONGITUD"].between(-180, 180)
        & df["LATITUD"].notna()
        & df["LONGITUD"].notna()
        & (df["LATITUD"] != 0)
        & (df["LONGITUD"] != 0)
    )

    missing_coords = df.loc[~valid].copy()
    geo_df = df.loc[valid].copy()
    invalid_count = int((~valid).sum())

    if len(geo_df) == 0:
        result = df.copy()
        result["PAQUETE"] = 1
        result["ORDEN"] = range(1, len(result) + 1)
        result["OBSERVACION"] = "REVISAR: SIN COORDENADAS"

        summary_df = pd.DataFrame([{
            "PAQUETE": 1,
            "VISITAS": len(result),
            "MUNICIPIO": " | ".join(
                sorted(result["MUNICIPIO"].fillna("").astype(str).unique())
            ),
            "DISTANCIA_PROMEDIO_KM": 0,
            "DISTANCIA_MAXIMA_KM": 0,
            "LATITUD_CENTRO": "",
            "LONGITUD_CENTRO": "",
            "ESTADO": "REVISAR",
            "ADVERTENCIA_DISTANCIA": "SIN COORDENADAS"
        }])

        control = pd.DataFrame({
            "CONTROL": [
                "TOTAL VISITAS", "VISITAS CON GPS", "VISITAS SIN GPS",
                "PAQUETES GEOGRÁFICOS", "PAQUETE ESPECIAL SIN COORDENADAS",
                "COORDENADAS INVERTIDAS DETECTADAS"
            ],
            "VALOR": [
                len(df), 0, invalid_count, 0, 1,
                "SÍ" if swapped else "NO"
            ]
        })
        return result, summary_df, control, swapped

    geo_df["_MUNICIPIO_KEY"] = (
        geo_df["MUNICIPIO"]
        .fillna("SIN MUNICIPIO")
        .astype(str)
        .str.strip()
        .replace("", "SIN MUNICIPIO")
    )

    # ==========================================================
    # V6 — AGRUPAMIENTO POR MUNICIPIO
    # ==========================================================
    next_package = 1
    package_centers = {}
    summary_rows = []
    assigned_frames = []

    for municipio in sorted(geo_df["_MUNICIPIO_KEY"].unique()):
        sub = geo_df[geo_df["_MUNICIPIO_KEY"] == municipio].copy()
        coords = sub[["LATITUD", "LONGITUD"]].to_numpy(dtype=float)

        labels, centers, k = geographic_priority_clustering(
            coords,
            min_size=min_size,
            max_size=max_size,
            seed_trials=seed_trials,
            target_radius=target_radius
        )

        # Ordenamos los subpaquetes por centro geográfico.
        order = sorted(
            range(k),
            key=lambda p: (centers[p, 0], centers[p, 1])
        )
        local_map = {old: i + 1 for i, old in enumerate(order)}

        for old_p in range(k):
            package_id = next_package + local_map[old_p] - 1
            mask = labels == old_p
            pts = coords[mask]
            center = pts.mean(axis=0)

            distances = haversine_km(center, pts)
            avg_d = float(np.mean(distances)) if len(distances) else 0.0
            max_d = float(np.max(distances)) if len(distances) else 0.0

            status = (
                "NORMAL" if min_size <= len(pts) <= max_size
                else "REDUCIDO" if len(pts) < min_size
                else "GRANDE"
            )

            warning = (
                "REVISAR DESPLAZAMIENTO"
                if max_d > WARNING_RADIUS_KM
                else ""
            )

            local = sub.loc[mask].copy()
            local["PAQUETE"] = package_id
            local["_DISTANCIA_CENTRO_KM"] = distances
            local["OBSERVACION"] = ""
            assigned_frames.append(local)

            package_centers[package_id] = center

            summary_rows.append({
                "PAQUETE": package_id,
                "MUNICIPIO": municipio,
                "VISITAS": len(pts),
                "DISTANCIA_PROMEDIO_KM": round(avg_d, 3),
                "DISTANCIA_MAXIMA_KM": round(max_d, 3),
                "LATITUD_CENTRO": round(float(center[0]), 6),
                "LONGITUD_CENTRO": round(float(center[1]), 6),
                "ESTADO": status,
                "ADVERTENCIA_DISTANCIA": warning
            })

        next_package += k

    result = pd.concat(assigned_frames, ignore_index=True)

    # Orden operativo: paquete y luego cercanía al centro.
    result = result.sort_values(
        ["PAQUETE", "_DISTANCIA_CENTRO_KM", "SOLICITUD"],
        kind="stable"
    ).reset_index(drop=True)

    result["ORDEN"] = result.groupby("PAQUETE").cumcount() + 1

    # Paquetes sin GPS al final, juntos y claramente marcados.
    special_package = None
    if invalid_count:
        special_package = next_package
        missing_coords["PAQUETE"] = special_package
        missing_coords["ORDEN"] = range(1, len(missing_coords) + 1)
        missing_coords["OBSERVACION"] = "REVISAR: SIN COORDENADAS"
        missing_coords["_DISTANCIA_CENTRO_KM"] = 0

        summary_rows.append({
            "PAQUETE": special_package,
            "MUNICIPIO": " | ".join(
                sorted(
                    missing_coords["_MUNICIPIO_KEY"]
                    if "_MUNICIPIO_KEY" in missing_coords.columns
                    else missing_coords["MUNICIPIO"].fillna("").astype(str)
                )
            ),
            "VISITAS": len(missing_coords),
            "DISTANCIA_PROMEDIO_KM": 0,
            "DISTANCIA_MAXIMA_KM": 0,
            "LATITUD_CENTRO": "",
            "LONGITUD_CENTRO": "",
            "ESTADO": "REVISAR",
            "ADVERTENCIA_DISTANCIA": "SIN COORDENADAS"
        })

        result = pd.concat([result, missing_coords], ignore_index=True)

    # Eliminar columnas internas.
    for col in ["_DISTANCIA_CENTRO_KM", "_MUNICIPIO_KEY"]:
        if col in result.columns:
            result = result.drop(columns=[col])

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "PAQUETE"
    ).reset_index(drop=True)

    control = pd.DataFrame({
        "CONTROL": [
            "TOTAL VISITAS",
            "VISITAS CON GPS",
            "VISITAS SIN GPS",
            "PAQUETES GEOGRÁFICOS",
            "PAQUETE ESPECIAL SIN COORDENADAS",
            "TAMAÑO MÍNIMO OBJETIVO",
            "TAMAÑO MÁXIMO OBJETIVO",
            "RADIO OBJETIVO GEOGRÁFICO (KM)",
            "RADIO DE ADVERTENCIA (KM)",
            "COORDENADAS INVERTIDAS DETECTADAS"
        ],
        "VALOR": [
            len(df),
            len(geo_df),
            invalid_count,
            len(summary_rows) - (1 if invalid_count else 0),
            special_package if special_package else "NO",
            min_size,
            max_size,
            target_radius,
            WARNING_RADIUS_KM,
            "SÍ" if swapped else "NO"
        ]
    })

    return result, summary_df, control, swapped



def extract_solicitud_from_pdf(pdf_bytes):
    """
    Extrae el número de SOLICITUD de una orden PDF.
    La llave oficial del cruce es SOLICITUD.
    """
    text = ""

    # pypdf es suficiente para las órdenes de texto; si no está disponible,
    # el usuario recibe un mensaje claro en la interfaz.
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pass
        text = "\n".join(pages)
    except Exception as e:
        raise RuntimeError(
            "No se pudo leer el PDF. Verifica que sea un PDF con texto."
        ) from e

    # Primero buscamos expresamente "Número solicitud".
    patterns = [
        r"N[uú]mero\s+solicitud\s*[:\-]?\s*([0-9]{4,})",
        r"N[uú]mero\s+de\s+solicitud\s*[:\-]?\s*([0-9]{4,})",
        r"SOLICITUD\s*[:\-]?\s*([0-9]{4,})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip(), text

    # Como respaldo, intentar localizar el número en el nombre del archivo
    # se hace fuera de esta función; aquí no se inventa ninguna solicitud.
    return None, text


def normalize_solicitud(value):
    """
    Normaliza la llave SOLICITUD para que Excel/PDF coincidan aunque
    Excel la haya leído como número decimal.
    """
    if pd.isna(value):
        return ""

    s = str(value).strip()

    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]

    digits = re.sub(r"\D", "", s)
    return digits if digits else s


def organize_pdfs(uploaded_pdfs, result):
    """
    Cruza PDFs contra el Excel usando EXCLUSIVAMENTE SOLICITUD.
    Devuelve:
      - dataframe de control
      - lista de archivos clasificados
      - lista de PDFs no encontrados
    """
    lookup = {}

    for _, row in result.iterrows():
        key = normalize_solicitud(row.get("SOLICITUD"))
        if key:
            lookup.setdefault(key, []).append(row)

    records = []
    classified = []
    unmatched = []
    duplicates = []

    for pdf_file in uploaded_pdfs:
        pdf_bytes = pdf_file.getvalue()
        filename = pdf_file.name

        try:
            solicitud, _ = extract_solicitud_from_pdf(pdf_bytes)
        except Exception as e:
            records.append({
                "PDF": filename,
                "SOLICITUD": "",
                "PAQUETE": "",
                "MUNICIPIO": "",
                "ESTADO": "ERROR LECTURA PDF",
                "DETALLE": str(e)
            })
            unmatched.append((filename, pdf_bytes))
            continue

        if not solicitud:
            records.append({
                "PDF": filename,
                "SOLICITUD": "",
                "PAQUETE": "",
                "MUNICIPIO": "",
                "ESTADO": "SOLICITUD NO ENCONTRADA",
                "DETALLE": "No se encontró Número solicitud dentro del PDF."
            })
            unmatched.append((filename, pdf_bytes))
            continue

        rows = lookup.get(normalize_solicitud(solicitud), [])

        if not rows:
            records.append({
                "PDF": filename,
                "SOLICITUD": solicitud,
                "PAQUETE": "",
                "MUNICIPIO": "",
                "ESTADO": "PDF_NO_ENCONTRADO",
                "DETALLE": "La SOLICITUD no existe en el Excel."
            })
            unmatched.append((filename, pdf_bytes))
            continue

        # Una solicitud debería ser única. Si el Excel tiene duplicados,
        # no elegimos arbitrariamente.
        if len(rows) > 1:
            records.append({
                "PDF": filename,
                "SOLICITUD": solicitud,
                "PAQUETE": "",
                "MUNICIPIO": "",
                "ESTADO": "SOLICITUD DUPLICADA EN EXCEL",
                "DETALLE": f"La llave aparece {len(rows)} veces en el Excel."
            })
            unmatched.append((filename, pdf_bytes))
            continue

        row = rows[0]
        paquete = int(row["PAQUETE"])
        municipio = str(row.get("MUNICIPIO", ""))

        if any(
            r["SOLICITUD"] == solicitud and r["ESTADO"] == "ENCONTRADO"
            for r in records
        ):
            estado = "DUPLICADO PDF"
            duplicates.append((filename, pdf_bytes, paquete, municipio, solicitud))
        else:
            estado = "ENCONTRADO"

        records.append({
            "PDF": filename,
            "SOLICITUD": solicitud,
            "PAQUETE": paquete,
            "MUNICIPIO": municipio,
            "ESTADO": estado,
            "DETALLE": ""
        })

        classified.append(
            (filename, pdf_bytes, paquete, municipio, solicitud)
        )

    # También reportamos visitas del Excel que no tienen PDF.
    pdf_keys = {
        normalize_solicitud(r["SOLICITUD"])
        for r in records
        if r["SOLICITUD"]
    }

    for _, row in result.iterrows():
        key = normalize_solicitud(row.get("SOLICITUD"))
        if not key:
            continue
        if key not in pdf_keys:
            records.append({
                "PDF": "",
                "SOLICITUD": key,
                "PAQUETE": int(row["PAQUETE"]),
                "MUNICIPIO": str(row.get("MUNICIPIO", "")),
                "ESTADO": "SIN PDF",
                "DETALLE": "La visita existe en el Excel pero no se recibió PDF."
            })

    control = pd.DataFrame(records)

    return control, classified, unmatched, duplicates


def build_pdf_zip(classified, unmatched):
    """
    Crea un ZIP:
      PAQUETE_XX_MUNICIPIO/
        PDF...
      PDF_NO_ENCONTRADO/
        PDF...
    """
    output = io.BytesIO()

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        used_names = set()

        for filename, pdf_bytes, paquete, municipio, solicitud in classified:
            safe_municipio = re.sub(
                r"[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñ _-]",
                "_",
                str(municipio)
            ).strip() or "SIN_MUNICIPIO"

            folder = f"PAQUETE_{paquete:02d}_{safe_municipio}"

            # Evitar sobrescritura si llegan dos PDFs con el mismo nombre.
            candidate = filename
            stem = Path(filename).stem
            suffix = Path(filename).suffix or ".pdf"
            counter = 2

            while f"{folder}/{candidate}" in used_names:
                candidate = f"{stem}_{counter}{suffix}"
                counter += 1

            path = f"{folder}/{candidate}"
            used_names.add(path)
            z.writestr(path, pdf_bytes)

        for filename, pdf_bytes in unmatched:
            candidate = filename
            stem = Path(filename).stem
            suffix = Path(filename).suffix or ".pdf"
            counter = 2

            while f"PDF_NO_ENCONTRADO/{candidate}" in used_names:
                candidate = f"{stem}_{counter}{suffix}"
                counter += 1

            path = f"PDF_NO_ENCONTRADO/{candidate}"
            used_names.add(path)
            z.writestr(path, pdf_bytes)

    return output.getvalue()



def excel_bytes(result, summary, control):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="VISITAS", index=False)
        summary.to_excel(writer, sheet_name="RESUMEN_PAQUETES", index=False)
        control.to_excel(writer, sheet_name="CONTROL", index=False)

        # Hoja especial para los registros que no tienen GPS.
        sin_gps = result[
            result["OBSERVACION"].astype(str).str.contains(
                "SIN COORDENADAS", na=False
            )
        ].copy()
        if len(sin_gps):
            sin_gps.to_excel(writer, sheet_name="SIN_GPS", index=False)

    return output.getvalue()


# ============================================================
# INTERFAZ
# ============================================================

st.title("📍 Organizador de Visitas — V6")

st.write(
    "Carga un Excel y el aplicativo organizará las visitas por proximidad "
    "geográfica. Primero separa por municipio y después busca grupos "
    "compactos. El objetivo es 12–15 visitas, pero la distancia tiene "
    "prioridad: si un municipio está disperso, se permiten paquetes "
    "reducidos antes que enviar una cuadrilla a recorrer grandes distancias."
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

    target_radius = st.number_input(
        "Radio geográfico objetivo (km)",
        min_value=1.0,
        max_value=100.0,
        value=float(TARGET_MAX_RADIUS_KM),
        step=1.0,
        help="El algoritmo intenta mantener cada paquete dentro de este radio desde su centro."
    )

    seed_trials = st.slider(
        "Calidad de búsqueda geográfica",
        min_value=10,
        max_value=80,
        value=50,
        step=10,
        help="Más pruebas pueden encontrar agrupaciones más compactas, pero tardan más."
    )

    st.caption("V6: municipio primero + distancia primero.")

uploaded = st.file_uploader(
    "Sube el archivo Excel",
    type=["xlsx", "xls"],
    help="Columnas requeridas: CUENTA, SOLICITUD, MUNICIPIO, DIRECCION, LONGITUD, LATITUD"
)

pdfs_uploaded = st.file_uploader(
    "📄 Sube los PDF de las órdenes",
    type=["pdf"],
    accept_multiple_files=True,
    help="El cruce se hace exclusivamente por SOLICITUD."
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
            with st.spinner("Analizando coordenadas y formando paquetes geográficos..."):
                try:
                    result, summary, control, swapped = organize_visits(
                        df,
                        min_size=int(min_size),
                        max_size=int(max_size),
                        seed_trials=int(seed_trials),
                        target_radius=float(target_radius)
                    )

                    st.session_state.result = result
                    st.session_state.summary = summary
                    st.session_state.control = control
                    st.session_state.swapped = swapped

                    st.success("Organización V6 completada.")

                except Exception as e:
                    st.error(f"No se pudo organizar el archivo: {e}")

    except Exception as e:
        st.error(f"No se pudo leer el Excel: {e}")


if "result" in st.session_state:
    result = st.session_state.result
    summary = st.session_state.summary
    control = st.session_state.control

    # ========================================================
    # MÓDULO PDF — llave SOLICITUD
    # ========================================================
    if pdfs_uploaded:
        st.subheader("📄 Cruce de órdenes PDF")

        if st.button("🔗 Cruzar PDF con las visitas"):
            with st.spinner("Leyendo SOLICITUD de cada PDF y asignándolo a su paquete..."):
                try:
                    pdf_control, classified, unmatched, duplicates = organize_pdfs(
                        pdfs_uploaded, result
                    )

                    st.session_state.pdf_control = pdf_control
                    st.session_state.pdf_classified = classified
                    st.session_state.pdf_unmatched = unmatched
                    st.session_state.pdf_duplicates = duplicates

                    st.success(
                        f"Proceso terminado: {len(pdfs_uploaded)} PDF analizados."
                    )
                except Exception as e:
                    st.error(f"No se pudieron procesar los PDF: {e}")

    if "pdf_control" in st.session_state:
        pdf_control = st.session_state.pdf_control
        classified = st.session_state.pdf_classified
        unmatched = st.session_state.pdf_unmatched

        st.dataframe(pdf_control, use_container_width=True)

        found = int(
            (pdf_control["ESTADO"] == "ENCONTRADO").sum()
        )
        no_pdf = int(
            (pdf_control["ESTADO"] == "SIN PDF").sum()
        )
        not_found = int(
            (pdf_control["ESTADO"] == "PDF_NO_ENCONTRADO").sum()
        )
        duplicate_pdf = int(
            (pdf_control["ESTADO"] == "DUPLICADO PDF").sum()
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("PDF encontrados", found)
        p2.metric("Visitas sin PDF", no_pdf)
        p3.metric("PDF sin solicitud en Excel", not_found)
        p4.metric("PDF duplicados", duplicate_pdf)

        if classified:
            pdf_zip = build_pdf_zip(classified, unmatched)

            st.download_button(
                "📦 Descargar ZIP con PDF organizados",
                data=pdf_zip,
                file_name="PDF_organizados_por_paquete.zip",
                mime="application/zip"
            )

        pdf_control_bytes = io.BytesIO()
        with pd.ExcelWriter(pdf_control_bytes, engine="openpyxl") as writer:
            pdf_control.to_excel(
                writer,
                sheet_name="CONTROL_PDF",
                index=False
            )

        st.download_button(
            "📊 Descargar control de PDF",
            data=pdf_control_bytes.getvalue(),
            file_name="control_PDF.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.session_state.swapped:
        st.warning(
            "Se detectó que LONGITUD y LATITUD estaban invertidas en el archivo "
            "original. El aplicativo las corrigió automáticamente."
        )

    st.subheader("Resumen de paquetes")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Visitas", len(result))
    c2.metric("Paquetes", len(summary))
    c3.metric(
        "Visitas por paquete",
        f"{summary['VISITAS'].min()}–{summary['VISITAS'].max()}"
    )

    normal_count = int((summary["ESTADO"] == "NORMAL").sum())
    c4.metric("Paquetes 12–15", normal_count)

    st.dataframe(summary, use_container_width=True)

    st.subheader("Visitas organizadas")
    st.dataframe(result, use_container_width=True)

    data = excel_bytes(result, summary, control)

    st.download_button(
        "⬇️ Descargar Excel organizado",
        data=data,
        file_name="visitas_organizadas_V6.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("Control")
    st.dataframe(control, use_container_width=True)
