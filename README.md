# Organizador de Visitas — V15

Aplicativo Streamlit para organizar visitas por proximidad geográfica y cruzarlas con órdenes PDF.

## Cambios V15

- Cantidad de paquetes configurable por municipio, manteniendo la distribución geográfica.
- Carga de PDF en varias tandas desde diferentes carpetas.
- Cruce exacto por `SOLICITUD`.
- PDFs consolidados por paquete en un único archivo para impresión.
- Exportación de Excel de resultados y archivos para Google My Maps.

## Columnas requeridas

- CUENTA
- SOLICITUD
- MUNICIPIO
- DIRECCION
- LONGITUD
- LATITUD

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Render

Build Command:
`pip install -r requirements.txt`

Start Command:
`streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
