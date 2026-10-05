# Organizador de Visitas

Aplicativo independiente para organizar visitas por proximidad geográfica.

## Columnas requeridas

- CUENTA
- SOLICITUD
- MUNICIPIO
- DIRECCION
- LONGITUD
- LATITUD

## Funciones

- Carga de Excel.
- Validación de coordenadas.
- Detección automática de LONGITUD/LATITUD invertidas.
- Agrupación geográfica con capacidad controlada.
- Objetivo de 12–15 visitas por paquete (configurable), sin bloquear cantidades que no permitan una distribución exacta.
- Exportación a Excel con VISITAS, RESUMEN_PAQUETES y CONTROL.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Render

1. Sube estos archivos a un repositorio de GitHub.
2. En Render crea un nuevo Web Service desde ese repositorio.
3. Build Command:
   `pip install -r requirements.txt`
4. Start Command:
   `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
5. En Environment Variables crea:
   `APP_PASSWORD = TU_CONTRASEÑA`
6. Publica.

Para pruebas se puede usar el plan Free. Para operación continua, conviene pasar a un servicio de pago.
