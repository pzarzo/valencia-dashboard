
# Valencia Dashboard (Streamlit MVP)

Dashboard interactivo para analizar los partidos del Valencia CF.

## Estructura
```
valencia-dashboard/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   └── valencia_partidos_enriquecido.csv
└── img/
```
> **Importante:** deja tu CSV final en `data/valencia_partidos_enriquecido.csv`.

## Ejecutar en local
```bash
# 1) Clona el repo
git clone https://github.com/<TU-USUARIO>/valencia-dashboard.git
cd valencia-dashboard

# 2) Crea entorno e instala deps
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 3) Ejecuta la app
streamlit run app.py
```

## Despliegue en Streamlit Cloud (desde GitHub)
1. Sube este proyecto a GitHub.
2. Ve a https://share.streamlit.io (o https://streamlit.io/cloud).
3. Conecta tu cuenta de GitHub y elige el repo `valencia-dashboard`.
4. **Main file path:** `app.py`
5. **Python version:** 3.10+ (automático)
6. **Requirements:** `requirements.txt`
7. Deploy. Listo.

## Despliegue en GitHub Codespaces
1. En tu repo, botón **Code → Codespaces → Create codespace**.
2. En la terminal del Codespace:
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
3. Abre el puerto que te sugiera Codespaces.

## Notas
- Si cambias el nombre o ruta del CSV, actualiza `DATA_PATH` en `app.py`.
- La app infiere V/E/D por puntos si falta `resultado_valencia`.
