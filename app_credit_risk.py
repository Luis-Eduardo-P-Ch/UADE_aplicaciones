"""
Simulador de Riesgo Crediticio — Streamlit App
===============================================
Interfaz para evaluar el riesgo crediticio de un solicitante
usando un modelo de regresión logística entrenado con datos simulados.

Autor: Curso Python
Versión: 1.0
"""

import streamlit as st
import plotly.graph_objects as go
from credit_model import CreditRiskModel

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Riesgo Crediticio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d0d0d;
        color: #e8e8e8;
    }
    h1 { font-family: 'IBM Plex Mono', monospace; color: #00e5a0; letter-spacing: -1px; }
    h2, h3 { color: #c8c8c8; font-weight: 500; }

    .stButton>button {
        background-color: #00e5a0;
        color: #0d0d0d;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        width: 100%;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: background 0.2s;
    }
    .stButton>button:hover { background-color: #00c88a; }

    .result-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin: 0.5rem 0;
    }
    .score-big {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 4rem;
        font-weight: 600;
        line-height: 1;
    }
    .label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 2px;
        color: #666;
        text-transform: uppercase;
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .sidebar-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 1.5px;
        color: #555;
        text-transform: uppercase;
        margin-bottom: -10px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODELO (se carga una sola vez en session_state)
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    with st.spinner("Entrenando modelo..."):
        st.session_state.model = CreditRiskModel()

model = st.session_state.model

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS DEL SOLICITANTE
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Datos del Solicitante")
    st.markdown("---")

    st.markdown('<p class="sidebar-label">Edad</p>', unsafe_allow_html=True)
    edad = st.slider("", 18, 75, 35, key="edad", label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Ingresos anuales ($)</p>', unsafe_allow_html=True)
    ingresos = st.number_input("", min_value=5000, max_value=500000,
                                value=60000, step=5000, key="ingresos",
                                label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Deudas actuales ($)</p>', unsafe_allow_html=True)
    deudas = st.number_input("", min_value=0, max_value=300000,
                              value=15000, step=1000, key="deudas",
                              label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Antigüedad laboral (años)</p>', unsafe_allow_html=True)
    antiguedad = st.slider("", 0, 40, 5, key="antiguedad", label_visibility="collapsed")

    st.markdown("---")
    evaluar = st.button("▶  EVALUAR CRÉDITO")

# ─────────────────────────────────────────────
# MAIN — ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("# Simulador de Riesgo Crediticio")
st.markdown("Modelo de regresión logística entrenado con datos simulados · Curso Python")
st.markdown("---")

# ─────────────────────────────────────────────
# RESULTADO
# ─────────────────────────────────────────────
if evaluar or "resultado" in st.session_state:

    if evaluar:
        st.session_state.resultado = model.predict(edad, ingresos, deudas, antiguedad)

    r = st.session_state.resultado

    # Colores por categoría
    color_map = {
        "green":      ("#00e5a0", "#0a2e22"),
        "lightgreen": ("#a8e06a", "#1a2e0a"),
        "orange":     ("#f0a500", "#2e200a"),
        "red":        ("#e05050", "#2e0a0a"),
    }
    accent, bg = color_map[r["color"]]

    # ── Fila de métricas ──────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="result-card" style="border-color:{accent}30;">
            <p class="label">Credit Score</p>
            <p class="score-big" style="color:{accent};">{r['score']}</p>
            <p style="color:#666; font-size:0.8rem; margin-top:0.3rem;">rango 300 – 850</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <p class="label">Prob. de Default</p>
            <p class="score-big" style="color:{accent};">{r['prob_default']}%</p>
            <p style="color:#666; font-size:0.8rem; margin-top:0.3rem;">estimada por el modelo</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="result-card">
            <p class="label">Ratio Deuda / Ingreso</p>
            <p class="score-big" style="color:{accent};">{r['ratio_deuda']}%</p>
            <p style="color:#666; font-size:0.8rem; margin-top:0.3rem;">recomendado: &lt; 35%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Categoría y recomendación ──────────────────────────
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(f"""
        <div class="result-card" style="background:{bg}; border-color:{accent}50; text-align:center;">
            <p class="label">Categoría</p>
            <p style="font-size:1.8rem; font-weight:700; color:{accent}; margin:0.5rem 0;">{r['categoria']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="result-card" style="border-color:{accent}30;">
            <p class="label">Recomendación del modelo</p>
            <p style="font-size:1.1rem; color:#e8e8e8; margin-top:0.5rem;">
                {r['recomendacion']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge chart ───────────────────────────────────────
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r["score"],
        number={"font": {"size": 48, "color": accent, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [300, 850], "tickfont": {"color": "#666", "size": 11}},
            "bar":  {"color": accent, "thickness": 0.25},
            "bgcolor": "#1a1a1a",
            "bordercolor": "#2a2a2a",
            "steps": [
                {"range": [300, 580], "color": "#2e0a0a"},
                {"range": [580, 670], "color": "#2e200a"},
                {"range": [670, 750], "color": "#1a2e0a"},
                {"range": [750, 850], "color": "#0a2e22"},
            ],
            "threshold": {
                "line": {"color": accent, "width": 3},
                "thickness": 0.8,
                "value": r["score"],
            },
        },
        title={"text": "Credit Score", "font": {"color": "#666", "size": 14,
                                                  "family": "IBM Plex Mono"}},
    ))

    fig.update_layout(
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font_color="#e8e8e8",
        height=320,
        margin=dict(t=60, b=20, l=40, r=40),
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    # ── Pantalla de bienvenida ─────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#444;">
        <p style="font-size:5rem; margin:0;">🏦</p>
        <p style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:#555; margin-top:1rem;">
            Ingresá los datos del solicitante en el panel izquierdo<br>y hacé clic en <strong style="color:#00e5a0;">EVALUAR CRÉDITO</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#333; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;">
    Simulador de Riesgo Crediticio · Curso Python · Modelo con fines educativos
</p>
""", unsafe_allow_html=True)
