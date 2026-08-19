"""
Aplicación principal Streamlit — Proyecto Multi-Caso de IA.

Permite navegar entre los tres casos de estudio a través del menú lateral:

    - Caso 1: (placeholder genérico)
    - Caso 2: (placeholder genérico)
    - Caso 3: Predicción de Diabetes (implementado en components/case3_ui.py)

Ejecución:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Proyecto Multi-Caso IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_placeholder(case_number: int, title: str) -> None:
    """Renderiza una página placeholder para los casos no implementados."""
    st.info(
        f"El **{title}** se encuentra en construcción. "
        f"Regresa pronto para ver su implementación completa."
    )


def _render_cases() -> None:
    """Navegación principal entre los tres casos disponibles."""
    st.sidebar.title("🤖 Proyecto Multi-Caso")
    st.sidebar.markdown("Selecciona un caso de estudio:")

    cases = {
        "Caso 1": "🔢 Caso 1",
        "Caso 2": "💧 Caso 2",
        "Caso 3 — Diabetes": "🩺 Caso 3 — Diabetes",
    }

    selection = st.sidebar.radio("Navegación", list(cases.values()), label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.caption("Tecnología: Streamlit · Scikit-learn · Pandas")

    if selection == cases["Caso 1"]:
        st.title("🔢 Caso 1")
        _render_placeholder(1, "Caso 1")
    elif selection == cases["Caso 2"]:
        st.title("💧 Caso 2")
        _render_placeholder(2, "Caso 2")
    else:
        from app.components.case3_ui import render_case3_ui
        render_case3_ui()


def main() -> None:
    """Punto de entrada principal de la aplicación."""
    _render_cases()


if __name__ == "__main__":
    main()