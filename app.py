import importlib.util
import os

import gradio as gr


def _cargar_modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


BASE = os.path.dirname(os.path.abspath(__file__))

california = _cargar_modulo(
    "housing_app", os.path.join(BASE, "california-housing", "housing_app.py")
)
wine = _cargar_modulo(
    "wine_app", os.path.join(BASE, "wine-quality", "wine_app.py")
)
diabetes = _cargar_modulo(
    "diabetes_app", os.path.join(BASE, "diabetes", "diabetes_app.py")
)

with gr.Blocks(title="Modelos Predictivos") as demo:
    gr.Markdown(
        "# 🧠 Modelos Predictivos\n"
        "Interfaz interactiva para probar la predicción de los **3 casos de estudio**: "
        "California Housing, Wine Quality y Diabetes."
    )
    with gr.Tab("🏠 California Housing"):
        california.crear_tab()
    with gr.Tab("🍷 Wine Quality"):
        wine.crear_tab()
    with gr.Tab("🩸 Diabetes"):
        diabetes.crear_tab()


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        theme=gr.themes.Soft(),
    )