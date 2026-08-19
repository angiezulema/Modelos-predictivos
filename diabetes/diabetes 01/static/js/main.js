/* ============================================================
   main.js - Lógica de predicción en vivo (AJAX) de la app
   ============================================================ */

(function () {
    "use strict";

    const btnPredict = document.getElementById("btn-predict");
    const resultBox = document.getElementById("prediction-result");
    const interpretacionBox = document.getElementById("interpretacion");
    const sliders = document.querySelectorAll(".slider");

    let debounceTimer = null;

    /**
     * Recoge los valores de todos los deslizadores.
     * @returns {Object} Mapa variable -> valor
     */
    function getValues() {
        const values = {};
        sliders.forEach(function (slider) {
            values[slider.id.replace("input-", "")] = parseFloat(slider.value);
        });
        return values;
    }

    /**
     * Actualiza el badge del valor, la tabla y dispara la predicción.
     */
    function updateUI() {
        sliders.forEach(function (slider) {
            const col = slider.id.replace("input-", "");
            const valueEl = document.getElementById("value-" + col);
            const tableEl = document.getElementById("table-" + col);
            if (valueEl) valueEl.textContent = slider.value;
            if (tableEl) tableEl.textContent = slider.value;
        });
        schedulePrediction();
    }

    /**
     * Programación de la predicción en vivo con debounce (250 ms).
     */
    function schedulePrediction() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(triggerPrediction, 250);
    }

    /**
     * Envía los valores al servidor y muestra el resultado.
     */
    function triggerPrediction() {
        const values = getValues();

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(values),
        })
            .then(function (response) {
                return response.json().then(function (data) {
                    return { ok: response.ok, data: data };
                });
            })
            .then(function (result) {
                if (!result.ok) {
                    throw new Error(
                        (result.data.errors || ["Error desconocido"]).join(", ")
                    );
                }
                renderResult(result.data);
            })
            .catch(function (err) {
                resultBox.innerHTML =
                    '<div class="prediction-placeholder">Error al predecir: ' +
                    err.message +
                    "</div>";
                interpretacionBox.innerHTML = "";
            });
    }

    /**
     * Renderiza la tarjeta de resultado y la interpretación.
     */
    function renderResult(data) {
        const interp = data.interpretacion;

        resultBox.innerHTML =
            '<div class="prediction-card">' +
            '   <div>' +
            '       <div class="prediction-label">Progresión de la enfermedad predicha (1 año)</div>' +
            '       <div class="prediction-number">' + data.prediccion + "</div>" +
            '       <div class="prediction-caption">Índice de progresión cuantitativa (Y)</div>' +
            "   </div>" +
            "</div>";

        interpretacionBox.innerHTML =
            '<span class="badge badge-' + interp.clase + '">' + interp.nivel + "</span>" +
            '<div class="interpretacion-desc">' + interp.descripcion + "</div>";
    }

    /* ---------------- Eventos ---------------- */
    btnPredict.addEventListener("click", triggerPrediction);
    sliders.forEach(function (slider) {
        slider.addEventListener("input", updateUI);
    });

    // Predicción inicial al cargar la página
    updateUI();
})();