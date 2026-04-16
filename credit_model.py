"""
Credit Risk Model
=================
Modelo de riesgo crediticio usando regresión logística
entrenado sobre datos simulados.

Autor: Curso Python
Versión: 1.0
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CreditRiskModel:
    """
    Modelo de scoring crediticio basado en regresión logística.
    Se entrena automáticamente con datos simulados al instanciarse.
    """

    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self._train()

    def _simulate_data(self, n=1000):
        """Genera datos simulados de solicitantes de crédito"""
        np.random.seed(42)

        # Variables del solicitante
        edad        = np.random.randint(22, 70, n)
        ingresos    = np.random.normal(50000, 20000, n).clip(5000, 200000)
        deudas      = np.random.normal(15000, 10000, n).clip(0, 100000)
        antiguedad  = np.random.randint(0, 30, n)   # años laborales

        # Ratio deuda/ingreso
        ratio = deudas / ingresos

        # Probabilidad real de default (función logística con ruido)
        log_odds = (
            -3.0
            + 2.5  * ratio
            - 0.03 * (edad - 22)
            - 0.8  * (ingresos / 50000)
            + 0.5  * (deudas  / 15000)
            - 0.05 * antiguedad
        )
        prob_default = 1 / (1 + np.exp(-log_odds))
        y = (np.random.rand(n) < prob_default).astype(int)

        X = np.column_stack([edad, ingresos, deudas, antiguedad, ratio])
        return X, y

    def _train(self):
        """Entrena el modelo con datos simulados"""
        X, y = self._simulate_data()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, edad, ingresos, deudas, antiguedad):
        """
        Predice la probabilidad de default y devuelve el score.

        Parameters
        ----------
        edad        : int   — edad del solicitante
        ingresos    : float — ingresos anuales
        deudas      : float — deudas totales actuales
        antiguedad  : int   — años en el empleo actual

        Returns
        -------
        dict con prob_default, score (0-1000), categoria y recomendacion
        """
        ratio = deudas / max(ingresos, 1)
        X = np.array([[edad, ingresos, deudas, antiguedad, ratio]])
        X_scaled = self.scaler.transform(X)

        prob_default = self.model.predict_proba(X_scaled)[0][1]

        # Score estilo FICO: mayor score = menor riesgo
        score = int((1 - prob_default) * 1000)
        score = max(300, min(score, 850))   # rango típico 300-850

        # Categoría
        if score >= 750:
            categoria     = "Excelente"
            recomendacion = "Aprobar — riesgo muy bajo"
            color         = "green"
        elif score >= 670:
            categoria     = "Bueno"
            recomendacion = "Aprobar con condiciones estándar"
            color         = "lightgreen"
        elif score >= 580:
            categoria     = "Regular"
            recomendacion = "Revisar — considerar garantías adicionales"
            color         = "orange"
        else:
            categoria     = "Alto Riesgo"
            recomendacion = "Rechazar o solicitar co-deudor"
            color         = "red"

        return {
            "prob_default" : round(prob_default * 100, 1),
            "score"        : score,
            "categoria"    : categoria,
            "recomendacion": recomendacion,
            "color"        : color,
            "ratio_deuda"  : round(ratio * 100, 1),
        }
