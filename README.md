# 🏨 Hotel Booking ADR Prediction & MLOps Pipeline

> Sistema completo de Machine Learning para predecir la tarifa diaria promedio (**ADR - Average Daily Rate**) de reservas de hotel, utilizando **FastAPI**, **Docker**, **GitHub Actions** y un panel web interactivo.

---

## 🚀 Características del Proyecto
- **Modelo Predictivo:** Árbol de Regresión (`DecisionTreeRegressor`) entrenado con Scikit-Learn.
- **API Web RESTful:** Desarrollada con **FastAPI** para realizar inferencias en tiempo real.
- **Contenedor Docker:** Empaquetado optimizado con Python 3.10 para despliegues estables en la nube (**Render**).
- **CI/CD Automatizado:** Workflow en GitHub Actions (`train.yml`) con soporte para reentrenamiento remoto configurando el número de hojas (`max_leaf_nodes`).
- **Frontend Interactivo:** Interfaz moderna en Modo Oscuro con Tailwind CSS que permite predecir y disparar reentrenamientos.

---

## 📁 Estructura del Repositorio
```text
proyecto-hotel-aleks/
├── .github/
│   └── workflows/
│       └── train.yml         # Workflow de GitHub Actions para reentrenamiento
├── model/
│   └── hotel-model-v1.joblib # Modelo entrenado empaquetado
├── app.py                    # Aplicación de FastAPI
├── train.py                  # Script de entrenamiento del modelo
├── hotel_bookings.csv        # Dataset de reservas de hotel
├── index.html                # Frontend web (Panel de control Dark Mode)
├── Dockerfile                # Configuración del contenedor Docker
└── requirements.txt          # Dependencias del proyecto
