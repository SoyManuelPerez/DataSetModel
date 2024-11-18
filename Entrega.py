from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import uvicorn

# Crear instancia de FastAPI
app = FastAPI()

# Definir el modelo Pydantic para la solicitud
class TrainModelRequest(BaseModel):
    file_path: str

# Ruta de inicio de la API
@app.get("/")
def home():
    """
    Endpoint de bienvenida.
    """
    return {"message": "Bienvenido a la API de clasificación de vinos"}

# Ruta para procesar y entrenar el modelo
@app.post("/train-model/")
def train_model(request: TrainModelRequest):
    """
    Endpoint para entrenar un modelo de clasificación a partir de un dataset CSV.
    Parámetro:
    - request: Un objeto con la ruta al archivo CSV.
    """
    try:
        file_path = request.file_path  # Obtener la ruta del archivo
        # Leer el archivo CSV
        data = pd.read_csv(file_path, delimiter=';', encoding='latin1')

        # Filtrar filas con al menos el 50% de valores no nulos
        thresh_value = int(data.shape[1] * 0.5)
        data = data.dropna(thresh=thresh_value)

        # Mostrar la distribución de clases en la columna 'quality'
        class_distribution = data['quality'].value_counts().to_dict()

        # Generar y guardar un gráfico pairplot para el análisis
        sns.pairplot(data, hue='quality', plot_kws={'alpha': 0.6})
        plt.savefig("pairplot.png")  # Guardar el gráfico en un archivo

        # Preparar los datos para el modelo
        X = data.iloc[:, :-1].values.astype(float)  # Todas las columnas excepto la última
        Y = data.iloc[:, -1].values  # Última columna (objetivo)

        # Dividir los datos en entrenamiento, validación y prueba
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.05, random_state=42, stratify=Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.15, random_state=42, stratify=Y_train)

        # Crear y entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, Y_train)

        # Evaluar el modelo
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        report = classification_report(Y_test, Y_pred, output_dict=True)  # Cambiado para devolver como dict
        confusion = confusion_matrix(Y_test, Y_pred).tolist()  # Convertir a lista para serializar JSON

        # Retornar los resultados
        return {
            "class_distribution": class_distribution,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion
        }

    except Exception as e:
        # Manejo de errores
        return {"error": str(e)}

# Ejecutar la API en local
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
