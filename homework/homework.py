# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd

data_train = pd.read_csv('../files/input/train_data.csv.zip', index_col= False, compression="zip")
data_test = pd.read_csv('../files/input/test_data.csv.zip', index_col= False, compression="zip")
columnas_diferentes = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

print(data_train.isnull().sum())


import numpy as np

data_train.rename(columns={"default payment next month" : "default"}, inplace=True)
data_test.rename(columns={"default payment next month" : "default"}, inplace=True)
data_train.drop(columns=["ID"], inplace=True)
data_test.drop(columns=["ID"], inplace=True)

data_train['EDUCATION'] = data_train['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
data_test['EDUCATION'] = data_test['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

data_train['EDUCATION'] = data_train['EDUCATION'].apply(lambda x: x if x > 0 else np.nan)
data_test['EDUCATION'] = data_test['EDUCATION'].apply(lambda x: x if x > 0 else np.nan)

data_train['MARRIAGE'] = data_train['MARRIAGE'].apply(lambda x: x if x > 0 else np.nan)
data_test['MARRIAGE'] = data_test['MARRIAGE'].apply(lambda x: x if x > 0 else np.nan)

pay_columns = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

data_train[pay_columns] = data_train[pay_columns].applymap(lambda x: x if x >= 0 else np.nan)
data_test[pay_columns] = data_test[pay_columns].applymap(lambda x: x if x >= 0 else np.nan)

data_train.dropna(inplace=True)
data_test.dropna(inplace=True)

data_train = data_train.astype(int)
data_test = data_test.astype(int)

print(data_test.isnull().sum())


import pickle


# Dividir en características (X) y etiquetas (y)
X_train = data_train.drop(columns="default")
y_train = data_train["default"]

X_test = data_test.drop(columns="default")
y_test = data_test["default"]


print(X_test.shape)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


#columnas categoricas

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

#preprocesamiento

preprocessor = ColumnTransformer(
    transformers= [
        ("cat", OneHotEncoder(dtype='int'), categorical_features)
    ],
    remainder='passthrough',
)

#modelo base

clf = RandomForestClassifier(random_state=42, class_weight="balanced")

#pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', clf)
])



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)


#hiperparametros

param_grid = {
    "clf__n_estimators": [150, 160, 170],  # Reducir el número de árboles
    "clf__max_depth": [21, 26, 31],  # Árboles más simples
    "clf__min_samples_split": [3, 5],  # Mayor número de muestras para dividir nodos
    "clf__min_samples_leaf": [2, 4]  # Mayor número de muestras por hoja
}

#balancear clases

model = GridSearchCV(pipeline, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1, refit=True)

model.fit(X_train, y_train)


print(model.best_params_)


best_model = model.best_estimator_


import os
import pickle
import gzip

# Ruta del directorio donde se guardará el archivo
dir_path = 'files/models'

# Verificar si el directorio existe, si no, crearlo
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Ruta del archivo GZIP
gzip_file_path = os.path.join(dir_path, 'model.pkl.gz')

# Guardar el modelo comprimido como un archivo GZIP
with gzip.open(gzip_file_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Modelo guardado correctamente en {gzip_file_path}")



# Predicciones
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)



# Métricas

metrics = {
    "Train": {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Balanced accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred),
        "Recall": recall_score(y_train, y_train_pred),
        "F1-Score": f1_score(y_train, y_train_pred)
    },
    "Test":{
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1-Score": f1_score(y_test, y_test_pred)
    }
}

print(metrics)



# Matriz de Confusión
for dataset, y_true, y_pred in [("Train", y_train, y_train_pred), (" Test", y_test, y_test_pred)]:
  cm = confusion_matrix(y_true, y_pred)
  print(f"Matriz de confusión ({dataset}):\n", cm)




import json

# Lista para almacenar las líneas del archivo JSON
results = []

# Agregar información de metrics para train y test
for dataset in metrics:
    results.append({
        'type': 'metrics',
        'dataset': dataset.lower(),
        'precision': float(metrics[dataset]['Precision']), 
        'balanced_accuracy': float(metrics[dataset]['Balanced accuracy']),       
        'recall': float(metrics[dataset]['Recall']),
        'f1_score': float(metrics[dataset]['F1-Score'])
    })

# Generar las matrices de confusión para train y test
for dataset, y_true, y_pred in [("Train", y_train, y_train_pred), ("Test", y_test, y_test_pred)]:
    # Calculamos la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Organizar la matriz de confusión en un diccionario
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset.lower(),  # 'train' o 'test'
        "true_0": {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        "true_1": {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]}
    }
    
    # Agregar la matriz de confusión a la lista de resultados
    results.append(cm_dict)


print(results)



import os
import json
import numpy as np

# Ruta donde se guardará el archivo JSON
output_path = "files/output"

# Crear la carpeta de salida si no existe
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Función para convertir tipos de datos de numpy (int64, float64) a tipos estándar de Python
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.int64):  # Si el valor es un int64 de numpy
        return int(obj)  # Convertir a tipo int de Python
    elif isinstance(obj, np.float64):  # Si el valor es un float64 de numpy
        return float(obj)  # Convertir a tipo float de Python
    else:
        return obj

# Guardar cada elemento en una línea separada del archivo JSON
with open('files/output/metrics.json', 'w', encoding='utf-8') as f:  # Abrir en modo texto con codificación UTF-8
    for result in results:
        result = convert_numpy_types(result)  # Convertir los valores de int64 y float64 a tipos estándar
        json.dump(result, f, ensure_ascii=False)  # Escribir el objeto en formato JSON
        f.write('\n')  # Escribir un salto de línea después de cada línea

print(f"Archivo guardado correctamente en {output_path}")



