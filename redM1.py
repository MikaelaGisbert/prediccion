import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import AdamW

def load_data_from_mysql():
    # Establecer conexión a la base de datos MySQL
    conexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="prediccion"
    )

    # Crear cursor para ejecutar consultas SQL
    cursor = conexion.cursor()

    # Definir la consulta SQL para obtener los datos bimestrales
    consulta = "SELECT ProductId, NameProduct, Date, UnitPrice, Quantity FROM preddic"

    # Ejecutar la consulta
    cursor.execute(consulta)

    # Obtener todos los resultados de la consulta
    datos_bimestrales = cursor.fetchall()

    # Cerrar cursor y conexión
    cursor.close()
    conexion.close()

    # Convertir los datos a un diccionario por producto
    datos_por_producto = {}
    for dato in datos_bimestrales:
        producto_id = dato[0]
        if producto_id not in datos_por_producto:
            datos_por_producto[producto_id] = []
        datos_por_producto[producto_id].append({
            "ProductId": dato[0],
            "NameProduct": dato[1],
            "Date": datetime.strptime(dato[2], '%Y-%m-%d'),  # Convertir la cadena de fecha a objeto de fecha
            "UnitPrice": dato[3],
            "Quantity": dato[4]
        })

    # Convertir los datos a un formato adecuado para el modelo
    X = []
    y = []
    for producto, datos_producto in datos_por_producto.items():
        for dato in datos_producto:
            # Normalizar ProductId
            producto_id = dato["ProductId"]
            # Normalizar la fecha
            fecha = dato["Date"]
            year = fecha.year
            month = fecha.month
            day = fecha.day
            # Agregar los datos normalizados a X
            X.append([year, month, day, dato["ProductId"], dato["UnitPrice"]])
            y.append(dato["Quantity"])

    # Convertir X y y a arrays de NumPy
    X = np.array(X)
    y = np.array(y)

    return X, y, datos_por_producto

def create_sequential_model(input_shape):
    model = keras.models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(150, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])
    return model

import tensorflow.keras.backend as K

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

def train_and_save_model(X_train, y_train, X_valid, y_valid, epochs=60):
    model = create_sequential_model(input_shape=X_train.shape[1:])
    model.compile(optimizer=AdamW(), loss='mse')

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=18, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[early_stopping_cb])
    model.save('modelo.keras')

    return model, history

def load_and_predict(model_file, X_new):
    model = keras.models.load_model(model_file)
    predictions = model.predict(X_new)
    return predictions

def visualize_training_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Función de Pérdida')
    plt.legend()
    plt.show()

def predict_future_demand(datos_por_producto, scaler_X, scaler_y, model_file='modelo.keras'):
    product_id = int(input("Ingresa el ProductId del producto: "))
    days_ahead = int(input("Ingresa la cantidad de días a predecir: "))

    # Encontrar la fecha más reciente para el producto seleccionado
    fechas_producto = [dato["Date"] for dato in datos_por_producto[product_id]]
    fecha_mas_reciente = max(fechas_producto)

    # Calcular la fecha futura
    fecha_futura = fecha_mas_reciente + timedelta(days=days_ahead)
    year = fecha_futura.year
    month = fecha_futura.month
    day = fecha_futura.day

    # Obtener el UnitPrice del producto seleccionado (usamos el último conocido)
    unit_price = datos_por_producto[product_id][-1]["UnitPrice"]

    # Preparar los datos para la predicción
    X_new = np.array([[year, month, day, product_id, unit_price]])
    X_new_scaled = scaler_X.transform(X_new)

    # Cargar el modelo y hacer la predicción
    predictions = load_and_predict(model_file, X_new_scaled)
    predicted_quantity = scaler_y.inverse_transform(predictions.reshape(-1, 1))

    print(f"La demanda predicha para el producto {product_id} en {days_ahead} días es {predicted_quantity[0][0]} unidades.")

# Cargar datos
X, y, datos_por_producto = load_data_from_mysql()

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar características
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Entrenar y guardar el modelo si es necesario
model, history = train_and_save_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

# Visualizar la pérdida durante el entrenamiento
visualize_training_loss(history)

# Predecir demanda futura
predict_future_demand(datos_por_producto, scaler_X, scaler_y, model_file='modelo.keras')
