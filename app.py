from flask import Flask, render_template, request
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Función para cargar datos desde MySQL
def load_data_from_mysql():
    conexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="prediccion"
    )
    cursor = conexion.cursor()
    consulta = "SELECT ProductId, NameProduct, Date, UnitPrice, Quantity FROM preddic"
    cursor.execute(consulta)
    datos_bimestrales = cursor.fetchall()
    cursor.close()
    conexion.close()

    datos_por_producto = {}
    for dato in datos_bimestrales:
        producto_id = dato[0]
        if producto_id not in datos_por_producto:
            datos_por_producto[producto_id] = []
        datos_por_producto[producto_id].append({
            "ProductId": dato[0],
            "NameProduct": dato[1],
            "Date": datetime.strptime(dato[2], '%Y-%m-%d'),
            "UnitPrice": dato[3],
            "Quantity": dato[4]
        })
    
    return datos_por_producto

# Función para cargar y predecir usando el modelo
def load_and_predict(model_file, X_new, scaler_X, scaler_y):
    model = keras.models.load_model(model_file)
    X_new_scaled = scaler_X.transform(X_new)
    predictions = model.predict(X_new_scaled)
    return scaler_y.inverse_transform(predictions.reshape(-1, 1))

# Configurar los datos y escaladores
def prepare_data_and_scalers():
    datos_por_producto = load_data_from_mysql()

    # Preparar datos para el modelo
    X = []
    y = []
    for producto, datos_producto in datos_por_producto.items():
        for dato in datos_producto:
            fecha = dato["Date"]
            X.append([fecha.year, fecha.month, fecha.day, dato["ProductId"], dato["UnitPrice"]])
            y.append(dato["Quantity"])

    X = np.array(X)
    y = np.array(y)

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar características
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    return datos_por_producto, scaler_X, scaler_y, X_train_scaled, y_train_scaled

# Preparar datos y escaladores al iniciar la aplicación
datos_por_producto, scaler_X, scaler_y, X_train_scaled, y_train_scaled = prepare_data_and_scalers()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        product_id = int(request.form['product_id'])
        days_ahead = int(request.form['days_ahead'])

        fechas_producto = [dato["Date"] for dato in datos_por_producto[product_id]]
        fecha_mas_reciente = max(fechas_producto)

        fecha_futura = fecha_mas_reciente + timedelta(days=days_ahead)
        year = fecha_futura.year
        month = fecha_futura.month
        day = fecha_futura.day
        unit_price = datos_por_producto[product_id][-1]["UnitPrice"]

        X_new = np.array([[year, month, day, product_id, unit_price]])

        predictions = load_and_predict('modelo.keras', X_new, scaler_X, scaler_y)
        predicted_quantity = predictions[0][0]

        return render_template('predict.html', prediction=predicted_quantity, product_id=product_id, days_ahead=days_ahead)
    return render_template('predict.html')

@app.route('/visualize', methods=['GET'])
def visualize():
    # Establecer la conexión con la base de datos
    conexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="prediccion"
    )

    # Consulta SQL para recuperar los datos de la tabla "prediccion"
    consulta_sql = "SELECT ProductId, NameProduct, Date, UnitPrice, Quantity FROM preddic"

    # Cargar los datos en un DataFrame de Pandas
    df = pd.read_sql_query(consulta_sql, conexion)

    # Cerrar la conexión con la base de datos
    conexion.close()

    # Convertir la columna 'Date' a tipo datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calcular la cantidad total vendida por cada producto
    ventas_por_producto = df.groupby('NameProduct')['Quantity'].sum().sort_values(ascending=False)

    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    ventas_por_producto.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Cantidad total vendida por producto')
    ax.set_xlabel('Producto')
    ax.set_ylabel('Cantidad vendida')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    # Convertir el gráfico a una imagen para mostrar en el navegador
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()

    # Crear el gráfico de dispersión
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['UnitPrice'], df['Quantity'], color='orange', alpha=0.7)
    ax.set_title('Relación entre el precio unitario y la cantidad vendida')
    ax.set_xlabel('Precio unitario')
    ax.set_ylabel('Cantidad vendida')
    plt.grid(True)
    plt.tight_layout()

    # Convertir el gráfico a una imagen para mostrar en el navegador
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode()

    # Crear el gráfico de dispersión con líneas
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Date'], df['Quantity'], marker='o', linestyle='-', color='green')
    ax.set_title('Variación de la demanda a lo largo del tiempo')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Cantidad vendida')
    plt.grid(True)
    plt.tight_layout()

    # Convertir el gráfico a una imagen para mostrar en el navegador
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url3 = base64.b64encode(img.getvalue()).decode()

    return render_template('visualize.html', plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3)

if __name__ == '__main__':
    app.run(debug=True)
