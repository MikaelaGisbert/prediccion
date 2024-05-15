from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
import mysql.connector
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


app = Flask(__name__)

# Conexión a la base de datos MySQL
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="prediccion"
)

# Consulta SQL para obtener los datos bimestrales
consulta = "SELECT ProductId, NameProduct, Date, UnitPrice, Quantity FROM datosf"

# Ejecutar la consulta
cursor = conexion.cursor()
cursor.execute(consulta)
datos_bimestrales = cursor.fetchall()
cursor.close()
conexion.close()

# Procesamiento de los datos
datos_por_producto = {}
for dato in datos_bimestrales:
    producto_id = dato[0]
    if producto_id not in datos_por_producto:
        datos_por_producto[producto_id] = []
    datos_por_producto[producto_id].append({
        "ProductId": dato[0],
        "NameProduct": dato[1],
        "Date": dato[2],
        "UnitPrice": dato[3],
        "Quantity": dato[4]
    })

X = []
y = []
for producto, datos_producto in datos_por_producto.items():
    for dato in datos_producto:
        X.append([dato["ProductId"], dato["UnitPrice"]])
        y.append(dato["Quantity"])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Gráfico de pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.savefig('static/loss_plot.png')
plt.close()

# Función para obtener la lista de productos disponibles
def get_available_products():
    productos_disponibles = list(datos_por_producto.keys())
    nombres_productos = [datos_por_producto[product_id][0]["NameProduct"] for product_id in productos_disponibles]
    return zip(productos_disponibles, nombres_productos)

# Función de inicio
@app.route('/')
def index():
    return render_template('index.html', products=get_available_products())

# Función de resultados
# Función de resultados
@app.route('/result', methods=['POST'])
def result():
    products = get_available_products()  # Obtener lista de productos disponibles
    return render_template('result.html', mse=mse, r2=r2, products=products)


# Función de predicción
# Función de predicción
@app.route('/predict', methods=['POST'])
def predict():
    products = get_available_products()  # Obtener lista de productos disponibles
    
    # Obtener ID del producto seleccionado del formulario
    selected_product_id = request.form['product']
    selected_product_name = datos_por_producto[selected_product_id][0]["NameProduct"]
    
    # Obtener datos del producto seleccionado para la predicción
    datos_producto_seleccionado = datos_por_producto[selected_product_id]
    X_pred = np.array([[dato["ProductId"], dato["UnitPrice"]] for dato in datos_producto_seleccionado])
    X_pred_scaled = scaler_X.transform(X_pred)
    
    # Realizar la predicción
    y_pred_scaled = model.predict(X_pred_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    prediction = np.mean(y_pred)  # Tomamos el promedio de todas las predicciones
    
    return render_template('result.html', mse=mse, r2=r2, products=products, prediction=prediction, selected_product_name=selected_product_name)

if __name__ == '__main__':
    app.run(debug=True)
