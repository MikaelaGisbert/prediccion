import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
consulta = "SELECT ProductId, NameProduct, Date, UnitPrice, Quantity FROM datosf"

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
        "Date": dato[2],
        "UnitPrice": dato[3],
        "Quantity": dato[4]
    })

# Convertir los datos a un formato adecuado para el modelo
X = []
y = []
for producto, datos_producto in datos_por_producto.items():
    for dato in datos_producto:
        X.append([dato["ProductId"], dato["UnitPrice"]])
        y.append(dato["Quantity"])

# Convertir X y y a arrays de NumPy
X = np.array(X)
y = np.array(y)


# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar características
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Definir arquitectura de la red neuronal
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2)

# Evaluación del modelo en conjunto de prueba
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Deshacer la normalización

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R^2):", r2)

# Visualización de la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
#*****************************************
# Interpretación de los resultados
# Aquí puedes agregar más análisis e interpretaciones según sea necesario

# Obtener lista de productos disponibles
productos_disponibles = list(datos_por_producto.keys())

print("Productos disponibles:")
for i, producto_id in enumerate(productos_disponibles, start=1):
    nombre_producto = datos_por_producto[producto_id][0]["NameProduct"]
    print(f"{i}. {nombre_producto}")

seleccion = int(input("Seleccione el número del producto para ver el análisis: ")) - 1

# Obtener el producto seleccionado
producto_seleccionado_id = productos_disponibles[seleccion]
producto_seleccionado_nombre = datos_por_producto[producto_seleccionado_id][0]["NameProduct"]

datos_producto_seleccionado = datos_por_producto[producto_seleccionado_id]



# Obtener la suma de la cantidad vendida para cada bimestre
demandas_bimestrales = {}
for dato in datos_producto_seleccionado:
    bimestre = dato["Date"].split("-")[1]  # Obtener el número del bimestre
    demanda = dato["Quantity"]
    if bimestre in demandas_bimestrales:
        demandas_bimestrales[bimestre] += demanda
    else:
        demandas_bimestrales[bimestre] = demanda

# Determinar el bimestre con mayor demanda
bimestre_max_demanda = max(demandas_bimestrales, key=demandas_bimestrales.get)
max_demanda = demandas_bimestrales[bimestre_max_demanda]

# Imprimir respuesta
print(f"\nAnálisis del producto '{producto_seleccionado_nombre}':")
print(f"Este producto tendrá una mayor demanda en el período de tiempo de {bimestre_max_demanda}:")
print(f"Total de cantidad vendida en ese bimestre: {max_demanda}")
print("-----------------------------------------------------------------")
print("Bimestre\tCantidad Vendida\tPrecio Unitario")
print("-----------------------------------------------------------------")
for dato in datos_producto_seleccionado:
    bimestre = dato["Date"].split("-")[1]  # Obtener el número del bimestre
    print(f"{bimestre}\t\t\t\t{dato['Quantity']}\t\t\t{dato['UnitPrice']}")