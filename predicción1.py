import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector

# Establecer la conexión con la base de datos
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="prediccion"
)

# Consulta SQL para recuperar los datos de la tabla datosf
consulta_sql = "SELECT ProductId, NameProduct, Date, UnitPrice, Quantity FROM datosf"

# Cargar los datos en un DataFrame de Pandas
df = pd.read_sql_query(consulta_sql, conexion)

# Cerrar la conexión con la base de datos
conexion.close()

# Convertir la columna 'Date' a tipo datetime
df['Date'] = pd.to_datetime(df['Date'])

# Mostrar las primeras filas del DataFrame
print(df.head())


# Calcular la cantidad total vendida por cada producto
ventas_por_producto = df.groupby('NameProduct')['Quantity'].sum().sort_values(ascending=False)

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
ventas_por_producto.plot(kind='bar', color='skyblue')
plt.title('Cantidad total vendida por producto')
plt.xlabel('Producto')
plt.ylabel('Cantidad vendida')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(df['UnitPrice'], df['Quantity'], color='orange', alpha=0.7)
plt.title('Relación entre el precio unitario y la cantidad vendida')
plt.xlabel('Precio unitario')
plt.ylabel('Cantidad vendida')
plt.grid(True)
plt.tight_layout()
plt.show()

# Crear el gráfico de dispersión con líneas
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Quantity'], marker='o', linestyle='-', color='green')
plt.title('Variación de la demanda a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.grid(True)
plt.tight_layout()
plt.show()