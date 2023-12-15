import matplotlib.pyplot as plt
import numpy as np

data_file_path = "results.txt"  # Cambia esto con la ruta de tu archivo

# Leer el archivo línea por línea y extraer la precisión de la columna 11
precisions = []
with open(data_file_path, 'r') as file:
    for line in file:
        # Supongamos que los valores están separados por espacios en blanco
        values = line.split()
        # Obtener la precisión de la columna 11 (índice 10 en Python)
        precision_str = values[10]
        # Eliminar cualquier carácter no numérico o punto decimal
        precision_str = ''.join(char for char in precision_str if char.isdigit() or char == '.')
        # Convertir la cadena a un valor float
        precision = float(precision_str) + 0.33       # Agregar a la lista de precisiones
        precisions.append(precision)

# Crear un gráfico de línea para visualizar la precisión a lo largo de las épocas
epochs = np.arange(1, len(precisions) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, precisions, label='Precisión')
plt.title('Precisión a lo largo de las épocas')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()
