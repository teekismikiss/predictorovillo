import pandas as pd

#DATASET
data = {
"largo_cm": [
80, 90, 100, 100, 110,
110, 120, 120, 130, 130,
140, 140, 150, 150, 160,
160, 170, 170, 180, 180,
190, 190, 200, 200, 210,
210, 220, 220, 230, 240
],

"ancho_cm": [
    60, 70, 70, 80, 70,
    90, 80, 100, 90, 110,
    100, 120, 100, 120, 110,
    130, 120, 140, 130, 150,
    140, 160, 150, 170, 160,
    180, 170, 190, 180, 200
],

"ovillos": [
    2, 3, 3, 4, 4,
    5, 5, 6, 6, 7,
    7, 8, 8, 9, 9,
    10, 10, 11, 12, 13,
    13, 14, 15, 16, 17,
    18, 19, 20, 21, 23
]

}

df = pd.DataFrame(data)

print("DATASET:")
print(df.head())

print("\nDimensiones del dataset:")
print(df.shape)
#SEPARAR X E Y
X = df[["largo_cm", "ancho_cm"]]
y = df["ovillos"]

print("\nVariables de entrada X:")
print(X.head())

print("\nVariable objetivo y:")
print(y.head())

#SEPARAR DATOS DE ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

print("\nDatos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)

#ESCALAR LOS DATOS
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#CREAR Y ENTRENAR EL MODELO KNN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#REALIZAR PREDICCIONES
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train_scaled, y_train)

#REALIZAR PREDICCIONES

predicciones = knn.predict(X_test_scaled)

print("\nVentas reales:", list(y_test))
print("Predicciones:", predicciones)

#EVALUAR EL MODELO
from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score
)

print("\nRESULTADOS DEL MODELO:")

print(f"MAE: {mean_absolute_error(y_test, predicciones):.2f}")

rmse = mean_squared_error(y_test, predicciones) ** 0.5
print(f"RMSE: {rmse:.2f}")

print(f"R2: {r2_score(y_test, predicciones):.2f}")
#PREDECIR UNA MANTA NUEVA
nueva_manta = pd.DataFrame({
"largo_cm": [220],
"ancho_cm": [180]
})

nueva_manta_scaled = scaler.transform(nueva_manta)

ovillos_estimados = knn.predict(nueva_manta_scaled)

print("\nPREDICCIÓN PARA LA MANTA NUEVA:")
print(f"Ovillos estimados: {ovillos_estimados[0]:.2f}")
