# Decision Trees: Python vs R

Una guía rápida y práctica sobre los árboles de decisión, sus fundamentos, y cómo implementarlos en Python y R.

---

## Tabla de Contenidos

1. [Explicación Teórica Básica](#explicación-teórica-básica)
2. [Implementación en Python](#implementación-en-python)
3. [Implementación en R](#implementación-en-r)
4. [Comparación: Diferencias y Similitudes](#comparación-diferencias-y-similitudes)
5. [Conclusiones](#conclusiones)

---

## Explicación Teórica Básica

Los árboles de decisión son algoritmos supervisados usados tanto para problemas de clasificación como de regresión. Su estructura está compuesta por:

- **Nodo raíz:** Punto inicial que contiene todo el dataset.
- **Nodos internos:** Decisiones basadas en una característica del dataset.
- **Hojas:** Resultado o categoría final.

El objetivo principal es dividir el dataset en subconjuntos más homogéneos mediante decisiones basadas en métricas como:

- **Gini Impurity**
- **Entropy (ID3, C4.5)**
- **Error de Clasificación**

La visualización de un árbol permite una interpretación clara y directa de las decisiones tomadas en cada nodo.

---

## Implementación en Python

El siguiente código implementa y visualiza un árbol de decisión en Python utilizando `scikit-learn` y `matplotlib`.

```python
# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el conjunto de datos 'iris'
iris = load_iris()

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=213
)

# Crear el modelo de árbol de decisión
modelo_arbol = DecisionTreeClassifier(random_state=123)
modelo_arbol.fit(X_train, y_train)

# Predecir las clases para los datos de prueba
predicciones = modelo_arbol.predict(X_test)

# Crear una tabla de confusión
tabla_confusion = confusion_matrix(y_test, predicciones)
print("Tabla de Confusión:\n", tabla_confusion)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print(f"Precisión del modelo: {precision * 100:.2f}%")

# Visualizar el árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(modelo_arbol, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Árbol de Decisión - Dataset Iris")
plt.show()

---

```R
# Instalamos los paquetes para arboles de regresion y su visualizacion
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

# Cargamos el conjunto incorporado en R 'iris'. De no estar familiarizado podemos echar un vistazo previamente a la estructura y columnas.
data(iris)

# Dividimos los datos en entrenamiento (70%) y prueba (30%). Estableceremos la semilla en 123 para la reproducibilidad.
set.seed(123)
sample_indices <- sample(1:nrow(iris), 0.7 * nrow(iris))
#Creamos el conjunto de datos de entrenamiento.
train_data <- iris[sample_indices, ]
#Creamos el conjunto de datos para test. Seran los restantes a los de entrenamiento.
test_data <- iris[-sample_indices, ]

# Creamos el modelo de árbol de decisión con rpart sobre la variable "Species" y prediciendo con todas las variables. Especificamos class porque es un problema de clasificacion de una variable categorica.
modelo_arbol <- rpart(Species ~ ., data = train_data, method = "class")

# Una vez entrenado el algoritmo con los datos de entrenamiento hacemos las predicciones con los datos de prueba
predicciones <- predict(modelo_arbol, test_data, type = "class")

# Creamos una tabla de confusión para evaluar el rendimiento
tabla_confusion <- table(Predicciones = predicciones, Actual = test_data$Species)
print(tabla_confusion)

# Calculamos la precisión del modelo
precision <- sum(diag(tabla_confusion)) / sum(tabla_confusion)
cat("Precisión del modelo:", round(precision * 100,2), "%\n")

# Visualizamos el árbol de decisión para el entrenamiento
rpart.plot(modelo_arbol)

---
Comparación: Diferencias y Similitudes
Aspecto	Python	R
Simplicidad	Requiere configuraciones adicionales para visualización.	La función rpart.plot permite una visualización rápida y sencilla.
Velocidad	Excelente para datasets grandes gracias a scikit-learn.	Ideal para análisis estadísticos en profundidad.
Comunidad	Comunidad amplia con recursos extensivos.	Orientado a investigación académica y estadística.

--- 
Conclusiones
Ambos lenguajes son herramientas poderosas para trabajar con árboles de decisión.
Python es más versátil para proyectos en producción y manejo de grandes datasets.
R sobresale en análisis exploratorios y visualización intuitiva.
