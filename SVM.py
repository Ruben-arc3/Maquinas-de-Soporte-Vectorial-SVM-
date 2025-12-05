import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Cargar dataset
df = pd.read_csv("SPAM.csv")  # Cambia el nombre si tu archivo se llama diferente

# Si tu archivo tiene más columnas, deja solo las necesarias
df = df[['Category', 'Message']]

# 2. Convertir etiquetas a valores numéricos
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# 3. Separar datos (X = mensajes, y = etiquetas)
X = df['Message']
y = df['Category']

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Convertir texto a números usando TF-IDF
vectorizador = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizador.fit_transform(X_train)
X_test_vec = vectorizador.transform(X_test)

# 6. Crear modelo SVM
modelo = SVC(kernel='linear')

# 7. Entrenar el modelo
modelo.fit(X_train_vec, y_train)

# 8. Predicciones
y_pred = modelo.predict(X_test_vec)

# 9. Evaluación del modelo
print("Precisión:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# 10. Probar con un mensaje nuevo
nuevo_mensaje = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
nuevo_mensaje_vec = vectorizador.transform(nuevo_mensaje)
prediccion = modelo.predict(nuevo_mensaje_vec)

if prediccion[0] == 1:
    print("\nEl mensaje es SPAM")
else:
    print("\nEl mensaje es HAM (no spam)")
