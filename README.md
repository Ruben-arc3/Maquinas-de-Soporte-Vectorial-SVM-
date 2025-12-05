# Clasificación de Mensajes Spam con SVM

Este proyecto implementa un sistema de clasificación de mensajes de texto (spam y ham) utilizando una Máquina de Soporte Vectorial (SVM) en Python. El objetivo es entrenar un modelo capaz de identificar si un mensaje es spam o no spam a partir de su contenido textual.

Dataset utilizado:  
https://www.kaggle.com/datasets/team-ai/spam-text-message-classification  
El dataset contiene dos columnas: category (spam o ham) y message (texto del mensaje).

Tecnologías utilizadas: Python 3, pandas, scikit-learn, TF-IDF Vectorizer, SVM, Git y GitHub.

Estructura del proyecto:
- spam.csv
- svm_spam.py
- requirements.txt
- .gitignore
- README.md

Instalación:
1. Clonar el repositorio:
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd TU_REPOSITORIO
2. Crear entorno virtual:
   python -m venv venv
3. Activar entorno virtual:
   En Windows: venv\Scripts\activate  
   En Linux/Mac: source venv/bin/activate
4. Instalar dependencias:
   pip install -r requirements.txt

Ejecución:
Ejecutar el programa con:
python svm_spam.py

Funcionamiento:
El programa carga el dataset, convierte las etiquetas spam y ham a valores numéricos, divide los datos en entrenamiento y prueba, transforma los mensajes en vectores usando TF-IDF, entrena una Máquina de Soporte Vectorial con kernel lineal, evalúa el modelo y permite probar la predicción con mensajes nuevos.

Resultados:
El modelo obtiene una alta precisión en la detección de mensajes spam, demostrando la efectividad de las SVM en la clasificación de texto.

Autor:
Proyecto desarrollado con fines educativos para el estudio de aprendizaje automático y clasificación de mensajes de texto.
