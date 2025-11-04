# Riesgo-Crediticio
Analisis de riesgo crediticio
🏦 Análisis de Riesgo Crediticio

Proyecto de Análisis de Datos | Python | ETL | EDA | Machine Learning

📋 Descripción general

El objetivo de este proyecto es analizar el riesgo crediticio de clientes a partir de datos financieros y personales, con el fin de detectar patrones que permitan predecir la probabilidad de incumplimiento de pago (default).

El análisis se realizó aplicando un proceso ETL completo (Extracción, Transformación y Carga) y un Análisis Exploratorio de Datos (EDA) utilizando Python y librerías especializadas de análisis y visualización.

🧩 Dataset

Fuente: Credit Risk Dataset - Kaggle

Registros: 32.581 (luego de limpieza)

Columnas originales: 12

Descripción general: información demográfica, ingresos, historial crediticio, antigüedad laboral y estado del préstamo.

⚙️ Proceso ETL
🔹 E - Extracción

Se importó el dataset desde Google Drive usando Pandas.

Archivo: credit_risk_dataset.csv.

df = pd.read_csv("/content/drive/MyDrive/Banco Riesgo Crediticio/credit_risk_dataset.csv")

🔹 T - Transformación

Renombrado de columnas para mejorar la legibilidad en español.

Eliminación de duplicados: 165 registros duplicados eliminados.

Tratamiento de valores nulos:

“Antigüedad (trabajo)” → 3% de nulos

“Tasa de interés” → 10% de nulos

Se eliminaron filas con valores nulos (dropna).

Outliers:

Edad máxima corregida a 84 años.

Antigüedad laboral máxima corregida a 41 años (se eliminaron casos irrealistas como 123 años).

df = df[df['Edad'] < 85]
df = df[df['Antiguedad (trabajo)'] < 100]


Datos limpios guardados: credit_risk_dataset_limpio.csv

🔹 L - Carga

El dataset limpio se exportó para futuras etapas de modelado y visualización (Power BI / ML).

df.to_csv('credit_risk_dataset_limpio.csv', index=False)

📊 Análisis Exploratorio (EDA)
🔸 1. Distribución del estado del préstamo
sns.countplot(x=df['Estado del prestamo'])


Conclusión: La mayoría de los préstamos están al día, aunque existe un grupo significativo de impagos (≈20%), útil para modelar riesgo.

🔸 2. Ingreso vs Estado del préstamo
sns.boxplot(x="Estado del prestamo", y="Ingreso", data=df)


Conclusión: Los clientes con ingresos más bajos presentan mayor tasa de incumplimiento, lo que confirma que el nivel de ingresos es un factor de riesgo importante.

🔸 3. Distribución de ingresos
sns.histplot(df['Ingreso'], bins=30, kde=True)


Conclusión: Distribución sesgada a la derecha — la mayoría de los clientes tiene ingresos medios/bajos, y unos pocos concentran ingresos altos.

🔸 4. Relación entre ingreso y monto del préstamo
sns.scatterplot(x='Ingreso', y='Monto del presamo', hue='Monto del presamo', data=df)


Conclusión: Se observa una correlación positiva: a mayor ingreso, mayor monto solicitado, aunque también hay casos de préstamos altos con ingresos bajos (riesgo elevado).

🔸 5. Antigüedad laboral y riesgo
sns.boxplot(x='Estado del prestamo', y='Antiguedad (trabajo)', data=df)


Conclusión: Los clientes con menor antigüedad laboral tienden a incumplir con más frecuencia.
La estabilidad laboral parece ser un factor protector frente al riesgo.

🔸 6. Matriz de correlación
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')


Conclusión:

Correlación negativa moderada entre Ingreso y Estado del préstamo → a menor ingreso, mayor probabilidad de impago.

Correlaciones positivas entre Monto del préstamo, Tasa de interés y Estado del préstamo → mayores montos o tasas se asocian a mayor riesgo.

📈 Conclusiones generales

Ingresos bajos, alta probabilidad de impago.
El ingreso es la variable más influyente en el riesgo crediticio.

Antigüedad laboral y estabilidad reducen el riesgo.
Los clientes con más años de empleo tienden a cumplir sus obligaciones.

Relación préstamo/ingreso clave para evaluar capacidad de pago.

Tasa de interés alta y montos grandes suelen aumentar la morosidad.

Dataset balanceado pero con sesgo de ingresos, ideal para aplicar un modelo de clasificación binaria
