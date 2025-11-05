import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # evita error de OpenMP

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos de NLTK (solo la primera vez)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")  # ← ESTA LÍNEA ES LA IMPORTANTE

stop_words = set(stopwords.words("spanish"))
lemmatizer = WordNetLemmatizer()

def limpiar_texto(texto):
    texto = str(texto).lower()
    palabras = texto.split()  # evita error con punkt_tab
    palabras = [lemmatizer.lemmatize(p) for p in palabras if p.isalpha() and p not in stop_words]
    return palabras

# ------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# ------------------------------------------------------
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")

st.title("🎧 Análisis de Opiniones de Clientes - Audífonos Xiaomi")

st.sidebar.title("📊 Navegación")
menu = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "📂 Dataset", "☁️ Visualizaciones", "🔍 Análisis de Sentimientos", "✍️ Predicción Nueva"],
)

# ------------------------------------------------------
# MODELO DE HUGGING FACE
# ------------------------------------------------------
modelo = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    device=0  # usa GPU si está disponible
)

# ------------------------------------------------------
# SECCIÓN: INICIO
# ------------------------------------------------------
if menu == "🏠 Inicio":
    st.markdown("""
    ### Bienvenido 👋  
    Esta aplicación permite analizar opiniones de clientes sobre productos, empresas o lugares.  
    Podrás:
    - Subir un archivo `.csv` o `.xlsx` con las opiniones.
    - Visualizar una **nube de palabras** y las **palabras más frecuentes**.
    - Analizar el **sentimiento (positivo, negativo o neutro)** de cada opinión.
    - Ingresar nuevas opiniones para clasificarlas en tiempo real.  
    """)

# ------------------------------------------------------
# SECCIÓN: CARGA DE DATASET
# ------------------------------------------------------
elif menu == "📂 Dataset":
    st.subheader("📂 Carga de Dataset")

    ruta_archivo = "Opiniones.xlsx"

    if os.path.exists(ruta_archivo):
        if ruta_archivo.endswith(".xlsx"):
            df = pd.read_excel(ruta_archivo)
        else:
            df = pd.read_csv(ruta_archivo)

        st.success("✅ Archivo cargado automáticamente desde el proyecto")
        st.dataframe(df.head())
        st.session_state["dataset"] = df
    else:
        st.error("❌ No se encontró 'Opiniones.xlsx' en la carpeta del proyecto.")
        st.info("💡 Puedes subirlo manualmente desde el botón a continuación:")

        uploaded_file = st.file_uploader("Sube tu archivo:", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success("✅ Archivo subido correctamente")
            st.dataframe(df.head())
            st.session_state["dataset"] = df

# ------------------------------------------------------
# SECCIÓN: VISUALIZACIONES
# ------------------------------------------------------
elif menu == "☁️ Visualizaciones":
    if "dataset" in st.session_state:
        df = st.session_state["dataset"]

        stop_words = set(stopwords.words("spanish"))
        lemmatizer = WordNetLemmatizer()

        def limpiar_texto(texto):
            palabras = nltk.word_tokenize(str(texto).lower())
            palabras = [lemmatizer.lemmatize(p) for p in palabras if p.isalpha() and p not in stop_words]
            return palabras

        # ✅ CORREGIDO: usar la columna "Opinion"
        df["tokens"] = df["Opinion"].apply(limpiar_texto)

        todas_palabras = [p for tokens in df["tokens"] for p in tokens]

        # --- Nube de Palabras ---
        st.subheader("☁️ Nube de Palabras")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(todas_palabras))
        st.image(wordcloud.to_array(), use_container_width=True)

        # --- Top 10 palabras ---
        st.subheader("📊 Top 10 Palabras Más Frecuentes")
        top_words = Counter(todas_palabras).most_common(10)
        freq_df = pd.DataFrame(top_words, columns=["Palabra", "Frecuencia"])

        fig, ax = plt.subplots()
        sns.barplot(data=freq_df, x="Frecuencia", y="Palabra", palette="Blues_d", ax=ax)
        st.pyplot(fig)

        # --- Gráfico adicional: longitud promedio ---
        st.subheader("🧩 Longitud Promedio de Opiniones")

        # ✅ CORREGIDO: usar "Opinion"
        df["longitud"] = df["Opinion"].apply(lambda x: len(str(x).split()))

        fig2, ax2 = plt.subplots()
        sns.histplot(df["longitud"], bins=10, kde=True, color="purple", ax=ax2)
        ax2.set_xlabel("Número de palabras por opinión")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Primero debes cargar el archivo en la sección 📂 Dataset.")

# ------------------------------------------------------
# SECCIÓN: ANÁLISIS DE SENTIMIENTOS
# ------------------------------------------------------
elif menu == "🔍 Análisis de Sentimientos":
    if "dataset" in st.session_state:
        df = st.session_state["dataset"]
        st.subheader("🔍 Clasificación de Opiniones")

        # ✅ CORREGIDO
        df["sentimiento"] = df["Opinion"].apply(lambda x: modelo(x)[0]["label"])

        st.dataframe(df[["Opinion", "sentimiento"]])

        # --- Gráfico de distribución ---
        st.subheader("📈 Distribución de Sentimientos")
        conteo = df["sentimiento"].value_counts()

        fig3, ax3 = plt.subplots()
        ax3.pie(conteo, labels=conteo.index, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig3)
    else:
        st.warning("⚠️ Primero debes cargar el archivo en la sección 📂 Dataset.")

# ------------------------------------------------------
# SECCIÓN: NUEVA PREDICCIÓN
# ------------------------------------------------------
elif menu == "✍️ Predicción Nueva":
    st.subheader("✍️ Analiza una nueva opinión")
    texto = st.text_area("Escribe aquí tu comentario sobre los audífonos:")
    
    if st.button("Analizar Sentimiento"):
        if texto.strip() != "":
            resultado = modelo(texto)[0]
            st.success(f"**Sentimiento detectado:** {resultado['label']}  (confianza: {resultado['score']:.2f})")
        else:
            st.warning("Por favor escribe una opinión antes de analizar.")

