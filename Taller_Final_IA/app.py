import streamlit as st
from PIL import Image
import easyocr
import os
import numpy as np
from groq import Groq
from huggingface_hub import InferenceClient

# --- CONFIGURACIÓN DE LA PÁGINA Y CLAVES ---

st.set_page_config(page_title="Taller IA: OCR + LLM", layout="wide")

# Cargar las claves de API desde los secretos de Streamlit
# Asegúrate de que los nombres coincidan con los que pusiste en Streamlit Cloud
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except KeyError:
    st.error("No se encontraron las claves de API en los secretos de Streamlit.")
    st.stop()

# --- MÓDULO 1: EL LECTOR DE IMÁGENES (OCR) [cite: 58] ---

# Desafío de Caché: Usamos @st.cache_resource
# para cargar el modelo OCR solo una vez.
@st.cache_resource
def load_ocr_model():
    """Carga el modelo EasyOCR en memoria (cacheado)."""
    # Usamos 'es' (español) e 'en' (inglés)
    reader = easyocr.Reader(['es', 'en'], gpu=False) 
    return reader

# 1. Crear la Interfaz Básica [cite: 61]
st.title("Taller IA: Construcción de una Aplicación Multimodal ")
st.header("Módulo 1: Lector de Imágenes (OCR) 📸")

# 2. Implementar la Carga de Archivos [cite: 66]
uploaded_file = st.file_uploader(
    "Sube una imagen para extraer el texto",
    type=["png", "jpg", "jpeg"] # [cite: 68]
)

# 3. Cargar y Ejecutar el Modelo OCR
if uploaded_file is not None:
    # Mostrar la imagen subida [cite: 73]
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir la imagen para EasyOCR [cite: 74]
    # Necesitamos pasarla como bytes o como un array de numpy
    img_bytes = uploaded_file.getvalue()
    
    with st.spinner("Procesando imagen con OCR..."):
        # Cargar el modelo (lo tomará del caché si ya está cargado)
        reader = load_ocr_model()
        
        # 4. Procesar y Mostrar Resultados [cite: 72]
        # Ejecutar el modelo OCR [cite: 75]
        results = reader.readtext(img_bytes)
        
        # Juntar todo el texto extraído
        extracted_text = " ".join([res[1] for res in results])

        # Desafío de Persistencia:
        # Guardar el texto extraído en el st.session_state
        st.session_state['extracted_text'] = extracted_text
        
        # Mostrar el texto extraído [cite: 76]
        st.text_area(
            "Texto Extraído por OCR:",
            extracted_text,
            height=250,
            key="ocr_output"
        )

# --- MÓDULOS 2 y 3: CONEXIÓN CON LLMS Y FLEXIBILIDAD [cite: 77, 95] ---

# Solo mostramos esta sección si ya hay texto extraído [cite: 88, 94]
if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
    
    st.divider()
    st.header("Módulos 2 y 3: Análisis con LLMs 🧠")
    
    # Texto extraído del estado de la sesión
    text_to_analyze = st.session_state['extracted_text']

    # --- Interfaz de Usuario (UI) ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Módulo 3: Elección de Proveedor [cite: 102]
        provider = st.radio(
            "Elige el proveedor de LLM:",
            ("GROQ", "Hugging Face"),
            key="provider"
        )

        # Módulo 2: Elección de Tarea [cite: 85]
        task_prompt = st.selectbox(
            "Elige la tarea a realizar:",
            (
                "Resumir el texto en 3 puntos clave",
                "Identificar las entidades principales (personas, lugares, organizaciones)",
                "Traducir el texto al inglés",
                "Analizar el sentimiento del texto (positivo, negativo o neutral)",
                "Generar 3 preguntas sobre el texto"
            ),
            key="task"
        )
        
        # Módulo 2: Elección de Modelo (solo para GROQ) [cite: 84]
        if provider == "GROQ":
            model_selection = st.selectbox(
                "Elige el modelo de GROQ:",
                ("llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"),
                key="groq_model"
            )
        else:
            # Módulo 3: Modelos de Hugging Face
            model_selection = st.text_input(
                "Modelo de Hugging Face (ej: mistralai/Mixtral-8x7B-Instruct-v0.1):",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                key="hf_model"
            )

    with col2:
        # Módulo 3: Control de Parámetros [cite: 98]
        temperature = st.slider(
            "Temperatura (Creatividad) [cite: 99]",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature"
        )
        
        max_tokens = st.slider(
            "Máximos Tokens (Longitud) [cite: 99]",
            min_value=50,
            max_value=4096,
            value=512,
            step=64,
            key="max_tokens"
        )

    # Módulo 2: Botón de Análisis [cite: 86]
    analyze_button = st.button("Analizar Texto con LLM", type="primary")

    # --- Lógica de la API ---
    
    if analyze_button:
        with st.spinner(f"Analizando texto con {provider}... Por favor espera."):
            try:
                # Módulo 3: Lógica Condicional [cite: 103]
                if provider == "GROQ":
                    # Módulo 2: Lógica de la API de GROQ [cite: 87]
                    client = Groq(api_key=GROQ_API_KEY) # [cite: 88]
                    
                    # Estructura correcta del prompt [cite: 90]
                    messages = [
                        {
                            "role": "system",
                            "content": f"Eres un asistente experto. El usuario te dará un texto y una tarea. Debes realizar la tarea solicitada sobre el texto. La tarea es: {task_prompt}."
                        },
                        {
                            "role": "user",
                            "content": f"El texto para analizar es el siguiente:\n\n---\n{text_to_analyze}\n---"
                        }
                    ]
                    
                    # Llamada a la API [cite: 89, 100]
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model_selection,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Mostrar la respuesta [cite: 91]
                    response_content = chat_completion.choices[0].message.content
                    st.markdown("### Respuesta de GROQ")
                    st.markdown(response_content)

                elif provider == "Hugging Face":
                    # Módulo 3: Lógica de la API de Hugging Face [cite: 104]
                    client = InferenceClient(token=HUGGINGFACE_API_KEY) # [cite: 106]
                    
                    # Estructura del prompt para un modelo instruct
                    # (Usamos text_generation que es más flexible [cite: 107])
                    prompt = f"""<s>[INST] Eres un asistente experto. El usuario te dará un texto y una tarea.
Tarea: {task_prompt}
Texto:
{text_to_analyze}
[/INST]
Respuesta:"""
                    
                    # Llamada a la API [cite: 107]
                    response = client.text_generation(
                        model=model_selection,
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=max(temperature, 0.01) # Temp 0 no es válida en HF, usamos 0.01
                    )
                    
                    # Mostrar la respuesta [cite: 110]
                    st.markdown("### Respuesta de Hugging Face")
                    st.markdown(response)

            except Exception as e:
                st.error(f"Error al contactar la API de {provider}: {e}")
