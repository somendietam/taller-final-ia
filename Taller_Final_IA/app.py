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
    st.error("No se encontraron las claves de API en los secretos de Streamlit. Asegúrate de haberlas configurado.")
    st.stop()

# --- MÓDULO 1: EL LECTOR DE IMÁGENES (OCR) ---

# Desafío de Caché: Usamos @st.cache_resource
# para cargar el modelo OCR solo una vez.
@st.cache_resource
def load_ocr_model():
    """Carga el modelo EasyOCR en memoria (cacheado)."""
    # Usamos 'es' (español) e 'en' (inglés)
    reader = easyocr.Reader(['es', 'en'], gpu=False) 
    return reader

# 1. Crear la Interfaz Básica
st.title("Taller IA: Construcción de una Aplicación Multimodal")
st.header("Módulo 1: Lector de Imágenes (OCR) 📸")

# 2. Implementar la Carga de Archivos
uploaded_file = st.file_uploader(
    "Sube una imagen para extraer el texto",
    type=["png", "jpg", "jpeg"] #
)

# 3. Cargar y Ejecutar el Modelo OCR
if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir la imagen para EasyOCR
    # Necesitamos pasarla como bytes o como un array de numpy
    img_bytes = uploaded_file.getvalue()
    
    with st.spinner("Procesando imagen con OCR..."):
        # Cargar el modelo (lo tomará del caché si ya está cargado)
        reader = load_ocr_model()
        
        # 4. Procesar y Mostrar Resultados
        # Ejecutar el modelo OCR
        results = reader.readtext(img_bytes)
        
        # Juntar todo el texto extraído
        extracted_text = " ".join([res[1] for res in results])

        # Desafío de Persistencia:
        # Guardar el texto extraído en el st.session_state
        st.session_state['extracted_text'] = extracted_text
        
        # Mostrar el texto extraído
        st.text_area(
            "Texto Extraído por OCR:",
            extracted_text,
            height=250,
            key="ocr_output"
        )

# --- MÓDULOS 2 y 3: CONEXIÓN CON LLMS Y FLEXIBILIDAD ---

# Solo mostramos esta sección si ya hay texto extraído
if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
    
    st.divider()
    st.header("Módulos 2 y 3: Análisis con LLMs 🧠")
    
    # Texto extraído del estado de la sesión
    text_to_analyze = st.session_state['extracted_text']

    # --- Interfaz de Usuario (UI) ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Módulo 3: Elección de Proveedor
        provider = st.radio(
            "Elige el proveedor de LLM:",
            ("GROQ", "Hugging Face"),
            key="provider"
        )

        # Módulo 2: Elección de Tarea
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
        
        # Módulo 2: Elección de Modelo (solo para GROQ)
        if provider == "GROQ":
            
            # --- ACTUALIZACIÓN ---
            # Se actualiza la lista de modelos de Groq
            model_selection = st.selectbox(
                "Elige el modelo de GROQ:",
                (
                    "llama-3.1-8b-instant", 
                    "llama-3.1-70b-instant", 
                    "mixtral-8x7b-32768", 
                    "gemma-7b-it"
                ),
                key="groq_model"
            )
            # --- FIN DE LA ACTUALIZACIÓN ---
            
        else:
            # Módulo 3: Modelos de Hugging Face
            model_selection = st.text_input(
                "Modelo de Hugging Face (ej: mistralai/Mixtral-8x7B-Instruct-v0.1):",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                key="hf_model"
            )

    with col2:
        # Módulo 3: Control de Parámetros
        temperature = st.slider(
            "Temperatura (Creatividad)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature"
        )
        
        max_tokens = st.slider(
            "Máximos Tokens (Longitud)",
            min_value=50,
            max_value=4096,
            value=512,
            step=64,
            key="max_tokens"
        )

    # Módulo 2: Botón de Análisis
    analyze_button = st.button("Analizar Texto con LLM", type="primary")

    # --- Lógica de la API ---
    
    if analyze_button:
        with st.spinner(f"Analizando texto con {provider}... Por favor espera."):
            try:
                # Módulo 3: Lógica Condicional
                if provider == "GROQ":
                    # Módulo 2: Lógica de la API de GROQ
                    client = Groq(api_key=GROQ_API_KEY) #
                    
                    # Estructura correcta del prompt
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
                    
                    # Llamada a la API
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model_selection,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Mostrar la respuesta
                    response_content = chat_completion.choices[0].message.content
                    st.markdown("### Respuesta de GROQ")
                    st.markdown(response_content)

                elif provider == "Hugging Face":
                    # Módulo 3: Lógica de la API de Hugging Face
                    client = InferenceClient(token=HUGGINGFACE_API_KEY) #
                    
                    # Estructura del prompt para un modelo instruct
                    # (Usamos text_generation que es más flexible)
                    prompt = f"""<s>[INST] Eres un asistente experto. El usuario te dará un texto y una tarea.
Tarea: {task_prompt}
Texto:
{text_to_analyze}
[/INST]
Respuesta:"""
                    
                    # Llamada a la API
                    response = client.text_generation(
                        model=model_selection,
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=max(temperature, 0.01) # Temp 0 no es válida en HF, usamos 0.01
                    )
                    
                    # Mostrar la respuesta
                    st.markdown("### Respuesta de Hugging Face")
                    st.markdown(response)

            except Exception as e:
                st.error(f"Error al contactar la API de {provider}: {e}")
