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
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except KeyError:
    st.error("No se encontraron las claves de API en los secretos de Streamlit. Asegúrate de haberlas configurado.")
    st.stop()

# --- MÓDULO 1: EL LECTOR DE IMÁGENES (OCR) ---

@st.cache_resource
def load_ocr_model():
    """Carga el modelo EasyOCR en memoria (cacheado)."""
    reader = easyocr.Reader(['es', 'en'], gpu=False) 
    return reader

st.title("Taller IA: Construcción de una Aplicación Multimodal")
st.header("Módulo 1: Lector de Imágenes (OCR) 📸")

uploaded_file = st.file_uploader(
    "Sube una imagen para extraer el texto",
    type=["png", "jpg", "jpeg"]
)

# Cargar y ejecutar el modelo OCR si se sube un archivo
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    img_bytes = uploaded_file.getvalue()
    
    with st.spinner("Procesando imagen con OCR..."):
        reader = load_ocr_model()
        results = reader.readtext(img_bytes)
        extracted_text = " ".join([res[1] for res in results])
        
        st.session_state['extracted_text'] = extracted_text
        
        st.text_area(
            "Texto Extraído por OCR:",
            extracted_text,
            height=250,
            key="ocr_output"
        )

# --- MÓDULOS 2 y 3: CONEXIÓN CON LLMS Y FLEXIBILIDAD ---

if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
    
    st.divider()
    st.header("Módulos 2 y 3: Análisis con LLMs 🧠")
    
    text_to_analyze = st.session_state['extracted_text']

    # --- Interfaz de Usuario (UI) ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.radio(
            "Elige el proveedor de LLM:",
            ("GROQ", "Hugging Face"),
            key="provider"
        )

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
        
        if provider == "GROQ":
            st.info("Usando el modelo: `llama-3.1-8b-instant`")
            model_selection = "llama-3.1-8b-instant"
            
        else:
            # --- CAMBIO DE MODELO ---
            # Cambiamos Mixtral por Mistral-7B-Instruct
            # Este modelo SÍ soporta 'text_generation'
            model_selection = st.text_input(
                "Modelo de Hugging Face:",
                "mistralai/Mistral-7B-Instruct-v0.2",
                key="hf_model",
                help="Este modelo debe soportar la tarea 'text_generation'."
            )
            # --- FIN DEL CAMBIO ---

    with col2:
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

    analyze_button = st.button("Analizar Texto con LLM", type="primary")

    # --- Lógica de la API (con text_generation para HF) ---
    
    if analyze_button:
        with st.spinner(f"Analizando texto con {provider}... Por favor espera."):
            try:
                if provider == "GROQ":
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    messages = [
                        {
                            "role": "system",
                            "content": f"Eres un asistente experto. Realiza esta tarea: {task_prompt}."
                        },
                        {
                            "role": "user",
                            "content": f"El texto para analizar es:\n\n---\n{text_to_analyze}\n---"
                        }
                    ]
                    
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model_selection,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    response_content = chat_completion.choices[0].message.content
                    st.markdown("### Respuesta de GROQ")
                    st.markdown(response_content)

                elif provider == "Hugging Face":
                    # Usamos 'text_generation' (confirmado por el cambio de modelo)
                    client = InferenceClient(token=HUGGINGFACE_API_KEY)
                    
                    # Formato de prompt para Mistral-7B-Instruct (similar a Mixtral)
                    hf_prompt = f"""[INST] Eres un asistente experto. El usuario te dará un texto y una tarea.
Tarea: {task_prompt}

Texto para analizar:
---
{text_to_analyze}
---
[/INST]
"""
                    
                    # Llamada a la API con .text_generation()
                    response_content = client.text_generation(
                        model=model_selection, # Usará "mistralai/Mistral-7B-Instruct-v0.2"
                        prompt=hf_prompt,
                        max_new_tokens=max_tokens, 
                        temperature=max(temperature, 0.01)
                    )
                    
                    st.markdown("### Respuesta de Hugging Face")
                    st.markdown(response_content)

            except Exception as e:
                st.error(f"Error al contactar la API de {provider}: {e}")
