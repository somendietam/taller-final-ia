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
    # Usamos 'es' (español) e 'en' (inglés)
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
    
    # Convertir la imagen a bytes para EasyOCR
    img_bytes = uploaded_file.getvalue()
    
    with st.spinner("Procesando imagen con OCR..."):
        reader = load_ocr_model()
        # Ejecutar OCR
        results = reader.readtext(img_bytes)
        # Juntar el texto detectado
        extracted_text = " ".join([res[1] for res in results])
        
        # Guardar en el estado de la sesión (Desafío de Persistencia)
        st.session_state['extracted_text'] = extracted_text
        
        # Mostrar el texto extraído
        st.text_area(
            "Texto Extraído por OCR:",
            extracted_text,
            height=250,
            key="ocr_output"
        )

# --- MÓDULOS 2 y 3: CONEXIÓN CON LLMS Y FLEXIBILIDAD ---

# Solo mostrar esta sección si hay texto en el estado de la sesión
if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
    
    st.divider()
    st.header("Módulos 2 y 3: Análisis con LLMs 🧠")
    
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
        
        if provider == "GROQ":
            # --- CORRECCIÓN 1: Se elimina el selectbox de GROQ ---
            st.info("Usando el modelo: `llama-3.1-8b-instant`")
            # Se asigna el modelo directamente
            model_selection = "llama-3.1-8b-instant"
            # --- FIN DE LA CORRECCIÓN 1 ---
            
        else:
            # Módulo 3: Modelo de Hugging Face
            model_selection = st.text_input(
                "Modelo de Hugging Face:",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                key="hf_model",
                help="Asegúrate que el modelo soporte la tarea 'chat_completion'."
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
            
            # Definir los mensajes (común para ambos proveedores)
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
            
            try:
                if provider == "GROQ":
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model_selection, # Usará "llama-3.1-8b-instant"
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    response_content = chat_completion.choices[0].message.content
                    st.markdown("### Respuesta de GROQ")
                    st.markdown(response_content)

                elif provider == "Hugging Face":
                    # --- CORRECCIÓN 2: Usamos client.chat_completion ---
                    client = InferenceClient(token=HUGGINGFACE_API_KEY)
                    
                    response = client.chat_completion(
                        messages=messages,
                        model=model_selection,
                        # HF usa 'max_new_tokens' en algunos endpoints, 
                        # pero 'chat_completion' usa 'max_tokens'
                        max_tokens=max_tokens, 
                        temperature=max(temperature, 0.01) # Temp 0.0 puede fallar
                    )
                    
                    # La respuesta tiene la misma estructura que la de Groq
                    response_content = response.choices[0].message.content
                    
                    st.markdown("### Respuesta de Hugging Face")
                    st.markdown(response_content)
                    # --- FIN DE LA CORRECCIÓN 2 ---

            except Exception as e:
                st.error(f"Error al contactar la API de {provider}: {e}")
