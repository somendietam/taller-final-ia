import streamlit as st
from PIL import Image
import easyocr
import os
import numpy as np
from groq import Groq
from huggingface_hub import InferenceClient

# --- CONFIGURACIÓN DE LA PÁGINA Y CLAVES ---

st.set_page_config(page_title="Taller IA: OCR + LLM", layout="wide")

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
st.header("Módulo 1: Lector de Imágenes (OCR)")

uploaded_file = st.file_uploader(
    "Sube una imagen para extraer el texto",
    type=["png", "jpg", "jpeg"]
)

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
    st.header("Módulos 2 y 3: Análisis con LLMs")
    
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
                "Traducir el texto al inglés",
                "Analizar el sentimiento del texto (positivo, negativo o neutral)",
                "Identificar las entidades principales (personas, lugares, organizaciones)",
                "Generar 3 preguntas sobre el texto"
            ),
            index=0, # Empezar con "Resumir"
            key="task"
        )
        
        if provider == "GROQ":
            st.info("Usando el modelo: `llama-3.1-8b-instant`")
            model_selection = "llama-3.1-8b-instant"
        else:
            st.info("Usando endpoints de tareas específicas de Hugging Face.")
            model_selection = None

    with col2:
        temperature = st.slider(
            "Temperatura (Creatividad)", 0.0, 1.0, 0.7, 0.1,
            key="temperature",
            help="Nota: El control de Temperatura solo aplica para GROQ."
        )
        
        max_tokens = st.slider(
            "Máximos Tokens (Longitud)", 50, 4096, 512, 64,
            key="max_tokens",
            help="Nota: Este slider solo aplica para GROQ."
        )

    analyze_button = st.button("Analizar Texto con LLM", type="primary")

    # --- Lógica de la API ---
    
    if analyze_button:
        with st.spinner(f"Analizando texto con {provider}... Por favor espera."):
            try:
                response_content = "" # Inicializar la variable
                
                if provider == "GROQ":
                    client = Groq(api_key=GROQ_API_KEY)
                    messages = [
                        { "role": "system", "content": f"Eres un asistente experto. Realiza esta tarea: {task_prompt}." },
                        { "role": "user", "content": f"El texto para analizar es:\n\n---\n{text_to_analyze}\n---" }
                    ]
                    chat_completion = client.chat.completions.create(
                        messages=messages, model="llama-3.1-8b-instant",
                        temperature=temperature, max_tokens=max_tokens
                    )
                    response_content = chat_completion.choices[0].message.content

                elif provider == "Hugging Face":
                    client = InferenceClient(token=HUGGINGFACE_API_KEY)
                    
                    # --- CORRECCIÓN: Truncar el texto ---
                    # Los modelos de HF tienen límites de tokens. 3000 caracteres es un límite seguro.
                    safe_text_to_analyze = text_to_analyze[:3000]
                    # --- FIN DE LA CORRECCIÓN ---

                    if "Resumir" in task_prompt:
                        response_list = client.summarization(
                            text=safe_text_to_analyze, # Usar el texto truncado
                            model="facebook/bart-large-cnn"
                        )
                        response_content = response_list[0]['summary_text']
                    
                    elif "Traducir" in task_prompt:
                        response_list = client.translation(
                            text=safe_text_to_analyze, # Usar el texto truncado
                            model="Helsinki-NLP/opus-mt-es-en"
                        )
                        response_content = response_list[0]['translation_text']

                    elif "sentimiento" in task_prompt:
                        response_list = client.sentiment_analysis(
                            text=safe_text_to_analyze, # Usar el texto truncado
                            model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
                        )
                        best_sentiment = max(response_list[0], key=lambda x: x['score'])
                        response_content = f"Sentimiento Detectado: **{best_sentiment['label']}** (Confianza: {best_sentiment['score']:.2f})"
                    
                    else:
                        st.error(f"La tarea '{task_prompt}' no tiene un endpoint de tarea simple en Hugging Face. \n\nPor favor, prueba 'Resumir', 'Traducir al inglés' o 'Analizar sentimiento' para el proveedor Hugging Face.")
                        st.stop()
                
                # Mostrar la respuesta
                st.markdown(f"### Respuesta de {provider}")
                st.markdown(response_content)

            except Exception as e:
                st.error(f"Error al contactar la API de {provider}: {e}")


