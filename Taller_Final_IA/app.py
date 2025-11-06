import streamlit as st
import os
import easyocr
from PIL import Image
from groq import Groq
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- 1. CONFIGURACIÓN Y CARGA DE MODELOS ---

# Cargar las claves de API desde el archivo .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# (Módulo 1 - Desafío) Cargar el modelo OCR usando el caché de Streamlit
# Esto asegura que el modelo (que es pesado) se cargue solo una vez.
@st.cache_resource
def load_ocr_model():
    """Carga el modelo EasyOCR en caché."""
    print("Cargando modelo EasyOCR...")
    reader = easyocr.Reader(['es', 'en']) # Puedes añadir más idiomas
    print("Modelo EasyOCR cargado.")
    return reader

# Cargar el modelo al inicio de la app
reader = load_ocr_model()

# --- 2. CONFIGURACIÓN DE LA INTERFAZ (UI) ---

# Configuración de la página (Módulo 1)
st.set_page_config(page_title="Taller IA: OCR + LLM", layout="wide")
st.title("Proyecto Final: Aplicación Multimodal con OCR y LLMs")

# Inicializar el estado de sesión (Módulo 2 - Desafío)
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# --- 3. MÓDULO 1: LECTOR DE IMÁGENES (OCR) ---

st.header("Módulo 1: Extracción de Texto (OCR) con EasyOCR")

# Widget para subir archivos (Módulo 1)
uploaded_file = st.file_uploader(
    "Sube una imagen para extraer el texto", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Mostrar la imagen subida (Módulo 1)
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Procesar la imagen y extraer texto
    with st.spinner("Extrayendo texto de la imagen..."):
        # Convertir la imagen a bytes, formato que easyocr puede leer
        image_bytes = uploaded_file.getvalue()
        
        # Ejecutar el modelo OCR (Módulo 1)
        result = reader.readtext(image_bytes)
        
        # Combinar los fragmentos de texto extraídos
        extracted_text = " ".join([text[1] for text in result])
        
        # (Módulo 2 - Desafío) Guardar el texto en el estado de sesión
        st.session_state.ocr_text = extracted_text

# Mostrar el texto extraído en un área de texto (Módulo 1)
# Esto se muestra incluso si se sube una nueva imagen, gracias al session_state
st.text_area(
    "Texto Extraído:", 
    value=st.session_state.ocr_text, 
    height=250, 
    key="ocr_output"
)

# --- 4. MÓDULO 2 & 3: PROCESAMIENTO CON LLMs ---

st.header("Módulo 2 y 3: Análisis de Texto con LLMs")

# Solo mostrar esta sección si hay texto para analizar
if st.session_state.ocr_text:
    
    # --- UI de Controles (Módulos 2 y 3) ---
    
    col1, col2 = st.columns(2)
    
    with col1:
        # (Módulo 3) Selección de Proveedor (Groq vs HF)
        provider = st.radio(
            "Elige el proveedor del LLM:",
            ("GROQ", "Hugging Face"),
            horizontal=True
        )

        # (Módulo 2) Tarea a realizar
        task = st.selectbox(
            "Elige la tarea a realizar:",
            (
                "Resumir el texto en 3 puntos clave",
                "Identificar las entidades principales (Personas, Lugares, Org.)",
                "Traducir el texto al inglés",
                "Analizar el sentimiento (Positivo, Negativo, Neutral)",
                "Generar 3 preguntas sobre el texto"
            )
        )

    with col2:
        # (Módulo 3) Controles de Parámetros
        temperature = st.slider(
            "Creatividad (Temperature):", 
            min_value=0.0, max_value=1.0, value=0.3, step=0.1
        )
        max_tokens = st.slider(
            "Longitud Máx. de Respuesta (Max Tokens):", 
            min_value=50, max_value=1024, value=200, step=50
        )

    # (Módulo 2) Botón para iniciar el análisis
    if st.button("Analizar Texto con LLM"):
        
        with st.spinner(f"Contactando a {provider} para analizar el texto..."):
            
            # --- Lógica Condicional del Proveedor (Módulo 3) ---
            
            # --- Opción 1: GROQ ---
            if provider == "GROQ":
                try:
                    # Instanciar el cliente de Groq (Módulo 2)
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # (Módulo 2) Construcción del Prompt
                    system_prompt = f"Eres un asistente experto. Tu tarea es la siguiente: {task}."
                    user_prompt = f"El texto que debes analizar es el siguiente:\n\n{st.session_state.ocr_text}"
                    
                    # (Módulo 2) Llamada a la API
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model="llama3-8b-8192", # Modelo de Groq
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # (Módulo 2) Mostrar respuesta
                    response = chat_completion.choices[0].message.content
                    st.markdown("### Respuesta de GROQ:")
                    st.markdown(response)

                except Exception as e:
                    st.error(f"Error al contactar la API de Groq: {e}")

            # --- Opción 2: HUGGING FACE ---
            elif provider == "Hugging Face":
                try:
                    # (Módulo 3) Instanciar el cliente de HF
                    client_hf = InferenceClient(token=HUGGINGFACE_API_KEY)
                    
                    # (Módulo 3) Implementación de Tareas (Adaptado)
                    # Nota: Las APIs de Inferencia de HF son más rígidas que Groq.
                    # Adaptamos la tarea a los 'pipelines' disponibles.
                    
                    if "Resumir" in task:
                        response = client_hf.summarization(
                            st.session_state.ocr_text,
                            parameters={"max_length": max_tokens}
                        )
                        response_text = response.summary_text
                    
                    elif "Traducir" in task:
                        response = client_hf.translation(
                            st.session_state.ocr_text, 
                            model="Helsinki-NLP/opus-mt-es-en" # Modelo específico para Es->En
                        )
                        response_text = response.translation_text
                    
                    elif "Analizar el sentimiento" in task:
                        response = client_hf.text_classification(st.session_state.ocr_text)
                        response_text = f"Sentimiento Detectado: {response[0]['label']} (Confianza: {response[0]['score']:.2%})"
                    
                    else:
                        # Tareas más complejas como Q&A o Entidades son menos directas
                        # Usamos un modelo de generación de texto como fallback
                        prompt = f"Tarea: {task}\n\nTexto: {st.session_state.ocr_text}\n\nRespuesta:"
                        response = client_hf.text_generation(
                            prompt, 
                            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            max_new_tokens=max_tokens,
                            temperature=temperature if temperature > 0 else 0.1 # temp 0 da error en HF
                        )
                        response_text = response
                    
                    # (Módulo 3) Mostrar respuesta
                    st.markdown("### Respuesta de Hugging Face:")
                    st.markdown(response_text)

                except Exception as e:
                    st.error(f"Error al contactar la API de Hugging Face: {e}")

else:
    st.warning("Por favor, sube una imagen para activar el análisis con LLM.")