import streamlit as st
from dotenv import load_dotenv
import os
import easyocr
from groq import Groq
from huggingface_hub import InferenceClient
from PIL import Image
import numpy as np

# --- 0. CONFIGURACIÓN INICIAL Y CARGA DE CLAVES ---
# Cargar las variables de entorno (claves API) desde el archivo .env [cite: 122, 123]
load_dotenv()

# Instanciar clientes de API
# Cliente para GROQ [cite: 130]
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Cliente para Hugging Face [cite: 148]
hf_client = InferenceClient(
    token=os.environ.get("HUGGINGFACE_API_KEY")
)

# --- 1. MÓDULO 1: EL LECTOR DE IMÁGENES (OCR) --- [cite: 100]

# Título de la aplicación [cite: 106]
st.title("Taller IA: Aplicación Multimodal (OCR + LLM) 🤖📄")

# Encabezado para la sección de OCR [cite: 107]
st.header("Módulo 1: Extracción de Texto con EasyOCR")

# Widget para subir archivos de imagen [cite: 109, 110]
uploaded_file = st.file_uploader("Sube una imagen (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

# Desafío de Caché: Cargar el modelo OCR solo una vez 
# Usamos @st.cache_resource para que el modelo persista en memoria
@st.cache_resource
def load_ocr_model():
    """Carga el modelo EasyOCR en memoria."""
    reader = easyocr.Reader(['es', 'en'])  # Puedes añadir más idiomas
    return reader

reader = load_ocr_model()

# Inicializar el estado de sesión para guardar el texto (Desafío Módulo 2) 
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

if uploaded_file is not None:
    # Procesar y mostrar la imagen [cite: 115]
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Convertir la imagen a un formato que EasyOCR pueda procesar (bytes) [cite: 116]
    img_bytes = uploaded_file.getvalue()
    
    st.write("Extrayendo texto...")
    
    # Ejecutar el modelo OCR [cite: 117]
    # .readtext devuelve una lista de (bounding_box, texto, confianza)
    try:
        results = reader.readtext(img_bytes)
        
        # Unir todo el texto detectado
        extracted_text = " ".join([result[1] for result in results])
        
        # Guardar en el estado de sesión para persistencia 
        st.session_state.extracted_text = extracted_text
        
        # Mostrar el texto extraído en un área de texto [cite: 118]
        st.text_area("Texto Extraído:", st.session_state.extracted_text, height=250)
    
    except Exception as e:
        st.error(f"Error al procesar la imagen con EasyOCR: {e}")

# --- 2. MÓDULO 2 Y 3: CONEXIÓN CON LLMS (GROQ Y HUGGING FACE) ---

st.header("Módulo 2 & 3: Análisis con LLMs")

# Verificar si hay texto para analizar
if not st.session_state.extracted_text:
    st.warning("Por favor, sube una imagen primero para extraer texto.")
else:
    # Mostrar el texto que se va a analizar
    st.write("Texto a analizar:")
    st.text_area("Texto extraído (editable):", key="extracted_text", height=150)

    # --- Controles de la Interfaz (Módulo 2 y 3) ---

    # Módulo 3: Selección de Proveedor (GROQ vs Hugging Face) 
    provider = st.radio(
        "Elige el proveedor del LLM:",
        ("GROQ", "Hugging Face"),
        key="provider"
    )

    # Módulo 2: Selección de Tarea [cite: 127]
    task = st.selectbox(
        "Elige la tarea a realizar:",
        ("Resumir en 3 puntos clave", 
         "Identificar las entidades principales", 
         "Traducir al inglés",
         "Analizar el sentimiento del texto"),
        key="task"
    )

    # Módulo 3: Sliders de Parámetros 
    st.subheader("Parámetros del Modelo")
    temperature = st.slider(
        "Temperatura (Creatividad):", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        key="temperature"
    )
    
    max_tokens = st.slider(
        "Máximos Tokens (Longitud):", 
        min_value=50, 
        max_value=500, 
        value=150, 
        step=50,
        key="max_tokens"
    )

    # Módulo 2: Botón de Análisis [cite: 128]
    if st.button("Analizar Texto", key="analyze_button"):
        
        # Obtener el texto actual (podría haber sido editado por el usuario)
        text_to_analyze = st.session_state.extracted_text

        # --- Lógica Condicional (Módulo 3) --- 
        
        with st.spinner("El LLM está pensando... 🧠"):
            try:
                # --- Opción 1: GROQ ---
                if provider == "GROQ":
                    st.write("Usando GROQ API...")

                    # Módulo 2: Selección de Modelo (GROQ) [cite: 126]
                    # Simplificado a un modelo para este ejemplo, pero podrías usar un selectbox
                    model = "llama3-8b-8192" 
                    
                    # Módulo 2: Construcción del Prompt [cite: 132]
                    system_prompt = f"Eres un asistente experto. El usuario te dará un texto y tú debes realizar la siguiente tarea: '{task}'."
                    user_prompt = f"El texto es el siguiente:\n\n{text_to_analyze}"

                    # Módulo 2: Llamada a la API de GROQ [cite: 131]
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    response = chat_completion.choices[0].message.content
                    st.markdown(response) # Módulo 2: Mostrar resultado [cite: 133]

                # --- Opción 2: HUGGING FACE ---
                elif provider == "Hugging Face":
                    st.write("Usando Hugging Face API...")
                    
                    # Módulo 3: Lógica de Hugging Face [cite: 146]
                    # Nota: La API de Inferencia de HF es mejor para tareas específicas.
                    # Adaptamos la "tarea" a los tipos de pipeline de HF. [cite: 149]
                    
                    if task == "Resumir en 3 puntos clave":
                        hf_task_pipeline = "summarization"
                        # El prompt debe ser formateado para el modelo específico (ej. BART)
                        prompt = f"Summarize the following text in 3 key points:\n\n{text_to_analyze}"
                        model = "facebook/bart-large-cnn"
                        
                        response = hf_client.text_generation(prompt, model=model, max_new_tokens=max_tokens, temperature=temperature if temperature > 0 else 0.1)
                        # La respuesta de hf_client.text_generation es una cadena
                        st.markdown(response)
                    
                    elif task == "Traducir al inglés":
                        hf_task_pipeline = "translation"
                        model = "Helsinki-NLP/opus-mt-es-en" # Modelo específico de ES a EN
                        
                        # El cliente de inferencia sabe cómo manejar este pipeline
                        response = hf_client.translation(text_to_analyze, model=model)
                        # La respuesta es un diccionario, ej: [{'translation_text': '...'}]
                        st.markdown(response[0]['translation_text'])
                    
                    else:
                        st.warning(f"La tarea '{task}' no está implementada de forma simple para Hugging Face en este demo. Prueba con 'Resumir' o 'Traducir'.")

            except Exception as e:
                st.error(f"Error al contactar la API del LLM: {e}")
