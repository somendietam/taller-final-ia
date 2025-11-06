import streamlit as st
from PIL import Image
import easyocr
from groq import Groq
from huggingface_hub import InferenceClient
import requests

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
            model_selection = st.text_input(
                "Modelo de Hugging Face:",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",  # ✅ modelo compatible con hf-inference
                key="hf_model"
            )

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
            max_value=2048,
            value=512,
            step=64,
            key="max_tokens"
        )

    analyze_button = st.button("Analizar Texto con LLM", type="primary")

    # --- Lógica de la API ---

    if analyze_button:
        with st.spinner(f"Analizando texto con {provider}... Por favor espera."):
            try:
                # --- GROQ ---
                if provider == "GROQ":
                    client = Groq(api_key=GROQ_API_KEY)
                    messages = [
                        {"role": "system", "content": f"Eres un asistente experto. Realiza esta tarea: {task_prompt}."},
                        {"role": "user", "content": f"El texto para analizar es:\n\n---\n{text_to_analyze}\n---"}
                    ]
                    chat_completion = client.chat.completions.create(
                        messages=messages, model=model_selection,
                        temperature=temperature, max_tokens=max_tokens
                    )
                    response_content = chat_completion.choices[0].message.content
                    st.markdown("### Respuesta de GROQ")
                    st.markdown(response_content)

                # --- HUGGING FACE ---
                elif provider == "Hugging Face":
                    api_url = f"https://router.huggingface.co/hf-inference/models/{model_selection}"
                    headers = {
                        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                        "Content-Type": "application/json"
                    }

                    hf_payload = {
                        "inputs": f"""Eres un asistente experto. Realiza esta tarea: {task_prompt}.

Texto para analizar:
---
{text_to_analyze}
---""",
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": max(temperature, 0.01)
                        }
                    }

                    response = requests.post(api_url, headers=headers, json=hf_payload)

                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            response_text = data[0].get("generated_text", "No se recibió texto.")
                        elif isinstance(data, dict):
                            response_text = data.get("generated_text", "No se recibió texto.")
                        else:
                            response_text = str(data)

                        st.markdown("### Respuesta de Hugging Face")
                        st.markdown(response_text)
                    else:
                        st.error(f"Error de Hugging Face API: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Error al contactar la API de {provider}: {e}")
