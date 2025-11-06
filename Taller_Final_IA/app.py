import streamlit as st
from PIL import Image
import easyocr
from groq import Groq
from huggingface_hub import InferenceClient
import requests

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Taller IA: OCR + LLM", layout="wide")

# --- CARGA DE CLAVES DE API ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except KeyError:
    st.error("⚠️ No se encontraron las claves de API en los secretos de Streamlit.")
    st.stop()

# --- MÓDULO 1: OCR (LECTOR DE IMÁGENES) ---

@st.cache_resource
def load_ocr_model():
    """Carga el modelo EasyOCR en memoria (cacheado)."""
    reader = easyocr.Reader(['es', 'en'], gpu=False)
    return reader

st.title("🧠 Taller IA: Construcción de una Aplicación Multimodal")
st.header("Módulo 1: Lector de Imágenes (OCR) 📸")

uploaded_file = st.file_uploader(
    "Sube una imagen para extraer el texto",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Imagen subida", use_column_width=True)

    with st.spinner("Procesando imagen con OCR..."):
        reader = load_ocr_model()
        results = reader.readtext(uploaded_file.getvalue())
        extracted_text = " ".join([res[1] for res in results])
        st.session_state['extracted_text'] = extracted_text

        st.text_area("Texto extraído por OCR:", extracted_text, height=250, key="ocr_output")

# --- MÓDULO 2 Y 3: LLMs (GROQ Y HUGGING FACE) ---

if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
    st.divider()
    st.header("Módulos 2 y 3: Análisis con Modelos de Lenguaje 🧩")

    text_to_analyze = st.session_state['extracted_text']

    col1, col2 = st.columns(2)

    # --- OPCIONES DE ANÁLISIS ---
    with col1:
        provider = st.radio("Elige el proveedor de LLM:", ("GROQ", "Hugging Face"))

        task_prompt = st.selectbox(
            "Elige la tarea a realizar:",
            (
                "Resumir el texto en 3 puntos clave",
                "Identificar las entidades principales (personas, lugares, organizaciones)",
                "Traducir el texto al inglés",
                "Analizar el sentimiento del texto (positivo, negativo o neutral)",
                "Generar 3 preguntas sobre el texto"
            ),
        )

        if provider == "GROQ":
            st.info("Usando modelo GROQ: `llama-3.1-8b-instant`")
            model_selection = "llama-3.1-8b-instant"
        else:
            model_selection = st.text_input(
                "Modelo de Hugging Face:",
                "tiiuae/falcon-7b-instruct",  # ✅ modelo público y compatible
                key="hf_model"
            )

    with col2:
        temperature = st.slider("Temperatura (creatividad)", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Máximos tokens (longitud)", 50, 2048, 512, 64)

    # --- BOTÓN DE EJECUCIÓN ---
    if st.button("🚀 Analizar Texto con LLM", type="primary"):
        with st.spinner(f"Analizando texto con {provider}..."):
            try:
                # --- GROQ ---
                if provider == "GROQ":
                    client = Groq(api_key=GROQ_API_KEY)
                    messages = [
                        {"role": "system", "content": f"Eres un asistente experto. Realiza esta tarea: {task_prompt}."},
                        {"role": "user", "content": f"Texto a analizar:\n\n{text_to_analyze}"}
                    ]

                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model_selection,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    response = chat_completion.choices[0].message.content
                    st.success("✅ Análisis completado con GROQ")
                    st.markdown("### 🧩 Respuesta de GROQ")
                    st.markdown(response)

                # --- HUGGING FACE ---
                else:
                    api_url = f"https://router.huggingface.co/hf-inference/models/{model_selection}"
                    headers = {
                        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                        "Content-Type": "application/json",
                    }

                    hf_payload = {
                        "inputs": f"Eres un asistente experto. Realiza esta tarea: {task_prompt}.\n\nTexto para analizar:\n{text_to_analyze}",
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": max(temperature, 0.01),
                        },
                    }

                    response = requests.post(api_url, headers=headers, json=hf_payload)

                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            output = data[0].get("generated_text", "No se recibió texto.")
                        elif isinstance(data, dict):
                            output = data.get("generated_text", "No se recibió texto.")
                        else:
                            output = str(data)

                        st.success("✅ Análisis completado con Hugging Face")
                        st.markdown("### 🤖 Respuesta de Hugging Face")
                        st.markdown(output)

                    elif response.status_code == 404:
                        st.error("❌ Modelo no encontrado. Prueba con otro modelo público como `facebook/opt-1.3b` o `google/gemma-2b-it`.")
                    elif response.status_code == 401:
                        st.error("🔒 Error de autenticación. Verifica tu token de Hugging Face.")
                    else:
                        st.error(f"⚠️ Error de Hugging Face API: {response.status_code}\n\n{response.text}")

            except Exception as e:
                st.error(f"🚨 Error inesperado: {e}")
