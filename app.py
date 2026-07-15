import streamlit as st
import pandas as pd
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from google import genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar página
st.set_page_config(
    page_title="RAG Chat - arXiv Papers",
    page_icon="📚",
    layout="wide"
)

st.title("RAG Chat with arXiv Papers")
st.caption("Ask about scientific papers. The system retrieves relevant abstracts and generates answers using Gemini.")

# --- Cargar modelos y datos (con caché) ---
@st.cache_resource
def load_models():
    # Cargar modelo de embeddings
    model_emb = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Cargar índice FAISS (priorizar el ligero)
    if os.path.exists("arxiv_index_light.faiss"):
        index_file = "arxiv_index_light.faiss"
    else:
        index_file = "arxiv_index.faiss"
    index = faiss.read_index(index_file)
    
    # Cargar metadatos (priorizar el ligero)
    if os.path.exists("arxiv_metadata_light.csv"):
        metadata_file = "arxiv_metadata_light.csv"
    else:
        metadata_file = "arxiv_metadata.csv"
    df = pd.read_csv(metadata_file)
    
    # Configurar Gemini con la nueva API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("No se encontró GEMINI_API_KEY. Configura la variable de entorno.")
        st.stop()
    
    client = genai.Client(api_key=api_key)
    
    return model_emb, index, df, client

try:
    model_emb, index, df, client = load_models()
except Exception as e:
    st.error(f"Error al cargar modelos: {e}")
    st.stop()

# --- Funciones de búsqueda y generación ---
def search(query, k=5):
    """Busca documentos relevantes"""
    query_emb = model_emb.encode([query])
    distances, indices = index.search(query_emb.astype(np.float32), k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(df):
            results.append(df.iloc[idx])
    return results

def generate_response(query, docs):
    """Genera respuesta con Gemini usando la nueva API"""
    if not docs:
        return "No se encontraron documentos relevantes en el corpus."
    
    # Construir contexto
    context = "\n\n".join([
        f"Document {i+1}: {doc['titles']}\n{doc['summaries']}" 
        for i, doc in enumerate(docs)
    ])
    
    prompt = f"""You are an AI assistant for academic research. Answer the user's question based ONLY on the provided context from arXiv papers.

IMPORTANT RULES:
1. Use ONLY information from the context
2. If the context contains specific applications, list them clearly.
3. If the context mentions applications but does not list them explicitly, infer 1-2 plausible applications based on the technical content described, and clearly note that these are inferred rather than explicitly stated.
4. If the context doesn't contain enough information, say: "I don't have enough information in the corpus to answer this question."

### Context:
{context}

### User Question:
{query}

### Answer:
"""
    
    try:
        # Cambiado a gemini-2.0-flash (modelo estable y disponible)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error al generar respuesta: {e}"

# --- Interfaz de chat ---
# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "evidences" in msg and msg["evidences"]:
            with st.expander("Ver evidencias"):
                for i, doc in enumerate(msg["evidences"]):
                    st.write(f"**{i+1}. {doc['titles']}**")
                    st.write(doc['summaries'][:300] + "...")

# Input del usuario
if prompt := st.chat_input("Ask a question about arXiv papers..."):
    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Recuperar y generar
    with st.spinner("Searching and generating answer..."):
        docs = search(prompt, k=15)  
        response = generate_response(prompt, docs)
    
    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(response)
        if docs:
            with st.expander("Ver evidencias utilizadas"):
                for i, doc in enumerate(docs):
                    st.write(f"**{i+1}. {doc['titles']}**")
                    st.write(doc['summaries'][:300] + "...")
    
    # Guardar en historial
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "evidences": docs
    })