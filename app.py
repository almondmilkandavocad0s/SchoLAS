import streamlit as st
from answer import retrieve, build_prompt, generate
from pathlib import Path
from ingest import extract_chunks, CHROMA_DIR
import chromadb
from chromadb.utils import embedding_functions
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Initialize persistent Chroma client and collection
client = chromadb.PersistentClient(path=CHROMA_DIR)
embedder = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL, url=OLLAMA_URL)
collection = client.get_or_create_collection(name="docs", embedding_function=embedder)


st.title("SchoLAS")
st.text("Scholarly Learning Analytics Synthesizer")

@st.cache_data(show_spinner=False)
def cached_retrieve(query):
    return retrieve(query)

@st.cache_data(show_spinner=False)
def cached_generate(prompt):
    return generate(prompt)

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = ""
if "hits" not in st.session_state:
    st.session_state.hits = []
if "answer" not in st.session_state:
    st.session_state.answer = ""

st.session_state.query = st.text_area("Ask me anything realated to LA Research!")

if st.button("Generate Response"): 
    if st.session_state.query.strip():
        with st.spinner("Retrieving relevant documents.."):
            st.session_state.hits = cached_retrieve(st.session_state.query)
        if not st.session_state.hits:
            st.warning("No relevant documents found.")
            st.session_state.answer = ""
        else:
            prompt = build_prompt(st.session_state.query, st.session_state.hits)
            with st.spinner("Generating response..."):
                st.session_state.answer = cached_generate(prompt)
    else:
        st.warning("Please enter a question.")
st.session_state.query = ""
    
if st.session_state.answer:
    st.subheader("Response: ")
    st.write(st.session_state.answer)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key="upload_pdf")

if uploaded_file is not None:
    upload_dir = Path("uploaded_pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True) 
    safe_name = uploaded_file.name.replace(" ", "_")
    temp_path = upload_dir / safe_name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {uploaded_file.name} successfully!")
    if st.button("Ingest PDF"):
        chunks = list(extract_chunks(temp_path))

        if not chunks:
            st.warning("No text found in the PDF.")
        else:
            docs = [c[0] for c in chunks]
            metas = [c[1] for c in chunks]
            ids = [str(uuid.uuid4()) for _ in chunks]

            BATCH_SIZE = 8
            for j in range(0, len(docs), BATCH_SIZE):
                collection.add(
                    documents=docs[j:j+BATCH_SIZE],
                    metadatas=metas[j:j+BATCH_SIZE],
                    ids=ids[j:j+BATCH_SIZE],
                )
            st.success(f"Ingested {len(docs)} chunks from {uploaded_file.name}!")

        # Optional: remove the temporary file after ingestion
        temp_path.unlink()


# Display top matches if available
if st.session_state.hits:
    st.subheader("Top Matches")
    for i, (_, meta, score) in enumerate(st.session_state.hits, 1):
        st.write(f"[{i}] {meta['source_file']} p.{meta['page_number']} — score={score:.3f}")




# Cached questions: 
# What are the implications of GenAI in Teaching and Learning?

