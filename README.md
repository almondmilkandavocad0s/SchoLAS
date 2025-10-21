# SchoLAS 
_Scholarly Learning Analytics Synthesizer_

![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)

**SchoLAS** is an experimental tool designed to create a queryable knowledge base from the latest research in learning analytics. The primary goal is to use a local Large Language Model (LLM) to answer questions based on the full-text content of recent publications.

This project is in the very early stages of development. The core logic and pipeline are currently being prototyped in a Jupyter Notebook. This README outlines the planned architecture and goals.

---

## Planned Architecture

The tool is being built as a three-stage pipeline:

### 1. Data Ingestion (Scraping)

* **Source:** We will use [SoLAR's Journal of Learning Analytics (JLA)](https://learning-analytics.info/) as our initial data source, given its high relevance and quality.
* **Method:** A Python script using `BeautifulSoup` will be developed to scrape the table of contents from recent issues. This scraper will gather metadata (titles, authors) and, most importantly, the URLs to the full-text paper (PDFs or HTML).

### 2. RAG Pipeline (The "Brain")

This is the core of the project, which will be built using a Retrieval-Augmented Generation (RAG) pipeline:

* **Load & Chunk:** Downloaded papers will be loaded and processed, splitting their text content into smaller, semantically meaningful chunks.
* **Embed & Store:** Using an embedding model, each text chunk will be converted into a vector representation and stored in a local vector database (e.g., ChromaDB, FAISS).
* **Retrieve & Augment:** When a user asks a question, the system will:
    1.  Embed the user's query into a vector.
    2.  Search the vector database to retrieve the most relevant text chunks from the papers.
    3.  Augment a **local LLM** (e.g., Llama 3, Mistral) by feeding these chunks as context along with the user's original question.
* **Generate:** The LLM will then synthesize the provided context to generate a precise, source-based answer.

### 3. Front-End (Interface)

* **Tool:** A simple, user-friendly web interface will be built using `Streamlit`.
* **Function:** This will provide a chat-like text box where users can ask questions (e.g., "What are the latest findings on student engagement in JLA?") and receive the LLM's generated answer.

---

## Core Technologies

* **Python 3.x**
* **Web Scraping:** `BeautifulSoup4`, `Requests`
* **RAG & LLM:** `LangChain` / `LlamaIndex` (or similar)
* **LLM:** A local, open-source model (e.g., Llama 3, Mistral)
* **Vector Store:** `ChromaDB` / `FAISS`
* **Web UI:** `Streamlit`
* **Prototyping:** `Jupyter Notebook`

---

## Preliminary Roadmap

-   [ ] **Scraper:** Develop a stable scraper for the JLA website to fetch paper URLs and metadata.
-   [ ] **RAG Proof-of-Concept:** Build the end-to-end RAG pipeline in the Jupyter Notebook (Load, Chunk, Embed, Retrieve, Generate).
-   [ ] **Model Selection:** Test and select a suitable local LLM that balances performance with resource requirements.
-   [ ] **Streamlit App:** Port the notebook logic into a functional `streamlit` application.
-   [ ] **Refinement:** Improve scraping stability, chunking strategy, and overall answer quality.