
# Enterprise AI Copilot

### Local AI-Powered Document Assistant using RAG

Enterprise AI Copilot is an intelligent document assistant that enables users to interact with PDF documents using natural language.

The application combines Large Language Models (LLMs), semantic search, vector databases, and Retrieval-Augmented Generation (RAG) to generate context-aware answers supported by document references.

## Key Features

- Interactive AI chatbot built with Streamlit
- Local LLM inference using Qwen 2.5 via Ollama
- Semantic search with Sentence Transformers
- Vector storage and retrieval using ChromaDB
- Multiple PDF document support
- PDF upload, indexing, and deletion
- Document-specific question answering
- Source references with page numbers and excerpts
- Conversation history during the active session

## Tech Stack

- Python
- Streamlit
- Ollama / Qwen 2.5:3b
- Sentence Transformers
- ChromaDB
- PyMuPDF
- Retrieval-Augmented Generation (RAG)

## RAG Architecture

1. PDF documents are uploaded by the user.
2. Text is extracted using PyMuPDF.
3. Text is divided into overlapping chunks.
4. Sentence Transformers generates vector embeddings.
5. Embeddings are stored in ChromaDB.
6. A user submits a question through Streamlit.
7. Relevant document chunks are retrieved.
8. Qwen 2.5 generates an answer using the retrieved context.
9. The application displays the answer and its source reference.

## Project Structure

```text
Enterprise_AI_Copilot/
├── app.py
├── src/
│   ├── __init__.py
│   ├── rag.py
│   └── ingestion.py
├── data/
│   └── pdf/
├── 01_Introduction_LLM.ipynb
├── 02_Embeddings.ipynb
├── 03_Premier_RAG.ipynb
├── 04_Import_Multiple_PDF.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Enterprise_AI_Copilot.git
cd Enterprise_AI_Copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Ollama

Download Ollama from https://ollama.com

Pull the Qwen model:

```bash
ollama pull qwen2.5:3b
```

### 6. Run the application

```bash
streamlit run app.py
```

## Example Use Case

**Question:**
How many remote working days are authorized per week?

**Answer:**
Remote work is authorized for up to two days per week for eligible positions.

**Source:**
HR Regulation — Page 2

## Future Improvements

- Advanced document retrieval
- RAG evaluation and benchmarking
- Improved user interface
- Performance optimization
- Deployment and containerization

## Author

**Majda Chouiekh**

Artificial Intelligence & Data Science
