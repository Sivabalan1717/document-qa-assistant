# Document Q&A Assistant using RAG

A local web application that lets you upload a PDF or TXT document and ask questions about its contents. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant text chunks before asking a local language model to generate an answer.

## Features

- Upload PDF and TXT documents through a Flask web interface.
- Extract text from supported documents and split it into overlapping chunks.
- Create semantic embeddings with Sentence Transformers.
- Retrieve relevant chunks with FAISS similarity search.
- Generate answers locally with Ollama and Qwen 2.5.
- Return a fallback response when the answer is not found in the retrieved context.
- Validate file types, upload size, and PDF page count.

## Tech stack

- Python 3.12
- Flask
- PyPDF
- LangChain text splitters
- Sentence Transformers — `all-MiniLM-L6-v2`
- FAISS
- Ollama — `qwen2.5:3b`
- HTML and CSS

## How the RAG pipeline works

1. **Upload:** The user submits a PDF or TXT file and a question.
2. **Text extraction:** Text is extracted from the document.
3. **Chunking:** The text is split into chunks of 800 characters with 150 characters of overlap.
4. **Embedding:** Sentence Transformers converts chunks and the question into vectors.
5. **Retrieval:** FAISS finds up to three chunks most similar to the question.
6. **Generation:** The retrieved context and question are sent to the local Ollama model.
7. **Answer:** The generated response is shown in the web interface.

```text
PDF / TXT + Question
        |
        v
Text Extraction
        |
        v
Chunking
        |
        v
Sentence Transformer Embeddings
        |
        v
FAISS Similarity Search
        |
        v
Relevant Context + Question
        |
        v
Ollama (Qwen 2.5)
        |
        v
Answer in Flask Web UI
```

## Requirements

- Windows, macOS, or Linux
- Python 3.12 recommended
- Ollama installed and running locally
- The Ollama model `qwen2.5:3b`

The application is designed to run locally. The language model is not hosted by this repository.

## Setup on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/Sivabalan1717/document-qa-assistant.git
cd document-qa-assistant
```

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, run the virtual environment's Python executable directly as shown below.

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com), then download the model:

```powershell
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running. The app sends requests to:

```text
http://localhost:11434/api/chat
```

### 5. Run the web application

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Upload a document and enter a question about its content.

## Project structure

```text
document-qa-assistant/
├── app.py
├── rag_pipeline.py
├── build_index.py
├── search_document.py
├── requirements.txt
├── .gitignore
├── templates/
│   └── index.html
└── static/
    └── style.css
```

- `app.py` — Flask web application and end-to-end RAG flow.
- `rag_pipeline.py` — document text extraction and chunking.
- `build_index.py` — standalone FAISS index-building example.
- `search_document.py` — terminal-based document question-answering example.
- `templates/index.html` and `static/style.css` — web interface.

## Current limits

- Maximum upload size: 10 MB.
- Maximum PDF length: 40 pages.
- Supported file types: PDF and TXT.
- Scanned PDFs without an extractable text layer are not processed; OCR is not included.
- Answers depend on the retrieved context and the local model, so they should be checked for accuracy.

## Privacy and safety

Documents are processed by the local application and sent to the locally running Ollama service for answer generation. Do not commit personal documents, private data, model files, virtual environments, or credentials to GitHub.

## Future improvements

- Reuse the embedding model between requests.
- Cache document embeddings and the FAISS index for repeated questions.
- Display retrieved source chunks with answers.
- Add automated tests and OCR support for scanned PDFs.

## License

No license has been added yet. Add a license file if you want to specify how others may use, modify, and distribute this project.
