# 🧠 RAG Document QA System



This project demonstrates a complete Retrieval Augmented Generation (RAG) pipeline for document based question answering using local large language models (LLMs). It integrates document ingestion, text chunking, vector embedding, and semantic retrieval with FAISS for efficient similarity search. The retrieved context is dynamically combined with user queries and passed to a local LLM via Ollama, enabling accurate, context aware answers without relying on external APIs. The system is deployed with a Streamlit web interface, offering an intuitive user experience for uploading files, processing documents, and interacting with the model. All 100% locally and offline.



---



## 🔎 Overview



This project demonstrates a complete RAG pipeline that allows users to:

- Upload documents (PDF, TXT, DOCX)

- Ask natural language questions about the content

- Get accurate answers with source citations

- Run everything locally with no cloud costs



Key Features:

- Local LLM execution (Llama 3.2, Mistral, Phi-3)

- Vector similarity search with FAISS

- Document chunking with context preservation

- Source citation and transparency

- Web UI with Streamlit

- **Real time evaluation framework** with 5 automated metrics

- Performance tracking and analysis



---

## 🟦 Architecture

<p align="center">
  <img src="assets/Architecture.png" alt="App Overview" width="600"/>
</p>

---

## 💻 Demo

<p align="center">
  <img src="assets/D1.png" alt="Overview" width="45%"/>
  <img src="assets/D2.png" alt="Upload" width="45%"/>
</p>

<p align="center">
  <img src="assets/D3.png" alt="App Overview" width="400"/>
</p>


---

## 📊 Evaluation Framework

Automated quality assessment system for RAG performance monitoring.

### Evaluation Metrics
The system tracks 5 key metrics on every query:

| Metric | What It Measures |
|--------|------------------|
| **Answer Quality** | Length and structure appropriateness (50-500 words optimal) |
| **Source Attribution** | Presence of citations and source references |
| **Retrieval Effectiveness** | Number and relevance of retrieved documents |
| **Response Performance** | Query response time and latency |
| **Completeness** | Question keyword coverage in answer |

### Real time Monitoring Features
- ✅ Automatic evaluation on every query
- ✅ Performance trend tracking 
- ✅ Historical query analysis
- ✅ CSV export for detailed reporting
- ✅ Dashboards with metrics breakdown


## ⚡ Quick Start & Setup

Prerequisites

- Python 3.10+

- Ollama installed

- 8GB+ RAM recommended & Dedicated GPU



Installation



1. Clone the repository

```bash

git clone https://github.com/lutfihadiCEX/rag-document-qa.git

cd rag-document-qa

```

2. Create and activate the Conda environment

```bash

conda create -n rag python=3.10 -y
conda activate rag

```
If you already have an environment set up (e.g. base), you can skip creating a new one, but using a dedicated environment avoids version conflicts.


3. Install dependencies

```bash

pip install -r requirements.txt

```



4. Install and start Ollama

```bash

# Install Ollama from https://ollama.ai



# Pull a model

ollama pull llama3.2       # Recommended - default
ollama pull mistral        # Optional
ollama pull phi3           # Optional


# Start Ollama service (keep terminal open)

ollama serve

```



5. Run the application

```bash

streamlit run ui/streamlit_app.py

```



---

## 👨‍💻 Usage

### Via Web UI

1. **Upload Documents**: Click "Upload Documents" in sidebar (PDF, TXT, DOCX)
2. **Process**: Click "Process Documents" button (creates vector embeddings)
3. **Ask Questions**: Type questions in the "Ask Questions" tab
4. **View Sources**: Expand answers to see source citations
5. **Check Metrics**: Navigate to "Evaluation" tab to view quality scores
6. **Export Data**: Download evaluation results as CSV for analysis

### Evaluation Dashboard

After asking questions, access the Evaluation tab to:
- View overall quality scores and performance metrics
- Analyze response time trends
- Track citation rates and completeness
- Export detailed reports for optimization

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | LangChain | Document loading, chunking, RAG pipeline |
| **Vector DB** | FAISS | Similarity search and document retrieval |
| **Embeddings** | Sentence-Transformers | all-MiniLM-L6-v2 (384-dim vectors) |
| **LLM Runtime** | Ollama | Local inference (Llama 3.2, Mistral, Phi-3) |
| **UI** | Streamlit | Web interface and dashboards |
| **Evaluation** | Custom metrics + Pandas | Quality tracking |

---

## Acknowledgments

- 🦜 LangChain for the RAG framework

- 🦙 Ollama for local LLM inference

- 🤗 Hugging Face for embeddings

- FAISS for vector search

---

## Author

Lutfihadi

## Disclaimer

This project is for research purposes as part of GenAI exploring and learning. Not intended for production use.

---

## Future Enhancements

### Evaluation & Quality
- [ ] **RAGAS Integration**: Add semantic faithfulness and answer relevancy metrics
- [ ] **Human Feedback**: Implement thumbs up/down ratings for continuous improvement
- [ ] **A/B Testing**: Compare different retrieval strategies and prompt templates
- [ ] **Judge Model**: Use dedicated high parameter LLM (Llama-3-70B) for unbiased evaluation

### Performance & Scale
- [ ] **Hardware Optimization**: GGUF quantization for larger models on limited VRAM
- [ ] **Hybrid Search**: Combine BM25 keyword search with FAISS semantic search
- [ ] **Caching Layer**: Redis for frequently asked questions
- [ ] **Batch Processing**: Handle multiple queries simultaneously

### Features
- [ ] **Multi document Context**: Conversation memory across documents
- [ ] **Document Summarization**: LangChain chains for automatic summaries
- [ ] **Multi language Support**: Extend to non English documents
- [ ] **Docker Deployment**: Containerization for easier deployment
   


