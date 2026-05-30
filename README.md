# 🤖 Advanced RAG Chatbot

A conversational AI with long-term memory, RAG, web search,
and stock price tools — built with LangGraph, Groq, and Streamlit.

## Features
- 🧠 Long-term memory via PostgreSQL
- 📄 PDF RAG — upload and query documents
- 🔍 Web search via DuckDuckGo
- 📈 Live stock prices via Alpha Vantage
- 💬 Multi-conversation thread management
- ⚡ Streaming responses via Groq LPU

## Tech Stack
| Layer | Technology |
|---|---|
| LLM | Groq llama-3.3-70b-versatile |
| Orchestration | LangGraph |
| Long-term memory | PostgreSQL |
| Short-term memory | SQLite |
| Vector store | FAISS |
| Frontend | Streamlit |

## Setup

### 1. Clone
```bash
git clone https://github.com/yourusername/advanced-rag-chatbot.git
cd advanced-rag-chatbot
```

### 2. Virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment variables
```bash
copy .env.example .env
# Edit .env and fill in your API keys
```

### 5. Run
```bash
streamlit run app.py
```