# Multimodal RAG Project

Welcome to the Multimodal RAG AI Search & Recommender Systems project.

## Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run CLI version
python multimodal_rag_final.py

# Run Web UI
streamlit run multimodal_rag_final_ui.py
```

## Features

- Image-based semantic search
- Multi-modal reasoning with LLaVA
- Vector database integration with ChromaDB
- Open-source, locally-hosted solution

## Project Structure

```
multimodal-rag/
├── multimodal_rag_final.py       # CLI interface
├── multimodal_rag_final_ui.py    # Web UI (Streamlit)
├── init_flower_db.py             # Database initialization
├── requirements.txt              # Dependencies
└── images/                       # Sample images
```

## Technologies

- **Ollama**: LLaVA vision-language model
- **ChromaDB**: Vector database
- **LangChain**: LLM orchestration
- **Streamlit**: Web interface

## License

Open source project for educational use.
