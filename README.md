# Tribly AI Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that helps university students discover professor reviews, events, and academic resources through natural language queries.

## What it Does

Tribly AI Assistant is an intelligent search and discovery system built for the Tribly university social platform. It enables students to:

- **Find Professor Reviews**: Ask natural language questions like "What do students think about Professor Smith's teaching style?" and get synthesized answers from real student reviews
- **Discover Events**: Find relevant campus events, study groups, and social hangouts based on interests and availability
- **Search Resources**: Locate study materials, notes, and guides for specific classes
- **Navigate the Platform**: Get help finding information that would otherwise require multiple searches

The system uses semantic search to understand the *meaning* behind queries (not just keywords) and generates helpful, contextual responses using Claude AI.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/tribly-ai-assistant
cd tribly-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Index the sample data
python scripts/index_documents.py

# Run the demo application
streamlit run app/streamlit_app.py
```

## Video Links

- **Demo Video**: [(https://youtu.be/C5Un_awL5V4)]
- **Technical Walkthrough**: [(https://youtu.be/9zT5eCVqJzw)]

## Evaluation

### Retrieval Performance

| Metric | Keyword Baseline | Vector Search | Hybrid |
|--------|------------------|---------------|--------|
| Precision@5 | TBD | TBD | TBD |
| Recall@5 | TBD | TBD | TBD |
| MRR | TBD | TBD | TBD |

### Response Quality

| Aspect | Score |
|--------|-------|
| Relevance | TBD |
| Accuracy | TBD |
| Helpfulness | TBD |

### Ablation Study Results

See `notebooks/05_ablation_study.ipynb` for detailed analysis of:
- Chunk size impact on retrieval quality
- Top-k retrieval variations
- Embedding model comparisons
- Prompt template effectiveness

## Project Structure

```
tribly-ai-assistant/
├── src/
│   ├── embeddings/      # Sentence transformer embeddings
│   ├── retrieval/       # Vector search and ranking
│   ├── generation/      # Gemeni LLM integration
│   ├── recommendation/  # Collaborative filtering
│   ├── guardrails/      # Safety and content filtering
│   ├── evaluation/      # Metrics and analysis
│   └── api/             # FastAPI endpoints
├── notebooks/           # Jupyter notebooks for experiments
├── data/                # Sample data and processed indices
├── results/             # Evaluation metrics and figures
├── app/                 # Streamlit demo application
├── tests/               # Unit tests
└── scripts/             # Utility scripts
```

## Individual Contributions

This is a solo project completed by Natnael Worku.

### Key Technical Contributions:
1. **RAG Pipeline**: End-to-end retrieval-augmented generation system
2. **Multi-modal Embeddings**: Combined semantic and behavioral features
3. **Evaluation Framework**: Comprehensive metrics and ablation studies
4. **Production-ready API**: FastAPI backend with proper error handling

