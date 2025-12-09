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

- **Demo Video**: [Link to non-technical demo showing the assistant in action]
- **Technical Walkthrough**: [Link to code walkthrough explaining ML techniques]

## Evaluation

### Retrieval Performance

| Metric | Keyword Baseline | Semantic Search | Hybrid |
|--------|------------------|-----------------|--------|
| Precision@1 | 0.24 | 0.40 | 0.44 |
| Precision@5 | 0.27 | 0.28 | 0.28 |
| Recall@5 | 0.39 | 0.44 | 0.44 |
| Recall@10 | 0.47 | 0.64 | 0.68 |
| MRR | 0.40 | 0.56 | 0.57 |
| Avg Latency | 1ms | 66ms | 25ms |

**Key Findings:**
- **Semantic search** achieves the best MRR (0.56), ranking relevant results higher
- **Hybrid search** provides the best Recall@10 (0.68) for comprehensive retrieval
- **Keyword search** is 66x faster but with lower quality metrics

### Evaluation Details

Evaluated on 25 test queries across 5 categories:
- Professor/teacher queries (5)
- Class/course queries (5)
- Study resource queries (5)
- Event queries (5)
- General/mixed queries (5)

### Comparison Analysis

See `notebooks/evaluation.ipynb` for detailed analysis including:
- Precision@k comparison charts
- Recall@k comparison charts
- MRR comparison visualization
- Latency comparison
- Performance heatmap

Generated charts saved to `results/figures/`

## Project Structure

```
tribly-ai-assistant/
├── src/
│   ├── embeddings/      # Sentence transformer embeddings
│   ├── retrieval/       # Vector search and ranking
│   ├── generation/      # Claude LLM integration
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

## License

This project is for educational purposes as part of COMPSCI 372 FALL 2025.
