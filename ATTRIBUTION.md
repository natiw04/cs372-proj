# Attribution

This document provides detailed attribution for all external resources, libraries, and tools used in this project.

## Libraries and Frameworks

### Machine Learning

| Library | License | Purpose | URL |
|---------|---------|---------|-----|
| PyTorch | BSD-3-Clause | Deep learning framework | https://pytorch.org/ |
| Sentence Transformers | Apache 2.0 | Text embeddings | https://www.sbert.net/ |
| scikit-learn | BSD-3-Clause | ML utilities, metrics | https://scikit-learn.org/ |
| NumPy | BSD-3-Clause | Numerical computing | https://numpy.org/ |
| Pandas | BSD-3-Clause | Data manipulation | https://pandas.pydata.org/ |

### Vector Database

| Library | License | Purpose | URL |
|---------|---------|---------|-----|
| ChromaDB | Apache 2.0 | Vector storage and search | https://www.trychroma.com/ |

### LLM Integration

| Service | License | Purpose | URL |
|---------|---------|---------|-----|
| Google Gemini API | Commercial | Response generation

### API and Web

| Library | License | Purpose | URL |
|---------|---------|---------|-----|
| FastAPI | MIT | API framework | https://fastapi.tiangolo.com/ |
| Uvicorn | BSD-3-Clause | ASGI server | https://www.uvicorn.org/ |
| Pydantic | MIT | Data validation | https://docs.pydantic.dev/ |
| Streamlit | Apache 2.0 | Demo UI | https://streamlit.io/ |

### Utilities

| Library | License | Purpose | URL |
|---------|---------|---------|-----|
| httpx | BSD-3-Clause | Async HTTP client | https://www.python-httpx.org/ |
| python-dotenv | BSD-3-Clause | Environment management | https://github.com/theskumar/python-dotenv |
| tqdm | MIT | Progress bars | https://tqdm.github.io/ |
| Rich | MIT | Terminal formatting | https://rich.readthedocs.io/ |

## Pretrained Models

### Embedding Model

| Model | Source | License | Purpose |
|-------|--------|---------|---------|
| all-MiniLM-L6-v2 | Hugging Face | Apache 2.0 | Text embeddings (384 dimensions) |

Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

This model was trained by the Sentence Transformers team and is based on Microsoft's MiniLM architecture.

### Language Model

| Model | Provider | License | Purpose |
|-------|----------|---------|---------|
| Gemini (gemeni-2.5-flash) | Google | Commercial API | Response generation |

Used via Google Gemini's API for generating natural language responses based on retrieved context.

## AI-Generated Content Disclosure

In accordance with academic integrity requirements, the following AI tools were used during development:

### Google Gemini

- **Purpose**: Code assistance, documentation writing, debugging help
- **Scope**:
  - Helped structure the project architecture
  - Assisted with writing boilerplate code
  - Provided suggestions for evaluation metrics
  - Helped draft documentation
- **Review Process**: All AI-generated code was reviewed, tested, and modified as needed

## Data Sources

### Sample Data

The sample data included in `data/raw/` was exported from a development instance of the Tribly application:

- **Reviews**: Anonymized student reviews of professors and classes
- **Events**: Campus event listings (hangouts and organization events)
- **Posts**: Social feed posts from the platform
- **Resources**: Study materials and class resources

All personal identifiable information (PII) has been removed or anonymized:
- User IDs replaced with anonymous identifiers
- Email addresses removed
- Real names replaced with pseudonyms where applicable

## Code References

### Inspiration and Reference Implementations

| Concept | Reference | How Used |
|---------|-----------|----------|
| RAG Architecture | LangChain Documentation | Conceptual guidance for retrieval pipeline |
| Sentence Embeddings | SBERT.net tutorials | Implementation patterns for embedding generation |
| ChromaDB Integration | ChromaDB Documentation | Vector store setup and querying |


## Acknowledgments

- Professor Fain and TAs for guidance on project requirements
