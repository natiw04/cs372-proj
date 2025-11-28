# Setup Instructions

This guide provides step-by-step instructions to set up and run the Tribly AI Assistant.

## Prerequisites

- **Python 3.11+** (3.10 may work but is not tested)
- **pip** or **conda** for package management
- **Git** for version control
- **Anthropic API Key** for Claude LLM (get one at https://console.anthropic.com/)

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, GPU with CUDA support for faster embeddings
- **Storage**: ~2GB for dependencies and model weights

## Installation Options

### Option 1: Using pip (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/tribly-ai-assistant
cd tribly-ai-assistant

# 2. Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; import sentence_transformers; import chromadb; print('All dependencies installed successfully!')"
```

### Option 2: Using Conda

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/tribly-ai-assistant
cd tribly-ai-assistant

# 2. Create conda environment from file
conda env create -f environment.yml

# 3. Activate environment
conda activate tribly-ai

# 4. Verify installation
python -c "import torch; import sentence_transformers; import chromadb; print('All dependencies installed successfully!')"
```

## Configuration

### Step 1: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env
```

### Step 2: Edit `.env` with Your Configuration

Open `.env` in your preferred editor and fill in the required values:

```bash
# Required: Your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Optional: Only needed if exporting fresh data from PocketBase
POCKETBASE_URL=http://localhost:8090
POCKETBASE_ADMIN_EMAIL=admin@example.com
POCKETBASE_ADMIN_PASSWORD=your-password

# These can usually be left as defaults
CHROMA_PERSIST_DIR=./data/chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=claude-sonnet-4-20250514
```

## Data Setup

### Option A: Use Included Sample Data

The repository includes sample data in `data/raw/`. To index it:

```bash
python scripts/index_documents.py
```

This will:
1. Load all JSON files from `data/raw/`
2. Chunk documents appropriately
3. Generate embeddings using sentence-transformers
4. Store vectors in ChromaDB at `data/chroma/`

### Option B: Export Fresh Data from PocketBase

If you have access to a running Tribly PocketBase instance:

```bash
# Make sure PocketBase is running and credentials are in .env
python scripts/export_data.py

# Then index the exported data
python scripts/index_documents.py
```

## Running the Application

### Demo UI (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

This will start a web interface at `http://localhost:8501`

### API Server (FastAPI)

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001
```

API documentation will be available at `http://localhost:8001/docs`

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

**1. CUDA/GPU not detected**
```bash
# Check if PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available())"
```
If False, embeddings will run on CPU (slower but functional).

**2. Out of memory when generating embeddings**
Reduce batch size in `scripts/index_documents.py`:
```python
BATCH_SIZE = 16  # Reduce from default 32
```

**3. Anthropic API rate limits**
The system includes exponential backoff. If you hit limits frequently, add delays:
```python
import time
time.sleep(1)  # Between API calls
```

**4. ChromaDB persistence errors**
Delete the existing database and re-index:
```bash
rm -rf data/chroma/
python scripts/index_documents.py
```

### Getting Help

- Check existing issues on GitHub
- Review the technical walkthrough video
- Ensure all prerequisites are met before opening an issue

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking (optional)

```bash
# Format code
black src/ tests/
isort src/ tests/
```

### Adding New Features

1. Create a new branch: `git checkout -b feature/your-feature`
2. Implement changes in the appropriate `src/` module
3. Add tests in `tests/`
4. Update documentation as needed
5. Submit a pull request
