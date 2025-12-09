# Technical Walkthrough Video Script

**Target Audience:** ML Engineers / Graders
**Duration:** 5-10 minutes
**Goal:** Explain how the system works, where ML concepts are applied, and what was challenging

---

## INTRO (30 seconds)

> "This is the technical walkthrough of the Tribly AI Assistant, an agentic RAG system for university students. I'll show you the architecture, walk through the code, explain where the ML concepts are applied, and discuss the challenges I faced."

---

## SECTION 1: Architecture Overview (1 minute)

**[Show this diagram on screen]**

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  CONTEXT MANAGER                    │
│  (Orchestrates the pipeline)        │
└─────────────────┬───────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│ GEMINI │  │  TOOL    │  │RETRIEVER │
│ CLIENT │  │ EXECUTOR │  │          │
└────────┘  └──────────┘  └────┬─────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │EMBEDDING │    │ VECTOR   │    │ RANKING  │
        │ SERVICE  │    │  STORE   │    │ SERVICE  │
        └──────────┘    └──────────┘    └──────────┘
```

> "The system has four main stages:
> 1. **Query embedding** - Convert text to vectors using Sentence Transformers
> 2. **Vector retrieval** - Find similar documents using ChromaDB
> 3. **Multi-signal ranking** - Re-rank using engineered features
> 4. **Response generation** - LLM synthesizes the final answer
>
> What makes this agentic is the LLM decides which tools to call autonomously."

---

## SECTION 2: Embedding Layer (1.5 minutes)

**[Show: `src/embeddings/embedding_service.py`]**

> "Let's start with embeddings. Open `embedding_service.py`."

**[Highlight lines 32-34]**
```python
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
```

> "I'm using the all-MiniLM-L6-v2 model from Sentence Transformers. It's a distilled BERT model that produces 384-dimensional embeddings. I chose this over larger models like mpnet because it's 5x faster with only a small quality tradeoff - important for real-time search."

**[Highlight lines 81-111 - embed_text method]**

> "The `embed_text` method handles single queries. Notice the caching mechanism - embeddings are cached to avoid recomputation for repeated queries. The cache uses simple LRU eviction when it hits 10,000 entries."

**[Highlight lines 182-200 - compute_similarity]**

> "For similarity, I use cosine similarity - standard for text embeddings because it's length-invariant. A short query should match a long document if they're semantically similar."

**ML Concepts Applied:**
- Pre-trained transformer model (BERT-based)
- Sentence embeddings for semantic similarity
- Cosine similarity metric

---

## SECTION 3: Vector Store & Retrieval (1.5 minutes)

**[Show: `src/retrieval/vector_store.py`]**

> "The vector store uses ChromaDB, which implements HNSW indexing for approximate nearest neighbor search."

**[Highlight lines 32-41]**
```python
COLLECTIONS = [
    "reviews", "events", "hangouts", "posts",
    "resources", "classes", "teachers", "groups"
]
```

> "I maintain 8 separate collections - one for each content type. This enables targeted search. When someone asks about professors, I only search `teachers` and `reviews`, not events."

**[Highlight lines 84-88]**
```python
self._collections[collection_name] = self._client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)
```

> "Each collection uses HNSW with cosine distance. HNSW builds a hierarchical graph for O(log n) search complexity instead of O(n) brute force."

**[Show: `src/retrieval/retriever.py`, lines 231-301 - keyword_search]**

> "I also implemented keyword search as a baseline. It uses simple term frequency scoring - count matching terms divided by query length. This lets me quantitatively compare semantic vs keyword approaches."

**[Show: lines 303-364 - hybrid_search]**

> "The hybrid method combines both using Reciprocal Rank Fusion. RRF is rank-based, not score-based, so it handles the different scoring scales naturally. The formula is 1/(k + rank) with k=60."

**ML Concepts Applied:**
- HNSW approximate nearest neighbor search
- Multiple retrieval strategies (semantic, keyword, hybrid)
- Reciprocal Rank Fusion for score combination

---

## SECTION 4: Multi-Signal Ranking (1.5 minutes)

**[Show: `src/retrieval/ranking.py`]**

> "Raw similarity scores don't always produce the best results. A 5-year-old review might be semantically similar but less useful than a recent one."

**[Highlight lines 20-36 - RankingSignals dataclass]**
```python
@dataclass
class RankingSignals:
    semantic_similarity: float = 0.0
    rating_score: float = 0.0
    vote_score: float = 0.0
    recency_score: float = 0.0
    popularity_score: float = 0.0
```

> "I engineered 5 ranking features:
> 1. **Semantic similarity** - from embeddings
> 2. **Rating score** - normalized user ratings
> 3. **Vote score** - upvote/downvote ratio with Laplace smoothing
> 4. **Recency score** - exponential decay with 30-day half-life
> 5. **Popularity score** - engagement metrics"

**[Highlight lines 59-89 - CONTEXT_WEIGHTS]**

> "Weights are context-specific. For events, recency gets 30% weight because past events are useless. For reviews, rating gets 25% because quality matters most."

**[Highlight lines 230-270 - recency calculation]**

> "Recency uses exponential decay: score = 0.5^(age_days/30). Content from today scores 1.0, 30 days ago scores 0.5, 60 days ago scores 0.25."

**ML Concepts Applied:**
- Feature engineering
- Weighted score combination
- Exponential decay for time-based features
- Laplace smoothing for vote scores

---

## SECTION 5: Agentic System (1.5 minutes)

**[Show: `src/generation/gemini_client.py`]**

> "This is where it gets interesting. The system is agentic - the LLM autonomously decides which tools to call."

**[Highlight lines 291-330 - run_agent method]**

> "The `run_agent` method implements an agentic loop. It sends the query to Gemini with available tools. Gemini either responds with text OR requests tool calls. If tools are requested, I execute them and feed results back. This loops until Gemini generates a final response."

**[Show: `src/generation/tools.py`, lines 79-118]**

> "I defined 8 tools the LLM can choose from. Each has a name, description, and parameter schema. The description is critical - it tells the LLM when to use each tool."

**[Show: `src/generation/context_manager.py`, lines 26-54]**

> "The system also maintains conversation memory. The `Conversation` class stores turns, and `get_messages()` retrieves history for context. This enables follow-up questions like 'tell me more about the second one.'"

**ML Concepts Applied:**
- LLM function calling / tool use
- Agentic loop with autonomous decision-making
- Multi-turn conversation with memory

---

## SECTION 6: Evaluation (1.5 minutes)

**[Show: `src/evaluation/metrics.py`]**

> "Rigorous evaluation was essential. I created 25 test queries with ground truth across 5 categories."

**[Highlight lines 33-60 - TEST_QUERIES sample]**

> "Each query specifies relevant collections and keywords for automatic relevance judgment. Manual labeling thousands of query-document pairs isn't feasible, so I use collection matching plus keyword presence as a proxy."

**[Highlight lines 252-262 - RetrievalEvaluator class]**

> "The evaluator computes standard IR metrics:
> - **Precision@k** - what fraction of top-k results are relevant
> - **Recall@k** - what fraction of all relevant docs are in top-k
> - **MRR** - reciprocal of first relevant result's rank
> - **Latency** - response time in milliseconds"

**[Show results table or chart]**

| Metric | Keyword | Semantic | Hybrid |
|--------|---------|----------|--------|
| MRR | 0.40 | 0.56 | 0.57 |
| Recall@10 | 0.47 | 0.64 | 0.68 |
| Latency | 1ms | 66ms | 25ms |

> "Key findings: Semantic search improves MRR by 40% over keyword baseline. Hybrid performs best overall. Latency is under 100ms - imperceptible to users."

**ML Concepts Applied:**
- Information retrieval evaluation metrics
- Baseline comparison
- Quantitative analysis

---

## SECTION 7: Challenges & Contributions (1 minute)

> "Let me discuss what was challenging and where the significant contributions are."

**Challenge 1: Type Mismatches**
> "Gemini sometimes returns floats instead of ints for parameters like `top_k=10.0`. ChromaDB rejects this. I added defensive type casting throughout the retrieval layer."

**Challenge 2: Relevance Judgment**
> "Without manual labels, I needed automatic relevance estimation. The combination of collection matching plus keyword presence works but isn't perfect. A future improvement would be human-labeled test sets."

**Challenge 3: Balancing Ranking Signals**
> "Tuning the 5 feature weights required experimentation. Too much recency and old but highly-rated content disappears. Too much similarity and low-quality recent content dominates."

**Significant Contributions:**
1. **End-to-end agentic RAG** - Not just retrieval, but autonomous tool selection
2. **Multi-signal ranking** - 5 engineered features with context-specific weights
3. **Comprehensive evaluation** - 25 queries, 3 methods, 8 metrics
4. **Modular architecture** - Clean separation enabling independent testing

---

## CONCLUSION (30 seconds)

> "To summarize: This is a production-grade agentic RAG system with:
> - Sentence Transformer embeddings
> - ChromaDB vector search with HNSW indexing
> - Multi-signal ranking with 5 engineered features
> - Agentic LLM with 8 tools and conversation memory
> - Rigorous evaluation showing 40% MRR improvement over baseline
>
> The code is modular, well-documented, and ready for extension. Thank you."

---

## Quick Reference: File Locations to Show

| Topic | File | Key Lines |
|-------|------|-----------|
| Embeddings | `src/embeddings/embedding_service.py` | 32-34, 81-111, 182-200 |
| Vector Store | `src/retrieval/vector_store.py` | 32-41, 84-88, 176-246 |
| Retrieval | `src/retrieval/retriever.py` | 166-229, 231-301, 303-364 |
| Ranking | `src/retrieval/ranking.py` | 20-36, 59-89, 230-270 |
| Agentic Loop | `src/generation/gemini_client.py` | 291-330 |
| Tools | `src/generation/tools.py` | 79-118 |
| Conversation | `src/generation/context_manager.py` | 26-54 |
| Evaluation | `src/evaluation/metrics.py` | 33-60, 252-262, 370-433 |

---

## Recording Tips

1. **Screen setup**: Have VS Code open with the files ready
2. **Use split view**: Code on left, terminal/diagrams on right
3. **Highlight code**: Use VS Code's line highlighting as you explain
4. **Pace**: Don't rush - pause when showing important code
5. **Show don't tell**: Actually scroll through the code, don't just describe it
