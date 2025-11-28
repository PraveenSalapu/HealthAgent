# Setup Guide - Health AI Chatbot

Complete setup instructions for running the Health AI Chatbot with Phase 2 RAG improvements.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Phase 2 RAG Configuration](#phase-2-rag-configuration)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or 3.11)
- **Git** for cloning the repository
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **Qdrant Cloud Account** (for Phase 2 RAG features) - [Free tier available](https://cloud.qdrant.io/)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/PraveenSalapu/HealthAgentDiabetic.git
cd HealthAgentDiabetic

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure secrets (see below)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API keys

# 6. Run the application
streamlit run app_modular.py
```

---

## Detailed Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/PraveenSalapu/HealthAgentDiabetic.git
cd HealthAgentDiabetic
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify activation:
```bash
which python  # Should show path to .venv/bin/python
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
- `streamlit` - Web UI framework
- `google-generativeai` - Gemini API client
- `xgboost` - ML prediction model
- `langchain` - RAG framework
- `qdrant-client` - Vector database client (Phase 2)
- `sentence-transformers` - Text embeddings (Phase 2)
- `rank-bm25` - Keyword search (Phase 2)

### Step 4: Configure API Keys

#### Create secrets file:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

#### Edit `.streamlit/secrets.toml`:

**Minimum configuration (Gemini Agent only):**
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-1.5-flash"
```

**Full configuration (with Phase 2 RAG):**
```toml
# Gemini Configuration
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-1.5-flash"

# Qdrant Cloud Configuration (for RAG Agent)
QDRANT_URL = "https://your-cluster.aws.cloud.qdrant.io"
QDRANT_API_KEY = "your_qdrant_api_key_here"
```

---

## Phase 2 RAG Configuration

The RAG (Retrieval-Augmented Generation) Agent provides evidence-based clinical information with citations from medical literature.

### Step 1: Get Qdrant Cloud Credentials

1. **Sign up** at [cloud.qdrant.io](https://cloud.qdrant.io/)

2. **Create a Cluster:**
   - Click "Create Cluster"
   - Name: `health-bot-cluster`
   - Select **Free Tier** (1GB storage, sufficient for testing)
   - Region: Choose closest to you
   - Click "Create"

3. **Get API Key:**
   - Navigate to "Data Access Control" or "API Keys"
   - Click "Create API Key"
   - **Copy and save immediately** (shown only once)

4. **Get Cluster URL:**
   - Go to cluster overview
   - Copy the **Cluster URL** (format: `https://xxx-xxx.region.cloud.qdrant.io`)

5. **Add to secrets.toml:**
   ```toml
   QDRANT_URL = "https://your-cluster.aws.cloud.qdrant.io"
   QDRANT_API_KEY = "qc_xxxxxxxxxxxxxxxxxxxxxx"
   ```

### Step 2: Add Clinical Documents (Optional)

The RAG agent works out-of-the-box with sample diabetes knowledge, but you can add your own documents:

1. **Create documents folder:**
   ```bash
   mkdir -p data/clinical_docs
   ```

2. **Add documents:**
   - Supported formats: **PDF** and **TXT**
   - Place files in `data/clinical_docs/`
   - Examples:
     - ADA diabetes guidelines (PDF)
     - Clinical research papers (PDF)
     - Medical reference texts (TXT)

3. **Documents are auto-indexed:**
   - On first run, the RAG agent automatically indexes all documents
   - Creates embeddings and stores in Qdrant Cloud
   - Takes 1-2 minutes for ~20 PDFs
   - Progress shown in terminal

### Step 3: Verify RAG Setup

1. **Start the application:**
   ```bash
   streamlit run app_modular.py
   ```

2. **Check terminal logs:**
   ```
   [INFO] Initializing Phase 2 components...
   [OK] Retrieved 6328 documents from vector store
   [OK] BM25 index built with 6328 documents
   [OK] Hybrid retriever initialized (alpha=0.7)
   [OK] Phase 2 components ready
   ```

3. **Test RAG Agent:**
   - Complete health assessment
   - Switch to **RAG Agent** in chat
   - Ask: "What are ADA recommendations for diabetes prevention?"
   - Response should include citations like `[Source: American Diabetes Association, 2024]`

---

## Phase 2 Features

When properly configured, you get these advanced features:

### 1. Hybrid Search
- **70% Semantic search** (meaning-based)
- **30% BM25 keyword search** (exact term matching)
- Better precision for medical terminology

### 2. Query Expansion
- Automatically expands medical terms
- Example: "A1C" → "HbA1c", "hemoglobin A1c", "glycated hemoglobin"

### 3. Cross-Encoder Re-Ranking
- Precise relevance scoring of retrieved documents
- Ensures best matches are selected
- 40-60% improvement over basic semantic search

### 4. Citation Validation
- Automated quality checks on responses
- Ensures all claims have citations
- Validates medical disclaimers included
- Detects speculative language

### 5. Quality Filtering
- Minimum relevance score: 0.65
- Evidence level filtering (A, B, C)
- Flags outdated guidelines (>5 years old)

---

## Running the Application

### Start the app:
```bash
streamlit run app_modular.py
```

### Access in browser:
```
Local URL: http://localhost:8501
```

### Using the Multi-Agent System:

1. **Complete Health Assessment**
   - Fill out the form with health metrics
   - Click "Predict Risk"

2. **Chat with Agents:**
   - **Gemini Agent**: General health advice, motivation, lifestyle tips
   - **RAG Agent**: Clinical insights with medical citations

3. **Switch Between Agents:**
   - Use the agent selector in the chat interface
   - Conversation history maintained separately per agent
   - Prediction context automatically shared

---

## Missing Files (Expected)

These files/folders are intentionally excluded from git for security or size reasons:

### 1. `.streamlit/secrets.toml` ❌
**Why missing:** Contains sensitive API keys
**How to fix:** Copy from `.streamlit/secrets.toml.example` and add your keys
**Required for:** All features

### 2. `data/clinical_docs/` ❌
**Why missing:** Large PDF files (100+ MB)
**How to fix:** Create folder and add your own medical PDFs, or use sample knowledge base
**Required for:** RAG Agent (optional - works with sample data)

### 3. `.env` ❌
**Why missing:** Alternative to secrets.toml (not used in this project)
**How to fix:** Not needed - use `.streamlit/secrets.toml` instead
**Required for:** Not used

### 4. `data/chroma_db/` ❌
**Why missing:** Old vector database (replaced by Qdrant Cloud in Phase 2)
**How to fix:** Not needed - Phase 2 uses Qdrant Cloud
**Required for:** Deprecated

---

## Environment Variables (Alternative to secrets.toml)

You can use environment variables instead of secrets.toml:

**Windows:**
```cmd
set GEMINI_API_KEY=your_key_here
set QDRANT_URL=your_url_here
set QDRANT_API_KEY=your_key_here
streamlit run app_modular.py
```

**macOS/Linux:**
```bash
export GEMINI_API_KEY=your_key_here
export QDRANT_URL=your_url_here
export QDRANT_API_KEY=your_key_here
streamlit run app_modular.py
```

**Priority order:**
1. Streamlit secrets.toml (recommended)
2. Environment variables
3. config/settings.py defaults

---

## Troubleshooting

### Issue: "ImportError: No module named X"
**Solution:**
```bash
pip install -r requirements.txt
# If still failing, try:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Issue: "Connection refused" (Qdrant)
**Causes:**
- Invalid QDRANT_URL or QDRANT_API_KEY
- Firewall blocking connection
- Free tier quota exceeded

**Solution:**
1. Verify credentials in secrets.toml
2. Check Qdrant cluster is "Active" in cloud.qdrant.io
3. Test connection:
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(url="your_url", api_key="your_key")
   print(client.get_collections())  # Should list collections
   ```

### Issue: "API key not valid" (Gemini)
**Solution:**
1. Get new key from https://makersuite.google.com/app/apikey
2. Update secrets.toml
3. Restart Streamlit

### Issue: RAG Agent says "I don't have information..."
**Causes:**
- No documents indexed
- Query doesn't match indexed content
- Qdrant connection failed

**Solution:**
1. Check terminal for indexing logs
2. Verify QDRANT_URL and QDRANT_API_KEY
3. Add documents to `data/clinical_docs/`
4. Restart app to trigger re-indexing

### Issue: Slow response times
**Normal:**
- First query: 15-20 seconds (loading models)
- Subsequent queries: 8-12 seconds (hybrid search + re-ranking)

**If slower:**
- Check internet connection (Gemini API calls)
- Reduce RAG_INITIAL_K in config/settings.py (default: 10)

### Issue: Model files not found
**Solution:**
```bash
# Verify model files exist:
ls model_output2/
# Should see: xgboost_model.json, preprocessor.pkl, etc.

# If missing, download from repository releases or train new model
```

---

## Advanced Configuration

Edit `config/settings.py` to customize:

### RAG Settings:
```python
# Retrieval
RAG_TOP_K = 3              # Final number of documents
RAG_INITIAL_K = 10         # Initial retrieval count
RAG_MIN_RELEVANCE_SCORE = 0.65  # Quality threshold

# Hybrid Search
RAG_USE_HYBRID_SEARCH = True   # Enable semantic + BM25
RAG_HYBRID_ALPHA = 0.7         # 70% semantic, 30% keyword

# Re-ranking
RAG_USE_RERANKING = True       # Enable cross-encoder
RAG_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Embeddings
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# For better medical accuracy: "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
```

### Chunking Settings:
```python
RAG_CHUNK_SIZE = 1000      # Characters per chunk
RAG_CHUNK_OVERLAP = 200    # Overlap between chunks
```

---

## Deployment to Streamlit Cloud

1. **Push to GitHub** (this repo)

2. **Go to [share.streamlit.io](https://share.streamlit.io/)**

3. **Deploy:**
   - Connect GitHub account
   - Select repository: `PraveenSalapu/HealthAgentDiabetic`
   - Main file: `app_modular.py`
   - Click "Deploy"

4. **Add Secrets:**
   - Go to app settings → Secrets
   - Paste contents of your local `secrets.toml`
   - Save

5. **App URL:** `https://your-app.streamlit.app`

---

## Testing the Installation

### Test 1: Basic Functionality
```bash
python -c "from models import load_model_components; print('✓ Models OK')"
python -c "from agents import GeminiAgent, RAGAgent; print('✓ Agents OK')"
python -c "from ui import render_chat_interface; print('✓ UI OK')"
```

### Test 2: Gemini Agent
```python
from agents import GeminiAgent

agent = GeminiAgent()
agent.initialize(api_key="your_key")
response = agent.generate_response("Tell me about diabetes prevention", {})
print(response[:200])  # Should get response
```

### Test 3: RAG Agent
```python
from agents import RAGAgent

agent = RAGAgent()
agent.initialize()
response = agent.generate_response("What does research say about exercise?", {})
print(response[:200])  # Should include citations
```

---

## Project Structure

```
HealthAgentDiabetic/
├── app_modular.py              # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── SETUP.md                    # This file (detailed setup)
├── .gitignore                  # Excluded files
│
├── .streamlit/
│   ├── secrets.toml.example   # Template for API keys
│   └── secrets.toml           # Your actual secrets (not in git)
│
├── agents/
│   ├── base_agent.py          # Abstract base class
│   ├── gemini_agent.py        # Gemini API agent
│   ├── rag_agent.py           # RAG agent with Phase 2 features
│   ├── retrieval_components.py # Phase 2: Hybrid search, re-ranking, etc.
│   └── agent_manager.py       # Multi-agent orchestration
│
├── config/
│   ├── settings.py            # All configuration constants
│   └── document_metadata.py   # Metadata for clinical documents
│
├── models/
│   ├── model_loader.py        # ML model loading
│   └── predictor.py           # Prediction logic
│
├── ui/
│   ├── chat_interface.py      # Chat UI components
│   ├── forms.py               # Input forms
│   ├── visualizations.py      # Charts and graphs
│   └── styles.py              # CSS styling
│
├── utils/
│   └── helpers.py             # Utility functions
│
├── data/
│   └── clinical_docs/         # Medical PDFs (not in git)
│
└── model_output2/             # Trained XGBoost model files
    ├── xgboost_model.json
    ├── preprocessor.pkl
    ├── optimal_threshold.json
    └── diabetic_averages.json
```

---

## Getting Help

- **GitHub Issues:** [Report bugs or request features](https://github.com/PraveenSalapu/HealthAgentDiabetic/issues)
- **Documentation:** See `README.md` for project overview
- **RAG Design:** See `IMPROVED_RAG_DESIGN.md` for technical details

---

## Next Steps After Setup

1. ✅ Complete health assessment form
2. ✅ Get diabetes risk prediction
3. ✅ Ask Gemini Agent for lifestyle advice
4. ✅ Ask RAG Agent for clinical insights with citations
5. ✅ Compare responses between agents
6. ✅ Explore Phase 2 features (hybrid search, re-ranking)

---

**Last Updated:** November 2024
**Phase 2 Status:** Production Ready ✅
