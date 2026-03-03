# Health AI Chatbot - Multi-Agent Diabetes Risk Prediction

A diabetes risk assessment tool combining machine learning risk prediction with evidence-based medical insights through RAG (Retrieval-Augmented Generation).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Missing Files & How to Handle Them](#missing-files--how-to-handle-them)
- [Multi-Agent System](#multi-agent-system)
- [Phase 2 RAG Features](#phase-2-rag-features)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Overview

The Health AI Chatbot is a diabetes risk assessment tool that combines:

1. **ML-Powered Risk Prediction**: XGBoost model trained on BRFSS data
2. **Gemini Agent**: Conversational health coaching and lifestyle guidance
3. **RAG Agent**: Evidence-based clinical insights from medical literature with citations

Phase 2 RAG uses hybrid search (semantic + keyword), cross-encoder re-ranking, query expansion, and citation validation.

---

## Features

### Multi-Agent Chatbot System

- **Gemini Agent**: Personalized health advice, motivation, and lifestyle coaching
- **RAG Agent**: Clinical insights with citations from ADA 2024 guidelines and medical literature
- **Seamless Switching**: Toggle between agents while maintaining context
- **Context-Aware**: Prediction results automatically shared with all agents

### Diabetes Risk Assessment

- XGBoost-based prediction model (trained on BRFSS 2015 dataset)
- Interactive visualizations (radar charts, bar charts, risk gauge)
- Personalized health insights based on 11 risk factors
- Comparison with diabetic population averages

### Phase 2 RAG Features

- **Hybrid Search**: Combines semantic (70%) + BM25 keyword search (30%)
- **Cross-Encoder Re-Ranking**: Improved relevance over standard retrieval
- **Query Expansion**: Automatically expands medical terms (e.g., A1C → HbA1c)
- **Citation Validation**: Citation coverage with quality checks
- **Quality Filtering**: Evidence-level filtering (Level A/B/C sources)
- **6,328+ Medical Chunks**: Indexed from ADA 2024 diabetes guidelines

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/PraveenSalapu/HealthAgentDiabetic.git
cd HealthAgentDiabetic

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your Gemini API key (and optionally Qdrant credentials)

# 5. Run application
streamlit run app_modular.py
```

**For detailed setup instructions, see [SETUP.md](SETUP.md)**

---

## Missing Files & How to Handle Them

When you clone this repository, **some files will be missing by design**. Here's what to expect and how to fix it:

### 1. `.streamlit/secrets.toml` - MISSING

**Why:** Contains sensitive API keys (excluded for security)

**How to fix:**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` and add your credentials:

**Minimum (Gemini Agent only):**

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
```

**Full (with RAG Agent):**

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
QDRANT_URL = "https://your-cluster.aws.cloud.qdrant.io"
QDRANT_API_KEY = "your_qdrant_api_key_here"
```

**Get API keys:**
- Gemini: https://makersuite.google.com/app/apikey
- Qdrant: https://cloud.qdrant.io/ (free tier available)

### 2. `data/clinical_docs/` - MISSING

**Why:** Large PDF files (100+ MB) excluded from git

**Impact:**
- RAG Agent will still work with built-in sample knowledge base
- For full medical literature access, add your own documents

**How to fix (optional):**

```bash
mkdir -p data/clinical_docs
# Add PDF or TXT files with medical content
```

**Note:** The RAG agent automatically indexes documents on first run.

### 3. `data/chroma_db/` - DEPRECATED

**Why:** Old vector database (Phase 1) replaced by Qdrant Cloud (Phase 2)

**Impact:** Not needed - Phase 2 uses cloud-based Qdrant for vector storage

**How to fix:** No action needed

### 4. Test/Script Files - EXCLUDED

**Files excluded:** `test_*.py`, `verify_rag.py`, `reindex_documents.py`

**Why:** Development/testing scripts not needed for production

**Impact:** None - application works without them

---

## Multi-Agent System

### Gemini Agent

**Purpose:** Conversational health coaching

**Capabilities:**
- General health advice and education
- Lifestyle recommendations (diet, exercise, sleep)
- Motivational support and habit coaching
- Stress management guidance

**Technology:** Google Gemini 2.5 Flash API

---

### RAG Agent

**Purpose:** Evidence-based clinical insights

**Capabilities:**
- Citations from ADA 2024 diabetes guidelines
- Research-backed medical recommendations
- Evidence-level transparency (Level A/B/C sources)
- Grounded responses (no speculation)

**Technology Stack:**
- **Vector DB:** Qdrant Cloud
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Search:** Hybrid (semantic + BM25)
- **Re-ranking:** Cross-encoder
- **Generation:** Gemini 2.5 Flash

---

## Phase 2 RAG Features

### Architecture Comparison

| Feature | Phase 1 | Phase 2 |
|---------|---------|--------|
| **Search Type** | Semantic only | Hybrid (semantic + BM25) |
| **Relevance Scoring** | Basic similarity | Cross-encoder re-ranking |
| **Query Understanding** | Literal | Medical synonym expansion |
| **Quality Control** | None | Citation validation + filtering |
| **Vector Database** | Local ChromaDB | Qdrant Cloud |
| **Documents Indexed** | Sample only | 6,328 chunks (ADA 2024) |

### Technical Architecture

```
User Query
  ↓
Query Expansion (A1C → HbA1c, hemoglobin A1c)
  ↓
Hybrid Retrieval (70% semantic + 30% BM25)
  ↓
Cross-Encoder Re-Ranking (top 3 of 10)
  ↓
Quality Filtering (Evidence Level A/B/C, score >0.65)
  ↓
Generation (grounded prompt)
  ↓
Citation Validation
  ↓
Response with Citations
```

---

## Project Structure

```
HealthAgentDiabetic/
├── app_modular.py              # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md
├── SETUP.md                    # Detailed setup instructions
├── .gitignore
│
├── .streamlit/
│   ├── secrets.toml.example    # Template for API keys
│   └── secrets.toml            # Your secrets (NOT in git)
│
├── agents/
│   ├── base_agent.py
│   ├── gemini_agent.py
│   ├── rag_agent.py
│   ├── retrieval_components.py # Hybrid search, re-ranking
│   └── agent_manager.py
│
├── config/
│   ├── settings.py
│   └── document_metadata.py
│
├── models/
│   ├── model_loader.py
│   └── predictor.py
│
├── ui/
│   ├── chat_interface.py
│   ├── forms.py
│   ├── visualizations.py
│   └── styles.py
│
├── utils/
│   └── helpers.py
│
├── data/
│   └── clinical_docs/          # Medical PDFs (NOT in git)
│
└── model_output2/
    ├── xgboost_model.json
    ├── preprocessor.pkl
    ├── optimal_threshold.json
    └── diabetic_averages.json
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[SETUP.md](SETUP.md)** | Complete setup guide with troubleshooting |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | Deploy to Streamlit Cloud |
| **[IMPROVED_RAG_DESIGN.md](IMPROVED_RAG_DESIGN.md)** | RAG architecture details |
| **[PHASE1_RESULTS.md](PHASE1_RESULTS.md)** | Phase 1 implementation results |
| **[PHASE2_RESULTS.md](PHASE2_RESULTS.md)** | Phase 2 testing and benchmarks |

---

## Usage

### 1. Start the Application

```bash
streamlit run app_modular.py
```

### 2. Complete Health Assessment

Fill out the form with 11 health metrics:
- General Health, BMI, Age
- High BP, High Cholesterol, Heart Disease
- Physical Activity, Difficulty Walking
- Physical Health Days, Income, Education

### 3. Get Risk Prediction

- XGBoost model predicts diabetes risk
- Visualizations show risk factors
- Comparison with diabetic population

### 4. Chat with Agents

- **Gemini:** For lifestyle tips, general wellness advice
- **RAG:** For clinical information with citations (e.g., ADA recommendations)
- Switch between agents seamlessly

---

## Configuration

Edit `config/settings.py` to customize:

### RAG Settings:

```python
RAG_USE_HYBRID_SEARCH = True    # Enable semantic + BM25
RAG_USE_RERANKING = True        # Enable cross-encoder
RAG_HYBRID_ALPHA = 0.7          # 70% semantic, 30% keyword
RAG_MIN_RELEVANCE_SCORE = 0.65  # Minimum similarity
RAG_TOP_K = 3                   # Final documents used
RAG_INITIAL_K = 10              # Initial retrieval count
```

### Model Selection:

```python
# For better medical accuracy (requires re-indexing):
RAG_EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
```

---

## Testing

```bash
python -c "from models import load_model_components; print('Models OK')"
python -c "from agents import GeminiAgent, RAGAgent; print('Agents OK')"
python -c "from ui import render_chat_interface; print('UI OK')"
```

---

## Deployment

### Streamlit Cloud:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect GitHub and select repository
4. Set main file: `app_modular.py`
5. Add secrets in deployment settings:
```toml
GEMINI_API_KEY = "your_key"
QDRANT_URL = "your_url"
QDRANT_API_KEY = "your_key"
```
6. Deploy

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details.

---

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes
4. **Push** to the branch
5. **Open** a Pull Request

---

## Disclaimer

**This application is for educational and informational purposes only.**

- Not a substitute for professional medical advice
- Not for diagnosis or treatment
- Not FDA approved or medically certified
- Always consult healthcare providers for medical decisions

---

## Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/PraveenSalapu/HealthAgentDiabetic/issues)
- **Developer:** Praveen Salapu
- **Repository:** https://github.com/PraveenSalapu/HealthAgentDiabetic

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Acknowledgments

- **ADA 2024 Guidelines:** American Diabetes Association Standards of Care
- **BRFSS Dataset:** CDC Behavioral Risk Factor Surveillance System
- **Technology:** Streamlit, LangChain, Qdrant, Google Gemini, XGBoost
