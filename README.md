# Health AI Chatbot - Multi-Agent Diabetes Risk Prediction

> **An intelligent health assistant combining machine learning risk prediction with evidence-based medical insights through advanced RAG (Retrieval-Augmented Generation) technology.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

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

## 🎯 Overview

The Health AI Chatbot is a comprehensive diabetes risk assessment tool that combines:

1. **ML-Powered Risk Prediction**: XGBoost model trained on BRFSS data
2. **Gemini Agent**: Conversational health coaching and lifestyle guidance
3. **RAG Agent**: Evidence-based clinical insights from medical literature with citations

**Key Differentiator:** Phase 2 RAG implementation with **hybrid search** (semantic + keyword), **cross-encoder re-ranking**, **query expansion**, and **automated citation validation** for trustworthy medical information.

---

## ✨ Features

### 🤖 Multi-Agent Chatbot System
- **Gemini Agent**: Personalized health advice, motivation, and lifestyle coaching
- **RAG Agent**: Clinical insights with citations from ADA 2024 guidelines and medical literature
- **Seamless Switching**: Toggle between agents while maintaining context
- **Context-Aware**: Prediction results automatically shared with all agents

### 📊 Diabetes Risk Assessment
- XGBoost-based prediction model (trained on BRFSS 2015 dataset)
- Interactive visualizations (radar charts, bar charts, risk gauge)
- Personalized health insights based on 11 risk factors
- Comparison with diabetic population averages

### 🔬 Phase 2 RAG Features (Advanced)
- **Hybrid Search**: Combines semantic (70%) + BM25 keyword search (30%)
- **Cross-Encoder Re-Ranking**: 40-60% better relevance than standard retrieval
- **Query Expansion**: Automatically expands medical terms (e.g., A1C → HbA1c)
- **Citation Validation**: Ensures 100% citation coverage with quality checks
- **Quality Filtering**: Evidence-level filtering (Level A/B/C sources)
- **6,328+ Medical Chunks**: Indexed from ADA 2024 diabetes guidelines

---

## 🚀 Quick Start

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

**📖 For detailed setup instructions, see [SETUP.md](SETUP.md)**

---

## ⚠️ Missing Files & How to Handle Them

When you clone this repository, **some files will be missing by design**. Here's what to expect and how to fix it:

### 1. `.streamlit/secrets.toml` ❌ MISSING

**Why:** Contains sensitive API keys (excluded for security)

**How to fix:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` and add your credentials:

**Minimum (Gemini Agent only):**
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-1.5-flash"
```

**Full (with RAG Agent):**
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-1.5-flash"
QDRANT_URL = "https://your-cluster.aws.cloud.qdrant.io"
QDRANT_API_KEY = "your_qdrant_api_key_here"
```

**Get API keys:**
- Gemini: https://makersuite.google.com/app/apikey
- Qdrant: https://cloud.qdrant.io/ (free tier available)

### 2. `data/clinical_docs/` ❌ MISSING

**Why:** Large PDF files (100+ MB) excluded from git

**Impact:**
- ✅ RAG Agent will still work with built-in sample knowledge base
- ⚠️ For full medical literature access, add your own documents

**How to fix (optional):**
```bash
mkdir -p data/clinical_docs
# Add PDF or TXT files with medical content
# Example: ADA diabetes guidelines, research papers, etc.
```

**Note:** The RAG agent automatically indexes documents on first run.

### 3. `data/chroma_db/` ❌ DEPRECATED

**Why:** Old vector database (Phase 1) replaced by Qdrant Cloud (Phase 2)

**Impact:** Not needed - Phase 2 uses cloud-based Qdrant for vector storage

**How to fix:** No action needed (ignore this folder)

### 4. Test/Script Files ❌ EXCLUDED

**Files excluded:** `test_*.py`, `verify_rag.py`, `reindex_documents.py`

**Why:** Development/testing scripts not needed for production

**Impact:** None - application works without them

**How to fix:** No action needed

---

## 🤖 Multi-Agent System

### Gemini Agent 💬
**Purpose:** Conversational health coaching

**Capabilities:**
- General health advice and education
- Lifestyle recommendations (diet, exercise, sleep)
- Motivational support and habit coaching
- Stress management guidance

**Technology:** Google Gemini 1.5 Flash API

**Use when:** You want encouragement, lifestyle tips, or general wellness advice

---

### RAG Agent 📚
**Purpose:** Evidence-based clinical insights

**Capabilities:**
- Citations from ADA 2024 diabetes guidelines
- Research-backed medical recommendations
- Evidence-level transparency (Level A/B/C sources)
- Grounded responses (no speculation)

**Technology Stack:**
- **Vector DB:** Qdrant Cloud (production-grade)
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Search:** Hybrid (semantic + BM25)
- **Re-ranking:** Cross-encoder for precision
- **Generation:** Gemini 1.5 Flash

**Use when:** You need clinical information with citations and evidence

**Example Response:**
```
**Answer:**
The ADA recommends lifestyle interventions including 5-7% weight loss
and 150 minutes of moderate physical activity weekly [Source: American
Diabetes Association, 2024].

**Supporting Evidence:**
The Diabetes Prevention Program demonstrated a 58% risk reduction with
lifestyle modifications [Source: American Diabetes Association, 2024]...

**Recommendations for High Risk:**
1. Target 5-7% weight loss through caloric reduction...
2. Engage in 150 minutes of moderate aerobic activity...

**Medical Disclaimer:**
This is educational information only. Consult healthcare providers...
```

---

## 🔬 Phase 2 RAG Features

### What Makes This RAG System Advanced?

| Feature | Phase 1 (Basic) | Phase 2 (Current) |
|---------|-----------------|-------------------|
| **Search Type** | Semantic only | Hybrid (semantic + BM25) |
| **Relevance Scoring** | Basic similarity | Cross-encoder re-ranking |
| **Query Understanding** | Literal | Medical synonym expansion |
| **Quality Control** | None | Citation validation + filtering |
| **Vector Database** | Local ChromaDB | Qdrant Cloud (scalable) |
| **Documents Indexed** | Sample only | 6,328 chunks (ADA 2024) |
| **Citation Rate** | ~60% | 100% (validated) |

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
Gemini Generation (grounded prompt)
    ↓
Citation Validation (automated quality check)
    ↓
Response with Citations
```

### Performance Metrics

- **Response Time:** ~9.7 seconds average
- **Citation Coverage:** 100% (11-12 citations per response)
- **Relevance Improvement:** +40-60% vs basic semantic search
- **Validation Pass Rate:** 100%
- **Medical Disclaimer:** Included in every response

---

## 📁 Project Structure

```
HealthAgentDiabetic/
├── app_modular.py              # Main application entry point ⭐
├── requirements.txt            # Python dependencies
├── README.md                   # This file (project overview)
├── SETUP.md                    # Detailed setup instructions
├── .gitignore                  # Excluded files
│
├── .streamlit/
│   ├── secrets.toml.example   # Template for API keys
│   └── secrets.toml           # Your secrets (NOT in git) ❌
│
├── agents/                     # Multi-agent system
│   ├── base_agent.py          # Abstract base class
│   ├── gemini_agent.py        # Gemini API agent
│   ├── rag_agent.py           # RAG agent (Phase 2) ⭐
│   ├── retrieval_components.py # Phase 2: Hybrid search, re-ranking
│   └── agent_manager.py       # Agent orchestration
│
├── config/
│   ├── settings.py            # All configuration constants
│   └── document_metadata.py   # Metadata for clinical documents
│
├── models/
│   ├── model_loader.py        # ML model loading
│   └── predictor.py           # Diabetes risk prediction
│
├── ui/
│   ├── chat_interface.py      # Chat UI components
│   ├── forms.py               # Health assessment form
│   ├── visualizations.py      # Charts and graphs
│   └── styles.py              # CSS styling
│
├── utils/
│   └── helpers.py             # Utility functions
│
├── data/
│   └── clinical_docs/         # Medical PDFs (NOT in git) ❌
│
└── model_output2/             # Trained XGBoost model
    ├── xgboost_model.json
    ├── preprocessor.pkl
    ├── optimal_threshold.json
    └── diabetic_averages.json
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[SETUP.md](SETUP.md)** | Complete setup guide with troubleshooting |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | Deploy to Streamlit Cloud |
| **[IMPROVED_RAG_DESIGN.md](IMPROVED_RAG_DESIGN.md)** | RAG architecture details (technical) |
| **[PHASE1_RESULTS.md](PHASE1_RESULTS.md)** | Phase 1 implementation results |
| **[PHASE2_RESULTS.md](PHASE2_RESULTS.md)** | Phase 2 testing and benchmarks |

---

## 🎮 Usage

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
- **Ask Gemini:** "How can I improve my lifestyle?"
- **Ask RAG:** "What does ADA recommend for prediabetes?"
- Switch between agents seamlessly

---

## 🔧 Configuration

Edit `config/settings.py` to customize:

### RAG Settings:
```python
# Search configuration
RAG_USE_HYBRID_SEARCH = True   # Enable semantic + BM25
RAG_USE_RERANKING = True       # Enable cross-encoder
RAG_HYBRID_ALPHA = 0.7         # 70% semantic, 30% keyword

# Quality thresholds
RAG_MIN_RELEVANCE_SCORE = 0.65 # Minimum similarity
RAG_TOP_K = 3                  # Final documents used
RAG_INITIAL_K = 10             # Initial retrieval count
```

### Model Selection:
```python
# For better medical accuracy (requires re-indexing):
RAG_EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
```

---

## 🧪 Testing

### Quick Test (Verify Installation):
```bash
python -c "from models import load_model_components; print('✓ Models OK')"
python -c "from agents import GeminiAgent, RAGAgent; print('✓ Agents OK')"
python -c "from ui import render_chat_interface; print('✓ UI OK')"
```

### Test Gemini Agent:
```python
from agents import GeminiAgent

agent = GeminiAgent()
agent.initialize(api_key="your_key")
response = agent.generate_response("Tell me about diabetes prevention", {})
print(response)
```

### Test RAG Agent:
```python
from agents import RAGAgent

agent = RAGAgent()
agent.initialize()  # Uses secrets.toml
response = agent.generate_response("What are ADA screening guidelines?", {})
print(response)  # Should include [Source: ...] citations
```

---

## 🚢 Deployment

### Streamlit Cloud (Recommended):

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

**See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions**

---

## 🤝 Contributing

Contributions welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup:
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/HealthAgentDiabetic.git
cd HealthAgentDiabetic

# Create branch
git checkout -b feature/my-feature

# Make changes, test, commit
git add .
git commit -m "feat: Add my feature"
git push origin feature/my-feature
```

---

## ⚠️ Disclaimer

**This application is for educational and informational purposes only.**

- ❌ Not a substitute for professional medical advice
- ❌ Not for diagnosis or treatment
- ❌ Not FDA approved or medically certified
- ✅ Always consult healthcare providers for medical decisions
- ✅ Use as a supplementary educational tool only

---

## 📧 Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/PraveenSalapu/HealthAgentDiabetic/issues)
- **Developer:** Praveen Salapu
- **Repository:** https://github.com/PraveenSalapu/HealthAgentDiabetic

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **ADA 2024 Guidelines:** American Diabetes Association Standards of Care
- **BRFSS Dataset:** CDC Behavioral Risk Factor Surveillance System
- **Technology:** Streamlit, LangChain, Qdrant, Google Gemini, XGBoost
- **Medical Knowledge:** All clinical information sourced from peer-reviewed guidelines

---

**Made with ❤️ for better diabetes prevention and health awareness**

**Phase 2 Status:** ✅ Production Ready | Last Updated: November 2024
