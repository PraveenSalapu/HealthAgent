"""
Configuration settings for the Health AI Chatbot application.

This module centralizes all configuration constants including:
- Model paths and thresholds
- Feature configurations
- Agent definitions
- UI styling constants
"""

import os

# ============================================================================
# MODEL PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model_output")

MODEL_JSON_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "optimal_threshold.json")
AVERAGES_PATH = os.path.join(MODEL_DIR, "diabetic_averages.json")

# ============================================================================
# FEATURE CONFIGURATIONS
# ============================================================================

# Age category mapping (for display purposes)
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+"
}

FEATURE_CONFIGS = {
    "GenHlth": {"type": "select", "options": [1, 2, 3, 4, 5],
                "labels": ["Excellent", "Very Good", "Good", "Fair", "Poor"]},
    "HighBP": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "DiffWalk": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "BMI": {"type": "number", "min": 10.0, "max": 70.0, "default": 25.0, "step": 0.1},
    "HighChol": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "Age": {"type": "select",
            "options": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "labels": ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]},
    "HeartDiseaseorAttack": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "PhysHlth": {"type": "number", "min": 0, "max": 30, "default": 0, "step": 1,
                 "help": "Days of poor physical health in past 30 days"},
    "Income": {"type": "select",
               "options": [1, 2, 3, 4, 5, 6, 7, 8],
               "labels": ["<$10k", "$10k-15k", "$15k-20k", "$20k-25k", "$25k-35k", "$35k-50k", "$50k-75k", ">$75k"]},
    "Education": {"type": "select",
                  "options": [1, 2, 3, 4, 5, 6],
                  "labels": ["Never attended", "Grades 1-8", "Grades 9-11", "High school/GED", "Some college", "College grad"]},
    "PhysActivity": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]}
}

# Feature display names
FEATURE_NAMES = {
    "GenHlth": "General Health",
    "HighBP": "High Blood Pressure",
    "DiffWalk": "Difficulty Walking",
    "BMI": "Body Mass Index",
    "HighChol": "High Cholesterol",
    "Age": "Age Range",
    "HeartDiseaseorAttack": "Heart Disease or Attack History",
    "PhysHlth": "Physical Health (poor days/month)",
    "Income": "Income Level",
    "Education": "Education Level",
    "PhysActivity": "Physical Activity (last 30 days)"
}

FEATURE_INFO = {
    "GenHlth": "1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor.",
    "HighBP": "0=No history of high blood pressure, 1=Diagnosed with high blood pressure.",
    "DiffWalk": "0=No difficulty walking or climbing stairs, 1=Difficulty present.",
    "BMI": "Body Mass Index (weight/height^2). Typical healthy adult range is 18.5-24.9.",
    "HighChol": "0=No high cholesterol diagnosis, 1=Diagnosed with high cholesterol.",
    "Age": "Select your age range from the dropdown.",
    "HeartDiseaseorAttack": "0=No prior heart disease/attack, 1=History of heart disease or heart attack.",
    "PhysHlth": "Number of poor physical health days in the past 30 days (0-30).",
    "Income": "1=<10K, 2=10-15K, 3=15-20K, 4=20-25K, 5=25-35K, 6=35-50K, 7=50-75K, 8=>75K.",
    "Education": "1=Never attended/Kindergarten only, 2=Grades 1-8, 3=Grades 9-11, 4=High school/GED, 5=Some college, 6=College graduate.",
    "PhysActivity": "0=No physical activity in past 30 days, 1=Performed physical activity or exercise."
}

# ============================================================================
# FORM SECTIONS
# ============================================================================

FORM_SECTIONS = [
    {
        "title": "Wellness Snapshot",
        "description": "Capture how you feel day to day alongside core wellness indicators.",
        "features": ["GenHlth", "BMI", "PhysHlth", "PhysActivity"],
    },
    {
        "title": "Cardiometabolic History",
        "description": "Document clinical risk factors that influence diabetes risk.",
        "features": ["HighBP", "HighChol", "HeartDiseaseorAttack", "DiffWalk"],
    },
    {
        "title": "Lifestyle & Support",
        "description": "Share information that shapes access to care and daily habits.",
        "features": ["Age", "Education", "Income"],
    },
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

RADAR_FEATURES = ["GenHlth", "BMI", "PhysHlth", "PhysActivity", "HighBP", "HighChol"]

IDEAL_PROFILE = {
    "GenHlth": 1,
    "BMI": 22.0,
    "PhysHlth": 0,
    "PhysActivity": 1,
    "HighBP": 0,
    "HighChol": 0,
}

# ============================================================================
# AGENT DEFINITIONS (Legacy - for backward compatibility)
# ============================================================================

AGENT_DEFINITIONS = {
    "coach": {
        "name": "Health Coach",
        "avatar": "🧭",
        "keywords": ["motivation", "routine", "habits", "exercise", "stress", "sleep", "plan"],
        "system_prompt": (
            "You are a supportive personal health coach helping someone interpret their diabetes risk results.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Focus on empowering next steps, mindset shifts, and habit coaching. Keep explanations warm and practical."
        ),
        "fallback_focus": (
            "Focus on motivation, small habit changes, and balancing lifestyle pillars like sleep, stress, and movement."
        ),
    },
    "doctor": {
        "name": "Clinical Advisor",
        "avatar": "🩺",
        "keywords": ["symptom", "medication", "diagnosis", "doctor", "treatment", "blood test", "lab"],
        "system_prompt": (
            "You are a cautious clinician interpreting model findings without giving direct medical orders.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Highlight questions to raise with a physician, relevant screenings or labs, and safety precautions."
        ),
        "fallback_focus": (
            "Suggest evidence-based checkpoints to review with a doctor and emphasize follow-up care."
        ),
    },
    "dietician": {
        "name": "Dietician",
        "avatar": "🥗",
        "keywords": ["diet", "meal", "food", "nutrition", "carb", "protein", "recipe", "eat"],
        "system_prompt": (
            "You are a registered dietician tailoring nutrition advice to the person's risk profile.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Offer practical meal planning, portion guidance, and nutrient strategies aligned with diabetes prevention."
        ),
        "fallback_focus": (
            "Share carbohydrate awareness tips, balanced meals, hydration guidance, and mindful eating strategies."
        ),
    },
    "scheduler": {
        "name": "Care Concierge",
        "avatar": "📅",
        "keywords": ["appointment", "schedule", "book", "nearby", "consultant", "specialist", "clinic"],
        "system_prompt": (
            "You are a care coordinator helping the user plan follow-up appointments.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Recommend how to choose local clinicians, what to bring to visits, and how to manage scheduling."
        ),
        "fallback_focus": (
            "Guide the user to gather insurance details, shortlist providers, prepare records, and set reminders."
        ),
    },
}

SCHEDULER_REQUIRED_FIELDS = [
    ("location", "To find nearby specialists, what city or ZIP code works best for you?"),
    ("specialty", "What type of clinician would you like to see (e.g., primary care, endocrinologist)?"),
    ("date", "When would you like the appointment? Share a specific date or a window such as 'next week'."),
]

# ============================================================================
# DEFAULT VALUES
# ============================================================================

# Default averages (fallback if file not found or missing features)
# Based on BRFSS 2015 diabetic population averages
DEFAULT_DIABETIC_AVERAGES = {
    "GenHlth": 3.29,
    "HighBP": 0.75,
    "DiffWalk": 0.37,
    "BMI": 31.94,
    "HighChol": 0.67,
    "Age": 9.38,
    "HeartDiseaseorAttack": 0.22,
    "PhysHlth": 7.95,
    "Income": 5.21,
    "Education": 4.75,
    "PhysActivity": 0.63
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

def get_api_key():
    """Get API key from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

def get_model_name():
    """Get model name from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize with environment variables, will be updated at runtime
GEMINI_API_KEY = get_api_key()
GEMINI_MODEL = get_model_name()

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

def get_qdrant_url():
    """Get Qdrant URL from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "QDRANT_URL" in st.secrets:
            return st.secrets["QDRANT_URL"]
    except Exception:
        pass
    return os.getenv("QDRANT_URL", "")

def get_qdrant_api_key():
    """Get Qdrant API key from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "QDRANT_API_KEY" in st.secrets:
            return st.secrets["QDRANT_API_KEY"]
    except Exception:
        pass
    return os.getenv("QDRANT_API_KEY", "")

# Vector store configuration
RAG_VECTOR_STORE = "qdrant"  # Options: "qdrant", "chromadb", "faiss"

# Embedding model options (Phase 2: Can use medical-specific embeddings)
RAG_EMBEDDING_OPTIONS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",  # Fast, general-purpose (384 dim)
    "medical": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",  # Medical-specific (768 dim)
    "balanced": "sentence-transformers/all-mpnet-base-v2",  # Better general model (768 dim)
}

# Current embedding model (change to "medical" or "balanced" for better medical accuracy)
RAG_EMBEDDING_MODEL = RAG_EMBEDDING_OPTIONS["general"]  # Default: fast general model

# Chunking configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200

# Retrieval configuration
RAG_TOP_K = 3  # Number of documents to retrieve (final count after filtering/re-ranking)
RAG_INITIAL_K = 10  # Initial retrieval count (before re-ranking)

# Phase 2: Advanced retrieval settings
RAG_USE_HYBRID_SEARCH = True  # Enable semantic + keyword (BM25) hybrid search
RAG_HYBRID_ALPHA = 0.7  # Weight for semantic search (0=pure keyword, 1=pure semantic, 0.7=70% semantic)
RAG_USE_RERANKING = True  # Enable cross-encoder re-ranking
RAG_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for re-ranking
RAG_MIN_RELEVANCE_SCORE = 0.50  # Minimum similarity score threshold (lowered from 0.65 to reduce fallback rate)

# Local paths (for document loading)
CLINICAL_DOCS_PATH = "data/clinical_docs"

# Qdrant configuration
QDRANT_URL = get_qdrant_url()
QDRANT_API_KEY = get_qdrant_api_key()
QDRANT_COLLECTION_NAME = "clinical_docs"
QDRANT_COLLECTION_NAME_LIGHTWEIGHT = "clinical_docs_lightweight"

# Legacy ChromaDB path (for migration reference)
CHROMA_PERSIST_DIR = "data/chroma_db"

# ============================================================================
# CHAT MODEL TYPES
# ============================================================================

CHAT_MODEL_GEMINI = "gemini"
CHAT_MODEL_RAG = "rag"
CHAT_MODEL_LIGHTWEIGHT_RAG = "lightweight_rag"

CHAT_MODEL_INFO = {
    CHAT_MODEL_GEMINI: {
        "name": "Gemini Health Coach",
        "icon": "💫",
        "description": "AI-powered health insights & personalized lifestyle coaching",
        "capabilities": [
            "🏃 General health & wellness advice",
            "🥗 Lifestyle & nutrition recommendations",
            "💪 Motivational support & accountability",
            "🎯 Personalized wellness coaching"
        ]
    },
    # CHAT_MODEL_RAG: {
    #     "name": "RAG Agent (LangChain)",
    #     "icon": "📚",
    #     "description": "Clinical insights from medical literature (heavy dependencies)",
    #     "capabilities": [
    #         "Evidence-based clinical information",
    #         "Medical literature references",
    #         "Research-backed recommendations",
    #         "Source citations"
    #     ]
    # },
    CHAT_MODEL_LIGHTWEIGHT_RAG: {
        "name": "Clinical Research Assistant",
        "icon": "🔬",
        "description": "Evidence-based clinical insights from medical research & guidelines",
        "capabilities": [
            "🩺 Evidence-based clinical information",
            "📊 Medical literature & research references",
            "✅ Guidelines-backed recommendations",
            "📖 Source citations & studies",
            "⚡ Cloud-optimized performance"
        ]
    }
}
