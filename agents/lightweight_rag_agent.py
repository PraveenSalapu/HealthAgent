"""
Lightweight RAG Agent using API-based embeddings.
Uses the new Google GenAI SDK (google-genai) and gemini-embedding-001.
"""
import time
from typing import Dict, List, Optional
from pathlib import Path

from .base_agent import BaseAgent
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CHAT_MODEL_INFO,
    CHAT_MODEL_LIGHTWEIGHT_RAG,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
    CLINICAL_DOCS_PATH,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    RAG_MIN_RELEVANCE_SCORE,
    RAG_USE_HYBRID_SEARCH,
    RAG_HYBRID_ALPHA
)
from utils.lightweight_rag import (
    Document,
    GeminiEmbeddings,
    QdrantVectorStore,
    BM25Retriever,
    HybridRetriever,
    chunk_text,
    load_pdf,
    load_text_file
)


class LightweightRAGAgent(BaseAgent):
    """Lightweight RAG agent using the new Google GenAI SDK for cloud deployment."""

    def __init__(self):
        info = CHAT_MODEL_INFO[CHAT_MODEL_LIGHTWEIGHT_RAG]
        super().__init__(name=info["name"], description=info["description"])

        self.client = None
        self.vectorstore = None
        self.embeddings = None
        self.api_key = self._get_api_key()
        self.model_name = GEMINI_MODEL
        self.capabilities = info["capabilities"]

        # Connection status tracking
        self.using_qdrant = False

        # Hybrid search components
        self.bm25_retriever = None
        self.hybrid_retriever = None

    def _get_api_key(self):
        """Get API key from Streamlit secrets or settings."""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
        return GEMINI_API_KEY

    def initialize(self, **kwargs) -> bool:
        """Initialize lightweight RAG agent using the new SDK."""
        try:
            from google import genai

            api_key = kwargs.get("api_key", self.api_key)
            if not api_key:
                raise ValueError("Gemini API key is required.")

            self.client = genai.Client(api_key=api_key)
            self.embeddings = GeminiEmbeddings(api_key=api_key)
            self._initialize_vector_store()

            self.is_initialized = True
            return True
        except Exception as e:
            print("[ERROR] Failed to initialize lightweight RAG agent: " + str(e))
            self.is_initialized = False
            return False

    def _initialize_vector_store(self):
        """Initialize Qdrant vector store with API-based embeddings."""
        from qdrant_client import QdrantClient
        if not QDRANT_URL or not QDRANT_API_KEY:
            print("[WARNING] Qdrant credentials not found. Using sample knowledge base.")
            self._create_sample_knowledge_base()
            return
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == QDRANT_COLLECTION_NAME_LIGHTWEIGHT for c in collections)

            if collection_exists:
                self.vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
                    embeddings=self.embeddings
                )
                self.using_qdrant = True
                print("[OK] Connected to existing collection: " + QDRANT_COLLECTION_NAME_LIGHTWEIGHT)
            else:
                docs_path = Path(CLINICAL_DOCS_PATH)
                if docs_path.exists() and any(docs_path.iterdir()):
                    self.index_documents()
                    self.using_qdrant = True
                else:
                    self._create_sample_knowledge_base()
        except Exception as e:
            print("[ERROR] Failed to connect to Qdrant: " + str(e))
            self._create_sample_knowledge_base()

    def index_documents(self, batch_size: int = 50) -> str:
        """Index clinical documents into Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        try:
            documents = []
            for txt_file in Path(CLINICAL_DOCS_PATH).glob("**/*.txt"):
                text = load_text_file(str(txt_file))
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": txt_file.name, "file_type": "txt"}
                    ))

            for pdf_file in Path(CLINICAL_DOCS_PATH).glob("**/*.pdf"):
                text = load_pdf(str(pdf_file))
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_file.name, "file_type": "pdf"}
                    ))

            if not documents:
                return "No documents found in " + CLINICAL_DOCS_PATH

            chunked_docs = []
            for doc in documents:
                chunks = chunk_text(doc.page_content, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
                for chunk in chunks:
                    chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            test_emb = self.embeddings.embed_query("test")
            dimension = len(test_emb)
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == QDRANT_COLLECTION_NAME_LIGHTWEIGHT for c in collections):
                self.qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT)

            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            self.vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
                embeddings=self.embeddings
            )

            for i in range(0, len(chunked_docs), batch_size):
                batch = chunked_docs[i:i + batch_size]
                self.vectorstore.add_documents(batch)
                time.sleep(0.5)

            if RAG_USE_HYBRID_SEARCH:
                self.bm25_retriever = BM25Retriever(chunked_docs)
                self.hybrid_retriever = HybridRetriever(
                    vectorstore=self.vectorstore,
                    bm25_retriever=self.bm25_retriever,
                    alpha=RAG_HYBRID_ALPHA
                )
            self.using_qdrant = True
            return "Successfully indexed " + str(len(chunked_docs)) + " chunks."
        except Exception as e:
            print("[ERROR] Indexing failed: " + str(e))
            return "Error: " + str(e)

    def _create_sample_knowledge_base(self):
        """Create sample fallback knowledge base."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        import tempfile

        sample_docs = [
            Document(
                page_content="Diabetes Prevention: Lifestyle interventions can reduce risk by 58%. Strategies: 5-7% weight loss, 150 min activity/week.",
                metadata={"source": "Sample Guidelines"}
            )
        ]

        temp_dir = tempfile.mkdtemp()
        self.qdrant_client = QdrantClient(path=temp_dir)
        dimension = len(self.embeddings.embed_query("test"))

        self.qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
            embeddings=self.embeddings
        )
        self.vectorstore.add_documents(sample_docs)
        self.using_qdrant = False
        print("[OK] Sample KB active (fallback).")

    def generate_response(self, message: str, context: Dict, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate RAG response using the new SDK."""
        if not self.is_initialized or not self.vectorstore:
            return self._generate_fallback_response(message, context)
        try:
            if RAG_USE_HYBRID_SEARCH and self.hybrid_retriever:
                docs_with_scores = self.hybrid_retriever.retrieve(message, k=RAG_TOP_K)
            else:
                docs_with_scores = self.vectorstore.similarity_search_with_score(message, k=RAG_TOP_K)

            filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= RAG_MIN_RELEVANCE_SCORE]
            if not filtered_docs:
                return self._generate_fallback_response(message, context)

            retrieved_context = self._format_retrieved_docs([doc for doc, _ in filtered_docs])
            prompt = self._build_rag_prompt(message, context, retrieved_context, conversation_history)

            if self.client:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            return self._generate_fallback_response(message, context)
        except Exception as e:
            print("[ERROR] RAG generation failed: " + str(e))
            return self._generate_fallback_response(message, context)

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            entry = "[Source " + str(i) + "] " + source + "\n" + doc.page_content + "\n"
            formatted.append(entry)
        return "\n".join(formatted)

    def _build_rag_prompt(self, message: str, context: Dict, retrieved_context: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        prob = context.get("probability", 0)
        risk = context.get("risk_level", "Unknown")
        summary = context.get("profile_summary", "No data")
        prob_str = "{:.1f}".format(prob)

        return (
            "You are a clinical diabetes specialist.\n"
            "PATIENT PROFILE: Risk Level " + risk + " (" + prob_str + "%), Metrics: " + summary + "\n"
            "CLINICAL EVIDENCE: " + retrieved_context + "\n"
            "QUESTION: " + message + "\n"
            "INSTRUCTIONS:\n"
            "1. Use ONLY the clinical evidence provided.\n"
            "2. Include citations [1], [2].\n"
            "3. End with a **References:** section.\n"
            "4. Include a medical disclaimer."
        )

    def _generate_fallback_response(self, message: str, context: Dict) -> str:
        prob = context.get("probability", 0)
        risk = context.get("risk_level", "Unknown")
        db_status = "Connected" if self.using_qdrant else "Fallback (Sample KB)"
        prob_str = "{:.1f}".format(prob)
        return (
            "**Clinical Assessment:** Risk " + prob_str + "% (" + risk + ").\n"
            "Database Status: " + db_status + ".\n"
            "Please consult a professional. Clinical evidence retrieval currently limited."
        )

    def is_ready(self) -> bool:
        return self.is_initialized and self.vectorstore is not None

    def get_capabilities(self) -> List[str]:
        return self.capabilities
