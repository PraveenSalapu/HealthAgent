"""
Lightweight RAG Agent using API-based embeddings.
Parallel implementation to rag_agent.py using Gemini API instead of local models.
"""

from typing import Dict, List, Optional
from pathlib import Path
import google.generativeai as genai

from .base_agent import BaseAgent
from config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL, CHAT_MODEL_INFO, CHAT_MODEL_LIGHTWEIGHT_RAG,
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
    CLINICAL_DOCS_PATH, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
    RAG_TOP_K, RAG_MIN_RELEVANCE_SCORE, RAG_USE_HYBRID_SEARCH, RAG_HYBRID_ALPHA
)

from utils.lightweight_rag import (
    Document, GeminiEmbeddings, QdrantVectorStore,
    BM25Retriever, HybridRetriever, chunk_text, load_pdf, load_text_file
)


class LightweightRAGAgent(BaseAgent):
    """Lightweight RAG agent using API-based embeddings for cloud deployment."""

    def __init__(self):
        info = CHAT_MODEL_INFO[CHAT_MODEL_LIGHTWEIGHT_RAG]
        super().__init__(name=info["name"], description=info["description"])

        self.model = None
        self.vectorstore = None
        self.embeddings = None
        self.api_key = self._get_api_key()
        self.model_name = GEMINI_MODEL
        self.capabilities = info["capabilities"]

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
        """Initialize lightweight RAG agent."""
        try:
            # Initialize Gemini for generation
            api_key = kwargs.get("api_key", self.api_key)
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)

            # Initialize embeddings (API-based, no local model)
            self.embeddings = GeminiEmbeddings(api_key=api_key)

            # Initialize vector store
            self._initialize_vector_store()

            self.is_initialized = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize lightweight RAG agent: {e}")
            self.is_initialized = False
            return False

    def _initialize_vector_store(self):
        """Initialize Qdrant vector store with API-based embeddings."""
        from qdrant_client import QdrantClient

        # Check for Qdrant credentials
        if not QDRANT_URL or not QDRANT_API_KEY:
            print("[WARNING] Qdrant credentials not found. Using sample knowledge base.")
            self._create_sample_knowledge_base()
            return

        try:
            # Connect to Qdrant Cloud
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == QDRANT_COLLECTION_NAME_LIGHTWEIGHT for c in collections)

            if collection_exists:
                # Use existing collection
                self.vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
                    embeddings=self.embeddings
                )
                print(f"[OK] Connected to existing collection: {QDRANT_COLLECTION_NAME_LIGHTWEIGHT}")
            else:
                # Create new collection from clinical docs
                docs_path = Path(CLINICAL_DOCS_PATH)
                if docs_path.exists() and any(docs_path.iterdir()):
                    self.index_documents()
                else:
                    self._create_sample_knowledge_base()

        except Exception as e:
            print(f"[ERROR] Failed to connect to Qdrant: {e}")
            self._create_sample_knowledge_base()

    def index_documents(self, batch_size: int = 50) -> str:
        """Index clinical documents into Qdrant with batching and retry logic."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        import time

        try:
            documents = []

            # Load text files
            print("[INFO] Loading text files...")
            for txt_file in Path(CLINICAL_DOCS_PATH).glob("**/*.txt"):
                text = load_text_file(str(txt_file))
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": txt_file.name, "file_type": "txt"}
                    ))

            # Load PDF files
            print("[INFO] Loading PDF files...")
            for pdf_file in Path(CLINICAL_DOCS_PATH).glob("**/*.pdf"):
                text = load_pdf(str(pdf_file))
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_file.name, "file_type": "pdf"}
                    ))

            if not documents:
                return f"No documents found in {CLINICAL_DOCS_PATH}"

            print(f"[INFO] Loaded {len(documents)} documents")

            # Chunk documents
            print("[INFO] Chunking documents...")
            chunked_docs = []
            for doc in documents:
                chunks = chunk_text(doc.page_content, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
                for chunk in chunks:
                    chunked_docs.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ))

            print(f"[INFO] Created {len(chunked_docs)} chunks")

            # Create Qdrant collection
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

            # Get embedding dimension (Gemini embeddings are 768-dim)
            test_emb = self.embeddings.embed_query("test")
            dimension = len(test_emb)

            # Check if collection exists, delete if it does (for reindexing)
            collections = self.qdrant_client.get_collections().collections
            if any(c.name == QDRANT_COLLECTION_NAME_LIGHTWEIGHT for c in collections):
                print(f"[INFO] Deleting existing collection: {QDRANT_COLLECTION_NAME_LIGHTWEIGHT}")
                self.qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT)

            print(f"[INFO] Creating collection with dimension {dimension}")
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )

            # Initialize vector store
            self.vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=QDRANT_COLLECTION_NAME_LIGHTWEIGHT,
                embeddings=self.embeddings
            )

            # Add documents in batches with retry logic
            print(f"[INFO] Uploading {len(chunked_docs)} chunks in batches of {batch_size}...")
            total_uploaded = 0
            for i in range(0, len(chunked_docs), batch_size):
                batch = chunked_docs[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(chunked_docs) + batch_size - 1) // batch_size

                # Retry logic for each batch
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"[INFO] Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                        self.vectorstore.add_documents(batch)
                        total_uploaded += len(batch)
                        print(f"[OK] Batch {batch_num}/{total_batches} uploaded ({total_uploaded}/{len(chunked_docs)} total)")
                        time.sleep(1)  # Rate limiting
                        break
                    except Exception as batch_error:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            print(f"[WARN] Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {batch_error}")
                            print(f"[INFO] Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            raise Exception(f"Batch {batch_num} failed after {max_retries} attempts: {batch_error}")

            # Build BM25 index for hybrid search
            if RAG_USE_HYBRID_SEARCH:
                print("[INFO] Building BM25 index...")
                self.bm25_retriever = BM25Retriever(chunked_docs)
                self.hybrid_retriever = HybridRetriever(
                    vectorstore=self.vectorstore,
                    bm25_retriever=self.bm25_retriever,
                    alpha=RAG_HYBRID_ALPHA
                )

            return f"Successfully indexed {len(chunked_docs)} chunks from {len(documents)} documents"

        except Exception as e:
            print(f"[ERROR] Failed to index documents: {e}")
            return f"Error indexing documents: {str(e)}"

    def _create_sample_knowledge_base(self):
        """Create sample diabetes knowledge base."""
        sample_docs = [
            Document(
                page_content="Diabetes Prevention: The Diabetes Prevention Program showed that lifestyle interventions can reduce diabetes risk by 58%. Key strategies include 5-7% weight loss and 150 minutes of moderate physical activity weekly.",
                metadata={"source": "Sample Guidelines", "topic": "Prevention"}
            ),
            Document(
                page_content="Screening Recommendations: Adults with BMI >=25 and one or more risk factors should be screened for diabetes. Screening tests include HbA1c >=6.5%, fasting plasma glucose >=126 mg/dL.",
                metadata={"source": "ADA Guidelines", "topic": "Screening"}
            ),
        ]

        # Create in-memory vector store with sample docs
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        import tempfile

        # Use local in-memory Qdrant
        temp_dir = tempfile.mkdtemp()
        self.qdrant_client = QdrantClient(path=temp_dir)

        test_emb = self.embeddings.embed_query("test")
        dimension = len(test_emb)

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

        print("[OK] Created sample knowledge base (local mode)")

    def generate_response(
        self,
        message: str,
        context: Dict,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate RAG response using retrieved clinical context."""
        if not self.is_initialized or not self.vectorstore:
            return self._generate_fallback_response(message, context)

        try:
            # Retrieve relevant documents
            if RAG_USE_HYBRID_SEARCH and self.hybrid_retriever:
                docs_with_scores = self.hybrid_retriever.retrieve(message, k=RAG_TOP_K)
            else:
                docs_with_scores = self.vectorstore.similarity_search_with_score(message, k=RAG_TOP_K)

            # Filter by relevance
            filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= RAG_MIN_RELEVANCE_SCORE]

            if not filtered_docs:
                return self._generate_fallback_response(message, context)

            # Format retrieved context
            retrieved_context = self._format_retrieved_docs([doc for doc, _ in filtered_docs])

            # Build prompt
            prompt = self._build_rag_prompt(message, context, retrieved_context, conversation_history)

            # Generate response
            if self.model:
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return self._generate_fallback_response(message, context)

        except Exception as e:
            print(f"[ERROR] Failed to generate RAG response: {e}")
            return self._generate_fallback_response(message, context)

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for prompt."""
        if not docs:
            return "No relevant clinical information found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source {i}] {source}\n{doc.page_content}\n")

        return "\n".join(formatted)

    def _build_rag_prompt(
        self,
        message: str,
        context: Dict,
        retrieved_context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build RAG prompt with retrieved clinical evidence."""
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        profile_summary = context.get("profile_summary", "No data available")

        return f"""You are a clinical diabetes specialist providing evidence-based medical guidance.

PATIENT PROFILE:
- Risk Level: {risk_level} ({probability:.1f}%)
- Clinical Metrics: {profile_summary}

RETRIEVED CLINICAL EVIDENCE:
{retrieved_context}

PATIENT QUESTION:
{message}

INSTRUCTIONS:
1. Answer using ONLY the retrieved clinical evidence above
2. Include numbered citations [1], [2] for each claim
3. Use clinical terminology while remaining accessible
4. If evidence does not address the question, state this clearly
5. Include medical disclaimer at the end
6. REQUIRED: End your response with a "**References:**" section listing all sources used with their citation numbers

RESPONSE FORMAT:
[Your evidence-based answer with inline citations [1], [2], etc.]

**Medical Disclaimer:** [Standard disclaimer]

**References:**
[1] [Source name from evidence]
[2] [Source name from evidence]

Provide a concise, evidence-based response following this exact format."""

    def _generate_fallback_response(self, message: str, context: Dict) -> str:
        """Generate generic response when RAG is unavailable."""
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")

        return f"""**Clinical Assessment:**
Based on your {probability:.1f}% risk probability ({risk_level} risk level), I recommend consulting with your healthcare provider for personalized guidance.

**General Recommendations:**
1. Lifestyle modifications (diet and exercise)
2. Regular screening and monitoring
3. Risk factor management

**Note:** RAG retrieval unavailable. Please consult a healthcare professional for specific clinical guidance.

**Medical Disclaimer:**
This information is for educational purposes only and does not constitute medical advice."""

    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return self.capabilities
