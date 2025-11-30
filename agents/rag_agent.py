"""
RAG Agent for clinical insights from medical literature.

This agent uses Retrieval-Augmented Generation to provide:
- Evidence-based clinical information
- Medical literature references
- Research-backed recommendations
- Source citations
"""

from typing import Dict, List, Optional
import os
from pathlib import Path

import google.generativeai as genai

from .base_agent import BaseAgent
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CHAT_MODEL_INFO,
    CHAT_MODEL_RAG,
    RAG_VECTOR_STORE,
    RAG_EMBEDDING_MODEL,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    RAG_INITIAL_K,
    RAG_USE_HYBRID_SEARCH,
    RAG_HYBRID_ALPHA,
    RAG_USE_RERANKING,
    RAG_RERANKER_MODEL,
    RAG_MIN_RELEVANCE_SCORE,
    CLINICAL_DOCS_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
)
from config.document_metadata import get_metadata_for_file, get_evidence_rank, is_recent_guideline

# Phase 2: Import advanced retrieval components
try:
    from .retrieval_components import (
        BM25Retriever,
        HybridRetriever,
    )
    PHASE2_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Phase 2 components not available: {e}")
    PHASE2_AVAILABLE = False


class RAGAgent(BaseAgent):
    """Agent that provides clinical insights using RAG with medical literature."""
    
    def __init__(self):
        """Initialize RAG agent."""
        import os
        info = CHAT_MODEL_INFO[CHAT_MODEL_RAG]
        super().__init__(
            name=info["name"],
            description=info["description"]
        )
        self.model = None
        self.vectorstore = None
        self.embeddings = None
        # Try to get API key from Streamlit secrets first, then settings, then environment
        self.api_key = self._get_api_key()
        self.model_name = GEMINI_MODEL
        self.capabilities = info["capabilities"]
        self.docs_loaded = False

        # Phase 2: Advanced retrieval components
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.use_phase2 = PHASE2_AVAILABLE

    def _get_api_key(self):
        """Get API key from various sources."""
        import os
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
        # Fall back to settings constant or environment variable
        return GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize RAG agent with vector store and embeddings.
        
        Args:
            **kwargs: Optional configuration overrides
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize Gemini for generation
            api_key = kwargs.get("api_key", self.api_key)
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)
            
            # Initialize embeddings and vector store
            self._initialize_vector_store()

            # Phase 2: Initialize advanced retrieval components
            if self.use_phase2:
                self._initialize_phase2_components()

            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing RAG agent: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_vector_store(self):
        """Initialize Qdrant vector store with embeddings."""
        try:
            # Initialize embeddings
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                except ImportError:
                    from langchain.embeddings import HuggingFaceEmbeddings
            
            try:
                from langchain_qdrant import QdrantVectorStore
            except ImportError:
                try:
                    from langchain_qdrant import Qdrant as QdrantVectorStore
                except ImportError:
                    try:
                        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
                    except ImportError:
                        from langchain.vectorstores import Qdrant as QdrantVectorStore

            from qdrant_client import QdrantClient

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=RAG_EMBEDDING_MODEL
            )

            # Check if Qdrant credentials are available
            if not QDRANT_URL or not QDRANT_API_KEY:
                print("[WARNING] Qdrant credentials not found. Using local FAISS vector store.")
                print("Set QDRANT_URL and QDRANT_API_KEY in Streamlit secrets or environment variables for cloud storage.")
                # Try to load local documents, fallback to sample data if not available
                docs_path = Path(CLINICAL_DOCS_PATH)
                if docs_path.exists() and any(docs_path.iterdir()):
                    self._load_and_index_documents_faiss()
                else:
                    self._create_sample_knowledge_base()
                return

            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )

            # Check if collection already exists
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections)

                if collection_exists:
                    # Load existing vector store
                    self.vectorstore = QdrantVectorStore(
                        client=self.qdrant_client,
                        collection_name=QDRANT_COLLECTION_NAME,
                        embedding=self.embeddings,
                    )
                    self.docs_loaded = True
                    print(f"[OK] Connected to existing Qdrant collection: {QDRANT_COLLECTION_NAME}")
                else:
                    # Create new collection and load documents
                    docs_path = Path(CLINICAL_DOCS_PATH)
                    if docs_path.exists() and any(docs_path.iterdir()):
                        status = self.index_documents()
                        print(f"[OK] {status}")
                    else:
                        # Create with sample diabetes information
                        self._create_sample_knowledge_base()
            except Exception as e:
                print(f"[WARNING] Could not connect to Qdrant: {e}")
                print("Creating sample knowledge base locally.")
                self._create_sample_knowledge_base()

        except ImportError as e:
            print(f"[WARNING] RAG dependencies not installed: {e}")
            print("Install with: pip install langchain qdrant-client sentence-transformers")
            self.vectorstore = None
        except Exception as e:
            print(f"[ERROR] Error initializing vector store: {e}")
            self.vectorstore = None
    
    def index_documents(self) -> str:
        """
        Load and index documents from clinical_docs directory into Qdrant.
        
        Returns:
            str: Status message
        """
        try:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
        except ImportError:
            from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
            
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError:
            try:
                from langchain_qdrant import Qdrant as QdrantVectorStore
            except ImportError:
                try:
                    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
                except ImportError:
                    from langchain.vectorstores import Qdrant as QdrantVectorStore

        from qdrant_client import QdrantClient

        try:
            from pathlib import Path
            import os

            documents = []

            # Load text documents with metadata
            txt_files = list(Path(CLINICAL_DOCS_PATH).glob("**/*.txt"))
            for txt_file in txt_files:
                try:
                    loader = TextLoader(str(txt_file))
                    docs = loader.load()

                    # Add rich metadata to each document
                    filename = os.path.basename(str(txt_file))
                    metadata = get_metadata_for_file(filename)

                    for doc in docs:
                        doc.metadata.update(metadata)
                        doc.metadata['filename'] = filename
                        doc.metadata['file_path'] = str(txt_file)

                    documents.extend(docs)
                    print(f"[OK] Loaded {filename} with metadata: {metadata.get('title', 'Unknown')}")
                except Exception as e:
                    print(f"[WARNING] Could not load {txt_file}: {e}")

            # Load PDF documents with metadata
            pdf_files = list(Path(CLINICAL_DOCS_PATH).glob("**/*.pdf"))
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()

                    # Add rich metadata to each document
                    filename = os.path.basename(str(pdf_file))
                    metadata = get_metadata_for_file(filename)

                    for doc in docs:
                        doc.metadata.update(metadata)
                        doc.metadata['filename'] = filename
                        doc.metadata['file_path'] = str(pdf_file)

                    documents.extend(docs)
                    print(f"[OK] Loaded {filename} ({len(docs)} pages) - {metadata.get('title', 'Unknown')}")
                except Exception as e:
                    print(f"[WARNING] Could not load {pdf_file}: {e}")

            if not documents:
                return f"No documents found in {CLINICAL_DOCS_PATH}"

            print(f"\n[OK] Total documents loaded: {len(documents)}")

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)

            # Phase 2: Build BM25 index for hybrid search
            if self.use_phase2:
                self._build_bm25_index(splits)

            # Create Qdrant client
            self.qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            
            # Check if collection exists
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections)
            except Exception:
                collection_exists = False

            if collection_exists:
                # Append to existing
                self.vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=QDRANT_COLLECTION_NAME,
                    embedding=self.embeddings,
                )
                if splits:
                    self.vectorstore.add_documents(splits)
                    msg = f"Successfully added {len(splits)} chunks to existing collection."
            else:
                # Create new collection explicitly
                from qdrant_client.models import VectorParams, Distance
                
                # Get embedding dimension (384 for all-MiniLM-L6-v2)
                # We try to get it dynamically, fallback to 384
                try:
                    # For HuggingFaceEmbeddings
                    test_emb = self.embeddings.embed_query("test")
                    dimension = len(test_emb)
                except Exception:
                    dimension = 384

                self.qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
                
                self.vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=QDRANT_COLLECTION_NAME,
                    embedding=self.embeddings,
                )

                if splits:
                    self.vectorstore.add_documents(splits)

                msg = f"Successfully created collection and indexed {len(splits)} chunks."
            
            self.docs_loaded = True
            return msg
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Error loading documents: {e}")
            return f"Error indexing documents: {str(e)}"

    def _load_and_index_documents_faiss(self):
        """Load and index documents from clinical_docs directory into FAISS (local mode)."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            from langchain.vectorstores import FAISS

        try:
            documents = []

            # Try to load PDF files
            try:
                try:
                    from langchain_community.document_loaders import PyPDFDirectoryLoader
                except ImportError:
                    from langchain.document_loaders import PyPDFDirectoryLoader
                    
                pdf_loader = PyPDFDirectoryLoader(CLINICAL_DOCS_PATH)
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                print(f"[OK] Loaded {len(pdf_docs)} PDF documents")
            except ImportError:
                print("[WARNING] pypdf not installed. Install with: pip install pypdf")
            except Exception as e:
                print(f"[WARNING] Could not load PDFs: {e}")

            # Try to load text files
            try:
                try:
                    from langchain_community.document_loaders import DirectoryLoader, TextLoader
                except ImportError:
                    from langchain.document_loaders import DirectoryLoader, TextLoader
                    
                txt_loader = DirectoryLoader(
                    CLINICAL_DOCS_PATH,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                txt_docs = txt_loader.load()
                documents.extend(txt_docs)
                if txt_docs:
                    print(f"[OK] Loaded {len(txt_docs)} text documents")
            except Exception as e:
                print(f"[WARNING] Could not load text files: {e}")

            if not documents:
                print(f"[WARNING] No documents found in {CLINICAL_DOCS_PATH}")
                self._create_sample_knowledge_base()
                return

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)

            # Create FAISS vector store (in-memory)
            self.vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings,
            )
            self.docs_loaded = True
            print(f"[OK] Indexed {len(splits)} document chunks from {len(documents)} documents using FAISS (local mode)")
        except Exception as e:
            print(f"[ERROR] Error loading documents: {e}")
            print(f"[WARNING] Falling back to sample knowledge base")
            self._create_sample_knowledge_base()
    
    def _create_sample_knowledge_base(self):
        """Create sample knowledge base with diabetes information."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.schema import Document

        # Sample clinical information about diabetes
        sample_docs = [
            Document(
                page_content="""Diabetes Mellitus Overview:
Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels.
Type 2 diabetes accounts for 90-95% of all diabetes cases and is strongly associated with
obesity, physical inactivity, and genetic factors. Key risk factors include BMI >25,
age >45, family history, and sedentary lifestyle.""",
                metadata={"source": "Clinical Guidelines", "topic": "Overview"}
            ),
            Document(
                page_content="""Diabetes Prevention Strategies:
The Diabetes Prevention Program (DPP) demonstrated that lifestyle interventions can reduce
diabetes risk by 58%. Key interventions include: 1) Weight loss of 5-7% of body weight,
2) At least 150 minutes of moderate physical activity per week, 3) Dietary modifications
emphasizing whole grains, vegetables, and lean proteins, 4) Stress management and adequate sleep.""",
                metadata={"source": "DPP Study", "topic": "Prevention"}
            ),
            Document(
                page_content="""Diabetes Screening Recommendations:
The American Diabetes Association recommends screening for adults with BMI ≥25 and one or
more risk factors, or all adults ≥45 years. Screening tests include: HbA1c ≥6.5%, fasting
plasma glucose ≥126 mg/dL, or 2-hour plasma glucose ≥200 mg/dL during OGTT. Prediabetes is
defined as HbA1c 5.7-6.4% or fasting glucose 100-125 mg/dL.""",
                metadata={"source": "ADA Guidelines", "topic": "Screening"}
            ),
            Document(
                page_content="""Nutrition for Diabetes Prevention:
Evidence-based dietary patterns for diabetes prevention include Mediterranean diet, DASH diet,
and plant-based diets. Key principles: 1) Limit refined carbohydrates and added sugars,
2) Increase fiber intake to 25-30g daily, 3) Choose healthy fats (olive oil, nuts, avocado),
4) Control portion sizes, 5) Limit processed foods and red meat.""",
                metadata={"source": "Nutrition Guidelines", "topic": "Diet"}
            ),
            Document(
                page_content="""Physical Activity and Diabetes:
Regular physical activity improves insulin sensitivity and glucose metabolism. Recommendations:
1) 150 minutes of moderate-intensity aerobic activity per week, 2) Resistance training 2-3 times
per week, 3) Reduce sedentary time, 4) Include flexibility and balance exercises. Even small
amounts of activity (10-minute walks after meals) can improve blood glucose control.""",
                metadata={"source": "Exercise Guidelines", "topic": "Physical Activity"}
            ),
        ]

        try:
            # Try to use Qdrant if credentials are available
            if QDRANT_URL and QDRANT_API_KEY:
                try:
                    from langchain_qdrant import QdrantVectorStore
                except ImportError:
                    try:
                        from langchain_qdrant import Qdrant as QdrantVectorStore
                    except ImportError:
                        try:
                            from langchain_community.vectorstores import Qdrant as QdrantVectorStore
                        except ImportError:
                            from langchain.vectorstores import Qdrant as QdrantVectorStore

                from qdrant_client import QdrantClient

                self.qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                )

                self.vectorstore = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=QDRANT_COLLECTION_NAME,
                    embedding=self.embeddings,
                )
                self.vectorstore.add_documents(sample_docs)
                self.docs_loaded = True
                print("[OK] Created sample knowledge base in Qdrant")
            else:
                # Fall back to in-memory vector store for local testing
                try:
                    from langchain_community.vectorstores import FAISS
                except ImportError:
                    from langchain.vectorstores import FAISS
                self.vectorstore = FAISS.from_documents(
                    documents=sample_docs,
                    embedding=self.embeddings,
                )
                self.docs_loaded = True
                print("[OK] Created sample knowledge base in-memory (local mode)")
        except Exception as e:
            print(f"[ERROR] Error creating sample knowledge base: {e}")
            self.vectorstore = None

    def _initialize_phase2_components(self):
        """Initialize Phase 2 advanced retrieval components."""
        if not self.use_phase2:
            return

        print("[INFO] Initializing Phase 2 components...")

        # Initialize BM25 and hybrid retriever if enabled
        if RAG_USE_HYBRID_SEARCH and self.vectorstore:
            try:
                print("[INFO] Building BM25 index for hybrid search...")
                # Retrieve existing documents from Qdrant to build BM25 index
                documents = self._retrieve_all_documents_from_vectorstore()
                if documents:
                    self._build_bm25_index(documents)
                else:
                    print("[WARNING] No documents found for BM25 indexing. Hybrid search will activate after documents are loaded.")
            except Exception as e:
                print(f"[WARNING] Hybrid search initialization failed: {e}")

        print("[OK] Phase 2 components ready")

    def _retrieve_all_documents_from_vectorstore(self):
        """
        Retrieve all documents from the vector store for BM25 indexing.

        Returns:
            List of documents or empty list if retrieval fails
        """
        if not self.vectorstore:
            return []

        try:
            # Try to use a very broad query to get all documents
            # We'll use a large k value to retrieve many documents
            all_docs = self.vectorstore.similarity_search(
                query="diabetes health medical clinical",
                k=10000  # Large number to get all documents
            )

            if all_docs:
                print(f"[OK] Retrieved {len(all_docs)} documents from vector store for BM25 indexing")
                return all_docs
            else:
                print("[WARNING] No documents retrieved from vector store")
                return []

        except Exception as e:
            print(f"[WARNING] Failed to retrieve documents from vector store: {e}")
            return []

    def _build_bm25_index(self, documents):
        """Build BM25 index from documents for hybrid search."""
        if not RAG_USE_HYBRID_SEARCH or not self.use_phase2:
            return

        try:
            # Build BM25 retriever
            self.bm25_retriever = BM25Retriever(documents)

            # Create hybrid retriever
            if self.bm25_retriever and self.vectorstore:
                self.hybrid_retriever = HybridRetriever(
                    vectorstore=self.vectorstore,
                    bm25_retriever=self.bm25_retriever,
                    alpha=RAG_HYBRID_ALPHA
                )
                print(f"[OK] Hybrid retriever initialized (alpha={RAG_HYBRID_ALPHA})")
        except Exception as e:
            print(f"[WARNING] Failed to build BM25 index: {e}")

    def generate_response(
        self,
        message: str,
        context: Dict,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response using RAG with quality controls.

        Args:
            message: User message
            context: Prediction context
            conversation_history: Previous messages

        Returns:
            str: RAG-generated response with citations
        """
        if not self.is_initialized or not self.vectorstore:
            return self._generate_fallback_response(message, context)

        try:
            # Retrieve relevant documents
            if self.use_phase2 and RAG_USE_HYBRID_SEARCH and self.hybrid_retriever:
                # Phase 2: Hybrid retrieval (semantic + BM25)
                print("[INFO] Using hybrid retrieval (semantic + BM25)")
                
                # Retrieve documents
                docs_with_scores = self.hybrid_retriever.retrieve(message, k=RAG_INITIAL_K)
                
                relevant_docs_with_scores = docs_with_scores[:RAG_INITIAL_K * 2]
            else:
                # Phase 1: Semantic-only retrieval
                relevant_docs_with_scores = self.vectorstore.similarity_search_with_score(
                    message,
                    k=RAG_INITIAL_K * 2
                )

            # Apply quality filtering
            filtered_docs = self._filter_by_relevance(
                relevant_docs_with_scores,
                min_score=RAG_MIN_RELEVANCE_SCORE
            )

            if not filtered_docs:
                print("[WARNING] No relevant documents found above quality threshold")
                return self._generate_fallback_response(message, context)

            # Take top K after filtering
            top_docs = filtered_docs[:RAG_TOP_K]

            # Extract just the documents (without scores)
            docs_only = [doc for doc, score in top_docs]

            # Build context from retrieved documents
            retrieved_context = self._format_retrieved_docs(docs_only)
            source_metadata = self._format_source_metadata(docs_only)

            # Build prompt with retrieved context and prediction data
            prompt = self._build_rag_prompt(message, context, retrieved_context, source_metadata)

            # Generate response using Gemini
            if self.model:
                response = self.model.generate_content(prompt)
                response_text = response.text
                return response_text
            else:
                return self._generate_fallback_response(message, context)

        except Exception as e:
            print(f"Error generating RAG response: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_response(message, context)

    def _filter_by_relevance(self, docs_with_scores, min_score=0.65, min_evidence_level='C'):
        """
        Filter documents by relevance score and quality metrics.

        Args:
            docs_with_scores: List of (document, score) tuples
            min_score: Minimum similarity score (0-1, lower is better for distance metrics)
            min_evidence_level: Minimum evidence level (A, B, or C)

        Returns:
            List of filtered (document, score) tuples
        """
        filtered = []
        min_evidence_rank = get_evidence_rank(min_evidence_level)

        for doc, score in docs_with_scores:
            # Note: Qdrant returns distance, so lower is better
            # For cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity-like score where higher is better
            similarity_score = 1 - (score / 2)  # Normalize to 0-1 range

            # Check similarity threshold
            if similarity_score < min_score:
                continue

            # Check evidence level
            evidence_level = doc.metadata.get('evidence_level', 'C')
            if get_evidence_rank(evidence_level) < min_evidence_rank:
                continue

            # Add warning flag if outdated
            pub_date = doc.metadata.get('publication_date')
            if pub_date and not is_recent_guideline(pub_date, max_age_years=5):
                doc.metadata['outdated_warning'] = True

            filtered.append((doc, similarity_score))

        # Sort by score (higher is better after conversion)
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered
    
    def get_capabilities(self) -> List[str]:
        """Return list of RAG agent capabilities."""
        return self.capabilities
    
    def _format_retrieved_docs(self, docs) -> str:
        """Format retrieved documents for prompt with rich citations."""
        if not docs:
            return "No relevant clinical information found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "Unknown")
            pub_date = doc.metadata.get("publication_date", "Unknown")
            evidence_level = doc.metadata.get("evidence_level", "C")
            doi = doc.metadata.get("doi", "")

            # Extract year from publication date
            try:
                year = pub_date.split("-")[0] if pub_date != "Unknown" else "Unknown"
            except:
                year = "Unknown"

            # Format citation
            citation = f"[Source {i}: {source}, {year}"
            if evidence_level:
                citation += f" | Evidence Level: {evidence_level}"
            citation += "]"

            content = doc.page_content

            # Add outdated warning if applicable
            if doc.metadata.get('outdated_warning'):
                citation += " ⚠️ Note: Guideline may be outdated"

            formatted.append(f"{citation}\n{content}\n")

        return "\n".join(formatted)

    def _format_source_metadata(self, docs) -> str:
        """Format source metadata summary for prompt."""
        if not docs:
            return "No sources available."

        metadata_lines = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "Unknown")
            evidence_level = doc.metadata.get("evidence_level", "C")
            doc_type = doc.metadata.get("document_type", "unknown")
            pub_date = doc.metadata.get("publication_date", "Unknown")

            try:
                year = pub_date.split("-")[0] if pub_date != "Unknown" else "Unknown"
            except:
                year = "Unknown"

            line = f"Source {i}: {source} ({year}) - {title}"
            line += f" [Evidence: {evidence_level}, Type: {doc_type}]"

            metadata_lines.append(line)

        return "\n".join(metadata_lines)
    
    def _build_rag_prompt(self, message: str, context: Dict, retrieved_context: str, source_metadata: str) -> str:
        """Build grounded generation prompt with retrieved context and prediction data."""
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        profile_summary = context.get("profile_summary", "No data available")

        return f"""You are a clinical information assistant providing evidence-based diabetes information.

CRITICAL RULES (YOU MUST FOLLOW THESE STRICTLY):
1. ONLY use information from the provided sources below - NO external knowledge
2. CITE sources explicitly using [Source: Organization, Year] notation for every claim
3. If sources contradict each other, mention both perspectives with citations
4. If sources don't contain the answer, say "I don't have information about that in the current knowledge base"
5. NEVER speculate or use general knowledge outside the provided sources
6. Always include medical disclaimers for clinical recommendations
7. Do not make up information - stay grounded in the evidence below

PATIENT CONTEXT:
- Risk Level: {risk_level}
- Risk Probability: {probability:.1f}%
- Key Metrics:
{profile_summary}

RETRIEVED EVIDENCE FROM CLINICAL GUIDELINES:
{retrieved_context}

SOURCE QUALITY INDICATORS:
{source_metadata}

USER QUESTION:
{message}

RESPONSE FORMAT (Follow this structure):

**Answer:**
[Provide a direct, evidence-based answer to the question in 2-3 sentences. Include a citation.]

**Supporting Evidence:**
[Provide detailed supporting information with specific citations from the sources. Use [Source: X, Year] format. Include relevant statistics, guidelines, or recommendations from the retrieved documents.]

**Recommendations for This Risk Level ({risk_level}):**
[Provide 2-4 actionable recommendations specific to the patient's risk level. Each recommendation should have a citation.]

**Important Medical Disclaimer:**
This information is for educational purposes only and is based on current clinical guidelines. It is not a substitute for professional medical advice, diagnosis, or treatment. Please consult with your healthcare provider for personalized medical recommendations.

---

Now provide your evidence-based response following the format above:"""
    
    def _generate_fallback_response(self, message: str, context: Dict) -> str:
        """Generate fallback response when RAG is unavailable."""
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        
        return f"""I'm currently unable to access the clinical knowledge base, but I can provide general evidence-based information.

Based on your {probability:.1f}% risk probability ({risk_level} risk):

**Evidence-Based Recommendations:**

1. **Lifestyle Modifications** (Diabetes Prevention Program):
   - Target 5-7% weight loss if overweight
   - 150 minutes moderate physical activity weekly
   - Dietary focus on whole grains, vegetables, lean proteins

2. **Screening** (ADA Guidelines):
   - Discuss HbA1c and fasting glucose testing with your doctor
   - Prediabetes range: HbA1c 5.7-6.4%
   - Regular monitoring if risk factors present

3. **Nutrition** (Evidence-Based):
   - Mediterranean or DASH diet patterns
   - Increase fiber to 25-30g daily
   - Limit refined carbohydrates and added sugars

4. **Physical Activity**:
   - Combination of aerobic and resistance training
   - Even 10-minute walks after meals help glucose control
   - Reduce sedentary time

**Your Question:** "{message}"

For detailed, personalized clinical guidance with specific literature references, please consult with your healthcare provider or try again when the knowledge base is available.

**Disclaimer:** This is general educational information based on clinical guidelines. Always consult healthcare professionals for personalized medical advice."""
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add new documents to the knowledge base.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not self.vectorstore:
            print("[WARNING] Vector store not initialized")
            return

        try:
            try:
                from langchain_core.documents import Document
            except ImportError:
                from langchain.schema import Document
                
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import RecursiveCharacterTextSplitter

            # Create Document objects
            docs = [
                Document(page_content=text, metadata=metadata[i] if metadata else {})
                for i, text in enumerate(documents)
            ]

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)

            # Add to vector store (works for both Qdrant and FAISS)
            self.vectorstore.add_documents(splits)
            print(f"[OK] Added {len(splits)} document chunks to knowledge base")
        except Exception as e:
            print(f"[ERROR] Error adding documents: {e}")
