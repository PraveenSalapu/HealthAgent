"""
Lightweight RAG components using API-based embeddings (no heavy ML dependencies).
Uses Gemini API for embeddings instead of local models - perfect for Streamlit Cloud.
"""

import re
import uuid
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class Document:
    """Minimal document class for storing content and metadata."""
    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class GeminiEmbeddings:
    """API-based embeddings using Gemini (no local model required)."""
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        import google.generativeai as genai
        results = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            results.append(result['embedding'])
        return results


class QdrantVectorStore:
    """Lightweight vector store wrapper for Qdrant."""
    def __init__(self, client, collection_name: str, embeddings: GeminiEmbeddings):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents and return with scores."""
        query_vector = self.embeddings.embed_query(query)

        # Use query_points for newer Qdrant client versions
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k
            )
            points = results.points
        except (AttributeError, TypeError):
            # Fallback: use search method for older Qdrant versions
            try:
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k
                )
            except AttributeError:
                # Last fallback: scroll through collection
                from qdrant_client.models import Filter
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=k
                )
                points = scroll_result[0]

        docs_with_scores = []
        for result in points:
            doc = Document(
                page_content=result.payload.get('content', ''),
                metadata=result.payload.get('metadata', {})
            )
            # Default score if not available
            score = getattr(result, 'score', 1.0)
            docs_with_scores.append((doc, score))

        return docs_with_scores

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        from qdrant_client.models import PointStruct

        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)

        points = []
        for doc, embedding in zip(documents, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)


class BM25Retriever:
    """Keyword-based retrieval using BM25 algorithm."""
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.bm25 = None
        self._build_index()

    def _build_index(self):
        """Build BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            print("[WARNING] rank-bm25 not installed. BM25 disabled.")

    def _tokenize(self, text: str) -> List[str]:
        """Medical-aware tokenization preserving clinical terms."""
        text_lower = text.lower()
        tokens = re.findall(r'\b\w+(?:[-./]\w+)*\b', text_lower)
        return tokens

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents using BM25 scoring."""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [(self.documents[i], scores[i]) for i in top_k_indices if i < len(self.documents)]
        return results


class HybridRetriever:
    """Combines semantic (vector) and keyword (BM25) search."""
    def __init__(self, vectorstore: QdrantVectorStore, bm25_retriever: Optional[BM25Retriever] = None, alpha: float = 0.7):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Hybrid retrieval with weighted fusion."""
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)

        if self.bm25_retriever:
            keyword_results = self.bm25_retriever.retrieve(query, k=k*2)
            combined = self._reciprocal_rank_fusion(semantic_results, keyword_results)
            return combined[:k]

        return semantic_results[:k]

    def _reciprocal_rank_fusion(self, sem_results: List[Tuple], kw_results: List[Tuple]) -> List[Tuple]:
        """Combine rankings using RRF algorithm."""
        k = 60
        scores = defaultdict(float)
        doc_map = {}

        for rank, (doc, _) in enumerate(sem_results):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] += self.alpha * (1 / (k + rank + 1))

        for rank, (doc, _) in enumerate(kw_results):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] += (1 - self.alpha) * (1 / (k + rank + 1))

        ranked_doc_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[doc_id], score) for doc_id, score in ranked_doc_ids]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple text chunking with overlap."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


def load_pdf(file_path: str) -> str:
    """Load text from PDF file."""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"[ERROR] Failed to load PDF {file_path}: {e}")
        return ""


def load_text_file(file_path: str) -> str:
    """Load text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"[ERROR] Failed to load text file {file_path}: {e}")
        return ""
