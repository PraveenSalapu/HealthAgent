"""
Lightweight retrieval components for RAG.

Components:
1. BM25Retriever - Sparse keyword-based retrieval
2. HybridRetriever - Combines dense (semantic) and sparse (keyword) search
"""

from typing import List, Tuple, Optional, Dict
import re
from collections import defaultdict


class BM25Retriever:
    """
    BM25 (Best Matching 25) sparse retrieval for keyword search.

    Complements semantic search by capturing exact keyword matches,
    which is important for medical terminology.
    """

    def __init__(self, documents: List = None):
        """
        Initialize BM25 retriever.

        Args:
            documents: List of documents to index
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []

        if documents:
            self._build_index(documents)

    def _build_index(self, documents):
        """Build BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[WARNING] rank-bm25 not installed. BM25 search disabled.")
            return

        # Tokenize documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"[OK] BM25 index built with {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be enhanced with medical tokenizers).

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def retrieve(self, query: str, k: int = 5) -> List[Tuple]:
        """
        Retrieve top-k documents using BM25.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if not self.bm25:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Return documents with scores
        results = [(self.documents[i], scores[i]) for i in top_k_indices if i < len(self.documents)]

        return results


class HybridRetriever:
    """
    Hybrid retrieval combining semantic (dense) and keyword (sparse) search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both methods.
    """

    def __init__(self, vectorstore, bm25_retriever: Optional[BM25Retriever] = None, alpha: float = 0.7):
        """
        Initialize hybrid retriever.

        Args:
            vectorstore: Dense vector store for semantic search
            bm25_retriever: BM25 retriever for keyword search
            alpha: Weight for semantic search (0=pure keyword, 1=pure semantic)
        """
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> List[Tuple]:
        """
        Hybrid retrieval with weighted fusion.

        Args:
            query: User query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        # Get semantic search results
        try:
            semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        except Exception as e:
            print(f"[WARNING] Semantic search failed: {e}")
            semantic_results = []

        # Get keyword search results (BM25)
        keyword_results = []
        if self.bm25_retriever:
            try:
                keyword_results = self.bm25_retriever.retrieve(query, k=k*2)
            except Exception as e:
                print(f"[WARNING] BM25 search failed: {e}")

        # If only one method available, return those results
        if not semantic_results:
            return keyword_results[:k]
        if not keyword_results:
            return semantic_results[:k]

        # Combine using Reciprocal Rank Fusion (RRF)
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            alpha=self.alpha
        )

        return combined[:k]

    def _reciprocal_rank_fusion(self, sem_results: List[Tuple], kw_results: List[Tuple], alpha: float = 0.7) -> List[Tuple]:
        """
        Combine rankings using Reciprocal Rank Fusion algorithm.

        RRF formula: score = sum(1 / (k + rank))
        Where k is a constant (typically 60)

        Args:
            sem_results: Semantic search results [(doc, score), ...]
            kw_results: Keyword search results [(doc, score), ...]
            alpha: Weight for semantic (1-alpha for keyword)

        Returns:
            Combined ranked list
        """
        k = 60  # RRF constant
        scores = defaultdict(float)
        doc_map = {}  # Map doc_id to actual document

        # Score semantic results
        for rank, (doc, _) in enumerate(sem_results):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] += alpha * (1 / (k + rank + 1))

        # Score keyword results
        for rank, (doc, _) in enumerate(kw_results):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] += (1 - alpha) * (1 / (k + rank + 1))

        # Sort by combined score
        ranked_doc_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return as list of (doc, score) tuples
        return [(doc_map[doc_id], score) for doc_id, score in ranked_doc_ids]