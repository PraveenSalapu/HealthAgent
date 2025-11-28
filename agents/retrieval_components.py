"""
Advanced retrieval components for Phase 2 RAG improvements.

Components:
1. BM25Retriever - Sparse keyword-based retrieval
2. HybridRetriever - Combines dense (semantic) and sparse (keyword) search
3. CrossEncoderReranker - Precise relevance scoring for re-ranking
4. CitationValidator - Validates citations in generated responses
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


class CrossEncoderReranker:
    """
    Cross-encoder for re-ranking retrieved documents.

    Cross-encoders are more accurate than bi-encoders (embeddings) because
    they process query and document together, but they're slower.
    Use for final re-ranking of top candidates.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            print(f"[OK] Cross-encoder loaded: {self.model_name}")
        except ImportError:
            print("[WARNING] sentence-transformers not available for re-ranking")
        except Exception as e:
            print(f"[WARNING] Failed to load cross-encoder: {e}")

    def rerank(self, query: str, documents: List, top_k: int = 3) -> List[Tuple]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: User query
            documents: List of documents (or (doc, score) tuples)
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of (document, score) tuples
        """
        if not self.model:
            # Fallback: return original documents
            if documents and isinstance(documents[0], tuple):
                return documents[:top_k]
            return [(doc, 0.0) for doc in documents[:top_k]]

        # Extract documents if input is (doc, score) tuples
        if documents and isinstance(documents[0], tuple):
            docs_only = [doc for doc, _ in documents]
        else:
            docs_only = documents

        if not docs_only:
            return []

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in docs_only]

        # Score pairs with cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"[WARNING] Cross-encoder prediction failed: {e}")
            return [(doc, 0.0) for doc in docs_only[:top_k]]

        # Sort by score (higher is better)
        doc_score_pairs = list(zip(docs_only, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_k]


class CitationValidator:
    """
    Validates that generated responses contain proper citations.

    Ensures responses are grounded in retrieved sources and not hallucinated.
    """

    def __init__(self):
        """Initialize citation validator."""
        self.citation_pattern = r'\[Source:.*?\]'
        self.speculative_terms = [
            "might", "could be", "possibly", "maybe", "perhaps",
            "I think", "probably", "likely", "seems"
        ]

    def validate(self, response: str, source_docs: List) -> Dict:
        """
        Validate response quality and citation coverage.

        Args:
            response: Generated response text
            source_docs: List of source documents used

        Returns:
            Dict with validation results
        """
        results = {
            "has_citations": False,
            "citation_count": 0,
            "has_disclaimer": False,
            "speculative_language": [],
            "validation_passed": False,
            "issues": []
        }

        # Check for citations
        citations = re.findall(self.citation_pattern, response)
        results["citation_count"] = len(citations)
        results["has_citations"] = len(citations) > 0

        if not results["has_citations"]:
            results["issues"].append("No citations found in response")

        # Check for medical disclaimer
        disclaimer_keywords = ["consult", "healthcare provider", "medical advice", "disclaimer"]
        has_disclaimer = any(keyword in response.lower() for keyword in disclaimer_keywords)
        results["has_disclaimer"] = has_disclaimer

        if not has_disclaimer:
            results["issues"].append("No medical disclaimer found")

        # Check for speculative language
        for term in self.speculative_terms:
            if term.lower() in response.lower():
                results["speculative_language"].append(term)

        if results["speculative_language"]:
            results["issues"].append(f"Speculative language detected: {', '.join(results['speculative_language'])}")

        # Overall validation
        results["validation_passed"] = (
            results["has_citations"] and
            results["has_disclaimer"] and
            len(results["speculative_language"]) == 0
        )

        return results

    def get_validation_summary(self, validation_result: Dict) -> str:
        """
        Get human-readable validation summary.

        Args:
            validation_result: Result from validate()

        Returns:
            Summary string
        """
        if validation_result["validation_passed"]:
            return f"[OK] Validation passed ({validation_result['citation_count']} citations)"
        else:
            issues = "; ".join(validation_result["issues"])
            return f"[WARNING] Validation issues: {issues}"


class QueryExpander:
    """
    Expands queries with medical synonyms and related terms.

    Helps retrieve relevant documents even when user uses different terminology.
    """

    def __init__(self):
        """Initialize query expander with medical term mappings."""
        # Medical term expansions (can be extended)
        self.term_expansions = {
            "diabetes": ["diabetes mellitus", "diabetic", "hyperglycemia", "high blood sugar"],
            "sugar": ["glucose", "blood sugar", "blood glucose"],
            "A1C": ["HbA1c", "hemoglobin A1c", "glycated hemoglobin", "glycohemoglobin"],
            "a1c": ["HbA1c", "hemoglobin A1c", "glycated hemoglobin"],
            "overweight": ["obesity", "obese", "BMI", "body mass index", "weight"],
            "exercise": ["physical activity", "movement", "fitness", "activity"],
            "diet": ["nutrition", "eating", "food", "dietary"],
            "prevention": ["prevent", "preventing", "avoid", "reduce risk"],
            "risk": ["risk factors", "predisposition", "susceptibility"],
            "prediabetes": ["pre-diabetes", "impaired glucose tolerance", "IFG", "IGT"],
        }

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with medical synonyms.

        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries to return

        Returns:
            List of queries (original + expansions)
        """
        expanded_queries = [query]  # Always include original

        # Find terms to expand
        query_lower = query.lower()
        for term, synonyms in self.term_expansions.items():
            if term.lower() in query_lower:
                # Create expanded versions
                for synonym in synonyms[:max_expansions-1]:  # Limit expansions
                    expanded = query.replace(term, synonym)
                    if expanded not in expanded_queries:
                        expanded_queries.append(expanded)

                # Don't expand too many terms at once
                if len(expanded_queries) >= max_expansions:
                    break

        return expanded_queries[:max_expansions]
