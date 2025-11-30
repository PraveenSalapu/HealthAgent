"""
Test RAG retrieval to verify documents are being found and used.
"""

from agents import LightweightRAGAgent

def test_retrieval():
    """Test document retrieval and response generation."""
    print("="*60)
    print("TESTING RAG RETRIEVAL")
    print("="*60)

    # Initialize agent
    print("\n[1/4] Initializing Lightweight RAG Agent...")
    agent = LightweightRAGAgent()

    if not agent.initialize():
        print("[ERROR] Agent initialization failed")
        return

    print(f"[OK] Agent initialized")
    print(f"[INFO] Vectorstore: {agent.vectorstore}")
    print(f"[INFO] Hybrid retriever: {agent.hybrid_retriever}")

    # Test context
    test_context = {
        "probability": 35.5,
        "risk_level": "Moderate",
        "profile_summary": "BMI: 28, Age: 45-49"
    }

    # Test query
    query = "What are the diabetes screening recommendations?"
    print(f"\n[2/4] Testing retrieval for query: '{query}'")

    # Direct retrieval test
    from config.settings import RAG_USE_HYBRID_SEARCH, RAG_TOP_K, RAG_MIN_RELEVANCE_SCORE

    if RAG_USE_HYBRID_SEARCH and agent.hybrid_retriever:
        print(f"[INFO] Using hybrid retrieval (top_k={RAG_TOP_K})")
        docs_with_scores = agent.hybrid_retriever.retrieve(query, k=RAG_TOP_K)
    else:
        print(f"[INFO] Using semantic retrieval (top_k={RAG_TOP_K})")
        docs_with_scores = agent.vectorstore.similarity_search_with_score(query, k=RAG_TOP_K)

    print(f"\n[3/4] Retrieved {len(docs_with_scores)} documents:")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n  Document {i}:")
        print(f"    Score: {score:.4f} (threshold: {RAG_MIN_RELEVANCE_SCORE})")
        print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
        # Encode to ASCII with error handling for Unicode characters
        preview = doc.page_content[:150].encode('ascii', errors='replace').decode('ascii')
        print(f"    Content preview: {preview}...")
        if score >= RAG_MIN_RELEVANCE_SCORE:
            print(f"    [OK] PASSES threshold")
        else:
            print(f"    [WARN] BELOW threshold")

    # Filter by relevance
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= RAG_MIN_RELEVANCE_SCORE]
    print(f"\n[INFO] {len(filtered_docs)}/{len(docs_with_scores)} documents pass relevance threshold")

    # Generate response
    print(f"\n[4/4] Generating response...")
    response = agent.generate_response(
        message=query,
        context=test_context
    )

    # Write response to file to avoid Unicode issues
    with open("test_rag_response.txt", "w", encoding="utf-8") as f:
        f.write(response)

    print("\n" + "="*60)
    print("FULL RESPONSE saved to: test_rag_response.txt")
    print("="*60)

    # Check if response is fallback
    if "RAG retrieval unavailable" in response or "consult a healthcare professional" in response.lower():
        print("\n[WARN] Fallback response detected (no relevant documents found)")
    else:
        print("\n[OK] RAG response generated successfully")

    # Check for citations
    import re
    citations = re.findall(r'\[\d+\]', response)
    if citations:
        print(f"[OK] Found {len(citations)} citations: {citations}")
    else:
        print("[WARN] No citations found in response")

    # Check for references section
    if "**References:**" in response or "**Sources:**" in response:
        print("[OK] References section found")
    else:
        print("[WARN] No References section found")


if __name__ == "__main__":
    test_retrieval()
