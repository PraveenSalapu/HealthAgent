"""
Index clinical documents into the lightweight RAG system.
Run this script after adding documents to data/clinical_docs/
"""

from agents import LightweightRAGAgent

def index_documents():
    """Index all documents in data/clinical_docs/"""
    print("="*60)
    print("INDEXING CLINICAL DOCUMENTS")
    print("="*60)

    # Initialize agent
    print("\n[1/3] Initializing Lightweight RAG Agent...")
    agent = LightweightRAGAgent()

    if not agent.initialize():
        print("[ERROR] Agent initialization failed")
        return

    print("[OK] Agent initialized")

    # Index documents
    print("\n[2/3] Indexing documents from data/clinical_docs/...")
    result = agent.index_documents()
    print(f"[OK] {result}")

    # Test query
    print("\n[3/3] Testing indexed documents...")
    test_context = {
        "probability": 35.5,
        "risk_level": "Moderate",
        "profile_summary": "BMI: 28, Age: 45-49"
    }

    response = agent.generate_response(
        message="What are the diabetes screening recommendations?",
        context=test_context
    )

    print("\n[OK] Test query successful!")
    print(f"\nResponse preview:\n{response[:300]}...\n")

    print("="*60)
    print("[SUCCESS] Documents indexed and ready to use!")
    print("="*60)


if __name__ == "__main__":
    index_documents()
