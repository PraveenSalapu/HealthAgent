"""
Admin page for uploading and indexing clinical documents.
Healthcare organizations can upload their clinical guidelines here.
"""

import streamlit as st
from pathlib import Path
import os
from agents import LightweightRAGAgent

st.set_page_config(page_title="Document Upload - Admin", page_icon="📤", layout="wide")

st.title("📤 Clinical Document Upload")
st.markdown("Upload clinical guidelines, protocols, and research papers for the RAG system.")

# Initialize session state
if 'lightweight_rag_agent' not in st.session_state:
    st.session_state.lightweight_rag_agent = None

# Admin password (simple protection)
admin_section = st.sidebar
with admin_section:
    st.subheader("🔐 Admin Access")
    password = st.text_input("Admin Password", type="password")

    # Simple password check (use environment variable in production)
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

    if password == ADMIN_PASSWORD:
        st.success("✓ Authenticated")
        is_admin = True
    else:
        st.warning("Enter admin password to continue")
        is_admin = False

if not is_admin:
    st.info("👆 Enter the admin password in the sidebar to upload documents.")
    st.stop()

# Main upload interface
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload Clinical Documents (PDF or TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload clinical guidelines, protocols, research papers, etc."
    )

    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} file(s) selected")

        # Show file details
        with st.expander("📋 File Details"):
            for file in uploaded_files:
                st.write(f"- **{file.name}** ({file.size / 1024:.1f} KB)")

with col2:
    st.subheader("ℹ️ Guidelines")
    st.info("""
    **Supported formats:**
    - PDF files (.pdf)
    - Text files (.txt)

    **Best practices:**
    - Use descriptive filenames
    - Keep files under 10MB
    - Avoid duplicate content
    """)

# Save and index section
st.markdown("---")

if uploaded_files:
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Documents", type="primary", use_container_width=True):
            # Create directory if doesn't exist
            docs_dir = Path("data/clinical_docs")
            docs_dir.mkdir(parents=True, exist_ok=True)

            # Save files
            saved_count = 0
            with st.spinner("Saving documents..."):
                for uploaded_file in uploaded_files:
                    file_path = docs_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1

            st.success(f"✓ Saved {saved_count} document(s) to {docs_dir}")

    with col2:
        if st.button("🔄 Index Documents", type="secondary", use_container_width=True):
            with st.spinner("Indexing documents... This may take a few minutes."):
                # Initialize agent if needed
                if st.session_state.lightweight_rag_agent is None:
                    agent = LightweightRAGAgent()
                    if agent.initialize():
                        st.session_state.lightweight_rag_agent = agent
                    else:
                        st.error("Failed to initialize RAG agent")
                        st.stop()

                # Index documents
                result = st.session_state.lightweight_rag_agent.index_documents()
                st.success(f"✓ {result}")
                st.balloons()

# Current indexed documents
st.markdown("---")
st.subheader("📚 Current Documents")

docs_dir = Path("data/clinical_docs")
if docs_dir.exists():
    files = list(docs_dir.glob("**/*.pdf")) + list(docs_dir.glob("**/*.txt"))

    if files:
        st.write(f"**Total documents:** {len(files)}")

        # Show in expandable table
        with st.expander("View all documents"):
            for i, file in enumerate(files, 1):
                size_kb = file.stat().st_size / 1024
                st.write(f"{i}. **{file.name}** ({size_kb:.1f} KB)")
    else:
        st.info("No documents found. Upload documents above to get started.")
else:
    st.info("Documents directory not found. Upload your first document to create it.")

# Test query section
st.markdown("---")
st.subheader("🧪 Test RAG System")

test_query = st.text_input(
    "Test Query",
    placeholder="e.g., What are the HbA1c thresholds for prediabetes?",
    help="Test if your documents are indexed correctly"
)

if test_query and st.button("🔍 Test Query"):
    if st.session_state.lightweight_rag_agent is None:
        agent = LightweightRAGAgent()
        if agent.initialize():
            st.session_state.lightweight_rag_agent = agent
        else:
            st.error("Failed to initialize RAG agent")
            st.stop()

    with st.spinner("Searching documents..."):
        test_context = {
            "probability": 25.0,
            "risk_level": "Low",
            "profile_summary": "Test query"
        }

        response = st.session_state.lightweight_rag_agent.generate_response(
            message=test_query,
            context=test_context
        )

    st.markdown("### Response:")
    st.markdown(response)

# Footer
st.markdown("---")
st.caption("💡 **Tip:** After indexing, documents will be available in the main chat interface for all users.")
