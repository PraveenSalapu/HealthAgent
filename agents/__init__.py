"""Agents module - Multi-agent support with Gemini and Lightweight RAG agents."""

from .base_agent import BaseAgent
from .gemini_agent import GeminiAgent
from .lightweight_rag_agent import LightweightRAGAgent
# REMOVED: Heavy RAG agent (LangChain-based)
# from .rag_agent import RAGAgent
from .agent_manager import AgentManager

__all__ = [
    'BaseAgent',
    'GeminiAgent',
    'LightweightRAGAgent',
    # REMOVED: Heavy RAG agent
    # 'RAGAgent',
    'AgentManager',
]
