"""
HuggingFace Qwen3-8B 4bit推理与小上下文RAG系统

这个包提供了基于HuggingFace Qwen3-8B模型的4bit量化推理系统，
集成ChromaDB向量数据库和LlamaIndex框架，实现高效的检索增强生成(RAG)。
"""

__version__ = "1.0.0"
__author__ = "RAG System Developer"
__email__ = "developer@example.com"

from .models import Qwen3Model
from .embeddings import EmbeddingService
from .vectordb import ChromaDBManager
from .rag import RAGQueryProcessor

__all__ = [
    "Qwen3Model",
    "EmbeddingService", 
    "ChromaDBManager",
    "RAGQueryProcessor"
]
