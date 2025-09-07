"""
嵌入模型服务模块

提供google/embeddinggemma-300m嵌入模型的文本向量化服务。
"""

from .embedding_service import EmbeddingService

__all__ = ["EmbeddingService"]
