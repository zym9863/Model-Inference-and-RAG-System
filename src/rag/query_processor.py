"""
RAG查询处理器

实现查询处理逻辑，包括向量检索、上下文构建和答案生成。
"""

from typing import List, Dict, Any, Optional, Union
import logging
import time
from pathlib import Path

from ..models.qwen3_model import Qwen3Model
from ..embeddings.embedding_service import EmbeddingService
from ..vectordb.chromadb_manager import ChromaDBManager
from .llama_index_integration import LlamaIndexRAG

logger = logging.getLogger(__name__)


class RAGQueryProcessor:
    """
    RAG查询处理器
    
    整合所有组件，提供完整的RAG查询处理功能。
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model_name: str = "google/gemma-2b",
        chroma_persist_dir: str = "./data/chromadb",
        collection_name: str = "rag_documents",
        use_llama_index: bool = True,
        max_context_length: int = 1500,
        top_k_retrieval: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        初始化RAG查询处理器
        
        Args:
            llm_model_name: 语言模型名称
            embedding_model_name: 嵌入模型名称
            chroma_persist_dir: ChromaDB持久化目录
            collection_name: 集合名称
            use_llama_index: 是否使用LlamaIndex框架
            max_context_length: 最大上下文长度
            top_k_retrieval: 检索的文档数量
            similarity_threshold: 相似度阈值
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.use_llama_index = use_llama_index
        self.max_context_length = max_context_length
        self.top_k_retrieval = top_k_retrieval
        self.similarity_threshold = similarity_threshold
        
        # 组件初始化
        self.llm = None
        self.embedding_service = None
        self.vector_db = None
        self.llama_index_rag = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """
        初始化所有组件
        """
        try:
            logger.info("正在初始化RAG系统组件...")
            
            # 初始化语言模型
            logger.info("初始化语言模型...")
            self.llm = Qwen3Model(
                model_name=self.llm_model_name,
                device="auto"
            )
            
            if self.use_llama_index:
                # 使用LlamaIndex框架
                logger.info("初始化LlamaIndex RAG框架...")
                self.llama_index_rag = LlamaIndexRAG(
                    embedding_model_name=self.embedding_model_name,
                    llm_model_name=self.llm_model_name,
                    chroma_persist_dir=self.chroma_persist_dir,
                    collection_name=self.collection_name,
                    similarity_threshold=self.similarity_threshold
                )
            else:
                # 使用自定义组件
                logger.info("初始化嵌入服务...")
                self.embedding_service = EmbeddingService(
                    model_name=self.embedding_model_name,
                    device="auto"
                )
                
                logger.info("初始化向量数据库...")
                self.vector_db = ChromaDBManager(
                    persist_directory=self.chroma_persist_dir,
                    collection_name=self.collection_name
                )
            
            logger.info("RAG系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Union[str, Dict[str, Any]]], 
        show_progress: bool = True
    ) -> List[str]:
        """
        添加文档到RAG系统
        
        Args:
            documents: 文档列表
            show_progress: 是否显示进度
            
        Returns:
            文档ID列表
        """
        if not documents:
            return []
        
        try:
            if self.use_llama_index:
                # 使用LlamaIndex添加文档
                self.llama_index_rag.add_documents(documents, show_progress)
                return [f"doc_{i}" for i in range(len(documents))]
            else:
                # 使用自定义组件添加文档
                return self._add_documents_custom(documents, show_progress)
                
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def _add_documents_custom(
        self, 
        documents: List[Union[str, Dict[str, Any]]], 
        show_progress: bool = True
    ) -> List[str]:
        """
        使用自定义组件添加文档
        """
        # 提取文本和元数据
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                texts.append(doc)
                metadatas.append({"source": f"document_{i}"})
            elif isinstance(doc, dict):
                texts.append(doc.get("text", ""))
                metadatas.append(doc.get("metadata", {"source": f"document_{i}"}))
        
        # 生成嵌入向量
        if show_progress:
            logger.info(f"正在生成 {len(texts)} 个文档的嵌入向量...")
        
        embeddings = self.embedding_service.encode_text(
            texts, 
            normalize=True, 
            show_progress=show_progress
        )
        
        # 添加到向量数据库
        doc_ids = self.vector_db.add_documents(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )
        
        return doc_ids
    
    def query(
        self, 
        query_text: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            query_text: 查询文本
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            include_sources: 是否包含源文档信息
            
        Returns:
            查询结果字典
        """
        if not query_text.strip():
            return {"error": "查询文本不能为空"}
        
        try:
            start_time = time.time()
            
            if self.use_llama_index:
                # 使用LlamaIndex查询
                result = self._query_with_llama_index(
                    query_text, max_new_tokens, temperature, include_sources
                )
            else:
                # 使用自定义组件查询
                result = self._query_custom(
                    query_text, max_new_tokens, temperature, include_sources
                )
            
            # 添加性能指标
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return {"error": f"查询处理失败: {str(e)}"}
    
    def _query_with_llama_index(
        self, 
        query_text: str, 
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        include_sources: bool
    ) -> Dict[str, Any]:
        """
        使用LlamaIndex执行查询
        """
        # 执行RAG查询
        rag_result = self.llama_index_rag.query(
            query_text=query_text,
            similarity_top_k=self.top_k_retrieval
        )
        
        result = {
            "query": query_text,
            "response": rag_result["response"],
            "method": "llama_index"
        }
        
        if include_sources:
            result["sources"] = rag_result["source_nodes"]
            result["num_sources"] = rag_result["num_sources"]
        
        return result
    
    def _query_custom(
        self, 
        query_text: str, 
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        include_sources: bool
    ) -> Dict[str, Any]:
        """
        使用自定义组件执行查询
        """
        # 1. 检索相关文档
        retrieval_results = self.vector_db.query_documents(
            query_texts=query_text,
            n_results=self.top_k_retrieval,
            include=["documents", "metadatas", "distances"]
        )
        
        # 2. 过滤相似度
        relevant_docs = []
        sources = []
        
        if retrieval_results.get("documents"):
            for i, (doc, metadata, distance) in enumerate(zip(
                retrieval_results["documents"][0],
                retrieval_results["metadatas"][0],
                retrieval_results["distances"][0]
            )):
                # 转换距离为相似度分数
                similarity = 1 - distance
                
                if similarity >= self.similarity_threshold:
                    relevant_docs.append(doc)
                    if include_sources:
                        sources.append({
                            "text": doc,
                            "score": similarity,
                            "metadata": metadata
                        })
        
        # 3. 构建上下文
        context = self._build_context(relevant_docs)
        
        # 4. 生成回答
        prompt = self._build_prompt(query_text, context)
        
        response = self.llm.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens or 512,
            temperature=temperature or 0.7
        )
        
        result = {
            "query": query_text,
            "response": response,
            "method": "custom",
            "context_length": len(context)
        }
        
        if include_sources:
            result["sources"] = sources
            result["num_sources"] = len(sources)
        
        return result
    
    def _build_context(self, documents: List[str]) -> str:
        """
        构建上下文文本
        
        Args:
            documents: 相关文档列表
            
        Returns:
            构建的上下文文本
        """
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"文档{i+1}: {doc.strip()}"
            
            # 检查长度限制
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建查询提示
        
        Args:
            query: 用户查询
            context: 上下文文档
            
        Returns:
            完整的提示文本
        """
        if context:
            prompt = f"""基于以下提供的文档内容，请回答用户的问题。如果文档中没有相关信息，请说明无法从提供的文档中找到答案。

相关文档：
{context}

用户问题：{query}

请基于上述文档内容回答："""
        else:
            prompt = f"""用户问题：{query}

由于没有找到相关的文档内容，请基于你的知识回答这个问题："""
        
        return prompt
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        info = {
            "llm_model": self.llm_model_name,
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection_name,
            "use_llama_index": self.use_llama_index,
            "max_context_length": self.max_context_length,
            "top_k_retrieval": self.top_k_retrieval,
            "similarity_threshold": self.similarity_threshold
        }
        
        # 添加模型信息
        if self.llm:
            info["llm_info"] = self.llm.get_model_info()
        
        if self.use_llama_index and self.llama_index_rag:
            info["index_info"] = self.llama_index_rag.get_index_info()
        elif self.embedding_service:
            info["embedding_info"] = self.embedding_service.get_model_info()
        
        if self.vector_db:
            info["vector_db_info"] = self.vector_db.get_collection_info()
        
        return info
    
    def clear_documents(self) -> None:
        """
        清空所有文档
        """
        try:
            if self.use_llama_index:
                self.llama_index_rag.clear_index()
            else:
                self.vector_db.reset_database()
            
            logger.info("文档已清空")
            
        except Exception as e:
            logger.error(f"清空文档失败: {e}")
            raise
