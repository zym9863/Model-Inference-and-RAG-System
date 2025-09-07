"""
ChromaDB向量数据库管理器

提供ChromaDB向量数据库的管理和操作功能，包括集合管理、向量存储和检索。
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import uuid
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    ChromaDB向量数据库管理器
    
    提供向量存储、检索和集合管理功能。
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "rag_documents",
        embedding_function: Optional[Any] = None
    ):
        """
        初始化ChromaDB管理器
        
        Args:
            persist_directory: 数据持久化目录
            collection_name: 默认集合名称
            embedding_function: 嵌入函数（可选）
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        self.client = None
        self.collection = None
        
        self._setup_client()
        self._setup_collection()
    
    def _setup_client(self) -> None:
        """
        设置ChromaDB客户端
        """
        try:
            # 确保持久化目录存在
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # 创建客户端
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB客户端已创建，持久化目录: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"ChromaDB客户端创建失败: {e}")
            raise
    
    def _setup_collection(self) -> None:
        """
        设置默认集合
        """
        try:
            self.collection = self.get_or_create_collection(self.collection_name)
            logger.info(f"默认集合已设置: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"集合设置失败: {e}")
            raise
    
    def get_or_create_collection(
        self, 
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        获取或创建集合
        
        Args:
            collection_name: 集合名称
            metadata: 集合元数据
            
        Returns:
            ChromaDB集合对象
        """
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"获取现有集合: {collection_name}")
            
        except Exception:
            # 创建新集合
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {},
                embedding_function=self.embedding_function
            )
            logger.info(f"创建新集合: {collection_name}")
        
        return collection
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档文本列表
            metadatas: 文档元数据列表
            ids: 文档ID列表（可选，自动生成）
            embeddings: 预计算的嵌入向量（可选）
            collection_name: 目标集合名称（可选）
            
        Returns:
            文档ID列表
        """
        if not documents:
            return []
        
        # 选择集合
        collection = self.collection
        if collection_name:
            collection = self.get_or_create_collection(collection_name)
        
        # 生成ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        try:
            # 添加文档
            if embeddings is not None:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"成功添加 {len(documents)} 个文档到集合 {collection.name}")
            return ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def query_documents(
        self,
        query_texts: Union[str, List[str]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        查询相似文档
        
        Args:
            query_texts: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 包含的返回字段
            collection_name: 目标集合名称
            
        Returns:
            查询结果字典
        """
        # 选择集合
        collection = self.collection
        if collection_name:
            collection = self.get_or_create_collection(collection_name)
        
        # 处理查询文本
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        
        try:
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            logger.info(f"查询完成，返回 {len(results.get('documents', [[]]))} 组结果")
            return results
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """
        更新文档
        
        Args:
            ids: 文档ID列表
            documents: 新文档内容
            metadatas: 新元数据
            embeddings: 新嵌入向量
            collection_name: 目标集合名称
        """
        # 选择集合
        collection = self.collection
        if collection_name:
            collection = self.get_or_create_collection(collection_name)
        
        try:
            collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"成功更新 {len(ids)} 个文档")
            
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            raise
    
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """
        删除文档
        
        Args:
            ids: 要删除的文档ID列表
            where: 删除条件
            collection_name: 目标集合名称
        """
        # 选择集合
        collection = self.collection
        if collection_name:
            collection = self.get_or_create_collection(collection_name)
        
        try:
            collection.delete(ids=ids, where=where)
            
            if ids:
                logger.info(f"成功删除 {len(ids)} 个文档")
            else:
                logger.info("根据条件删除文档完成")
                
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合信息字典
        """
        # 选择集合
        collection = self.collection
        if collection_name:
            collection = self.get_or_create_collection(collection_name)
        
        try:
            count = collection.count()
            metadata = collection.metadata
            
            return {
                "name": collection.name,
                "count": count,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """
        列出所有集合
        
        Returns:
            集合名称列表
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
            
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> None:
        """
        删除集合
        
        Args:
            collection_name: 集合名称
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"成功删除集合: {collection_name}")
            
            # 如果删除的是默认集合，重新创建
            if collection_name == self.collection_name:
                self._setup_collection()
                
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise
    
    def reset_database(self) -> None:
        """
        重置数据库（删除所有数据）
        """
        try:
            self.client.reset()
            logger.info("数据库已重置")
            
            # 重新设置默认集合
            self._setup_collection()
            
        except Exception as e:
            logger.error(f"重置数据库失败: {e}")
            raise
