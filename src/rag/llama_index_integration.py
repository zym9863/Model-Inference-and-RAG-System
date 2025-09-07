"""
LlamaIndex RAG框架集成

使用LlamaIndex构建RAG管道，集成向量检索和文档处理。
"""

from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    ServiceContext
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

import chromadb
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LlamaIndexRAG:
    """
    LlamaIndex RAG框架集成类
    
    提供基于LlamaIndex的RAG管道构建和查询功能。
    """
    
    def __init__(
        self,
        embedding_model_name: str = "google/embeddinggemma-300m",
        llm_model_name: str = "Qwen/Qwen3-8B",
        chroma_persist_dir: str = "./data/chromadb",
        collection_name: str = "rag_documents",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.7
    ):
        """
        初始化LlamaIndex RAG系统
        
        Args:
            embedding_model_name: 嵌入模型名称
            llm_model_name: 语言模型名称
            chroma_persist_dir: ChromaDB持久化目录
            collection_name: 集合名称
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            similarity_threshold: 相似度阈值
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        
        self.embed_model = None
        self.llm = None
        self.vector_store = None
        self.index = None
        self.query_engine = None
        
        self._setup_models()
        self._setup_vector_store()
        self._setup_index()
    
    def _setup_models(self) -> None:
        """
        设置嵌入模型和语言模型
        """
        try:
            # 设置嵌入模型
            logger.info(f"正在设置嵌入模型: {self.embedding_model_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name,
                trust_remote_code=True
            )
            
            # 设置语言模型（用于生成）
            logger.info(f"正在设置语言模型: {self.llm_model_name}")
            try:
                self.llm = HuggingFaceLLM(
                    model_name=self.llm_model_name,
                    tokenizer_name=self.llm_model_name,
                    context_window=2048,
                    max_new_tokens=512,
                    generate_kwargs={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True,
                        "pad_token_id": None  # 让模型自己处理
                    },
                    model_kwargs={
                        "torch_dtype": "float16",
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "quantization_config": {
                            "load_in_4bit": True,
                            "bnb_4bit_compute_dtype": "float16",
                            "bnb_4bit_use_double_quant": True,
                            "bnb_4bit_quant_type": "nf4"
                        }
                    },
                    tokenizer_kwargs={
                        "trust_remote_code": True,
                        "padding_side": "left"  # 对于生成任务使用左填充
                    },
                    device_map="auto",
                    stopping_ids=[]
                )
                logger.info("HuggingFaceLLM 初始化成功")
            except Exception as e:
                logger.warning(f"HuggingFaceLLM 初始化失败: {e}")
                logger.info("将使用默认LLM配置")
                # 如果HuggingFaceLLM初始化失败，使用更简单的配置
                try:
                    from llama_index.llms.openai import OpenAI
                    self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
                    logger.info("使用OpenAI替代模型")
                except ImportError:
                    logger.warning("无法导入OpenAI模型，将使用None")
                    self.llm = None
            
            # 设置全局配置
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap
            
            logger.info("模型设置完成")
            
        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            raise
    
    def _setup_vector_store(self) -> None:
        """
        设置向量存储
        """
        try:
            # 确保目录存在
            self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建ChromaDB客户端
            chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir)
            )
            
            # 获取或创建集合
            chroma_collection = chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            
            # 创建向量存储
            self.vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection
            )
            
            logger.info(f"向量存储设置完成: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"向量存储设置失败: {e}")
            raise
    
    def _setup_index(self) -> None:
        """
        设置向量索引
        """
        try:
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 创建或加载索引
            try:
                # 尝试从现有向量存储加载索引
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context
                )
                logger.info("从现有向量存储加载索引")
                
            except Exception:
                # 创建新索引
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context
                )
                logger.info("创建新的向量索引")
            
            # 设置查询引擎
            self._setup_query_engine()
            
        except Exception as e:
            logger.error(f"索引设置失败: {e}")
            raise
    
    def _setup_query_engine(self) -> None:
        """
        设置查询引擎
        """
        try:
            # 创建检索器
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5
            )
            
            # 创建后处理器
            postprocessor = SimilarityPostprocessor(
                similarity_cutoff=self.similarity_threshold
            )
            
            # 创建查询引擎
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[postprocessor]
            )
            
            logger.info("查询引擎设置完成")
            
        except Exception as e:
            logger.error(f"查询引擎设置失败: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Union[str, Dict[str, Any]]], 
        show_progress: bool = True
    ) -> None:
        """
        添加文档到索引
        
        Args:
            documents: 文档列表，可以是字符串或包含text和metadata的字典
            show_progress: 是否显示进度
        """
        if not documents:
            return
        
        try:
            # 转换为Document对象
            doc_objects = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    doc_obj = Document(
                        text=doc,
                        metadata={"source": f"document_{i}"}
                    )
                elif isinstance(doc, dict):
                    doc_obj = Document(
                        text=doc.get("text", ""),
                        metadata=doc.get("metadata", {"source": f"document_{i}"})
                    )
                else:
                    continue
                
                doc_objects.append(doc_obj)
            
            if show_progress:
                logger.info(f"正在处理 {len(doc_objects)} 个文档")
            
            # 解析文档为节点
            parser = SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(doc_objects)
            
            # 添加到索引
            self.index.insert_nodes(nodes)
            
            # 重新设置查询引擎
            self._setup_query_engine()
            
            logger.info(f"成功添加 {len(doc_objects)} 个文档，生成 {len(nodes)} 个节点")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def query(
        self, 
        query_text: str, 
        similarity_top_k: Optional[int] = None,
        response_mode: str = "compact",
        fallback_to_custom_llm: bool = True
    ) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            query_text: 查询文本
            similarity_top_k: 检索的相似文档数量
            response_mode: 响应模式
            fallback_to_custom_llm: 是否在LlamaIndex查询失败时回退到自定义LLM
            
        Returns:
            查询结果字典
        """
        if self.query_engine is None:
            raise RuntimeError("查询引擎未初始化")
        
        try:
            # 更新检索参数
            if similarity_top_k is not None:
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=similarity_top_k
                )
                
                postprocessor = SimilarityPostprocessor(
                    similarity_cutoff=self.similarity_threshold
                )
                
                self.query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    node_postprocessors=[postprocessor]
                )
            
            # 执行查询
            logger.info(f"正在执行RAG查询: {query_text}")
            response = self.query_engine.query(query_text)
            
            # 调试日志：输出响应对象的信息
            logger.debug(f"查询响应对象类型: {type(response)}")
            logger.debug(f"响应对象属性: {dir(response)}")
            if hasattr(response, '__dict__'):
                logger.debug(f"响应对象详细信息: {response.__dict__}")
            
            # 提取源节点信息
            source_nodes = []
            if hasattr(response, 'source_nodes'):
                logger.info(f"找到 {len(response.source_nodes)} 个源节点")
                for node in response.source_nodes:
                    source_nodes.append({
                        "text": node.text,
                        "score": getattr(node, 'score', None),
                        "metadata": node.metadata
                    })
            else:
                logger.warning("响应对象没有 source_nodes 属性")
            
            # 提取响应文本，处理不同类型的响应对象
            response_text = ""
            logger.debug("正在提取响应文本...")
            
            if hasattr(response, 'response'):
                response_text = response.response
                logger.debug(f"使用 response.response: '{response_text}'")
            elif hasattr(response, 'text'):
                response_text = response.text
                logger.debug(f"使用 response.text: '{response_text}'")
            elif hasattr(response, 'content'):
                response_text = response.content
                logger.debug(f"使用 response.content: '{response_text}'")
            else:
                response_text = str(response)
                logger.debug(f"使用 str(response): '{response_text}'")
            
            # 如果响应为空且启用了回退模式，使用自定义LLM生成响应
            if (not response_text.strip() or response_text.strip().lower() in ["empty response", ""]) and fallback_to_custom_llm:
                logger.warning("LlamaIndex响应为空，尝试使用自定义LLM生成响应")
                response_text = self._generate_response_with_custom_llm(query_text, source_nodes)
            
            # 确保响应不为空
            if not response_text.strip():
                logger.warning(f"响应文本为空，原始响应对象: {response}")
                response_text = "抱歉，无法生成有效的回答。请检查系统配置或尝试其他查询。"
            
            result = {
                "query": query_text,
                "response": response_text,
                "source_nodes": source_nodes,
                "num_sources": len(source_nodes)
            }
            
            logger.info(f"查询完成，返回 {len(source_nodes)} 个源节点，响应长度: {len(response_text)}")
            return result
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise
    
    def _generate_response_with_custom_llm(self, query_text: str, source_nodes: List[Dict]) -> str:
        """
        使用自定义LLM生成响应（回退方案）
        
        Args:
            query_text: 查询文本
            source_nodes: 源节点列表
            
        Returns:
            生成的响应文本
        """
        try:
            # 导入自定义的Qwen3模型
            from ..models.qwen3_model import Qwen3Model
            
            # 初始化Qwen3模型
            logger.info("初始化自定义Qwen3模型作为回退方案")
            custom_llm = Qwen3Model(
                model_name=self.llm_model_name,
                device="auto"
            )
            
            # 构建上下文
            context_parts = []
            for i, node in enumerate(source_nodes[:3]):  # 只使用前3个最相关的文档
                context_parts.append(f"参考文档 {i+1}：{node['text']}")
            
            context = "\n\n".join(context_parts) if context_parts else "没有找到相关文档。"
            
            # 构建提示
            prompt = f"""基于以下提供的文档内容，请回答用户的问题。如果文档中没有相关信息，请说明无法从提供的文档中找到答案。

{context}

用户问题：{query_text}

请基于上述文档内容回答："""
            
            # 生成响应
            response = custom_llm.generate_text(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.7
            )
            
            logger.info("使用自定义LLM成功生成响应")
            return response
            
        except Exception as e:
            logger.error(f"自定义LLM回退方案失败: {e}")
            return f"基于检索到的相关文档，无法生成完整回答。请参考源文档内容。检索到 {len(source_nodes)} 个相关文档片段。"
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            索引信息字典
        """
        if self.index is None:
            return {}
        
        try:
            # 获取文档统计
            docstore = self.index.docstore
            num_docs = len(docstore.docs)
            
            return {
                "collection_name": self.collection_name,
                "num_documents": num_docs,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model_name
            }
            
        except Exception as e:
            logger.error(f"获取索引信息失败: {e}")
            return {}
    
    def clear_index(self) -> None:
        """
        清空索引
        """
        try:
            # 重新创建空索引
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context
            )
            
            # 重新设置查询引擎
            self._setup_query_engine()
            
            logger.info("索引已清空")
            
        except Exception as e:
            logger.error(f"清空索引失败: {e}")
            raise
