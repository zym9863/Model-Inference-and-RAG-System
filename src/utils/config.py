"""
配置管理模块

提供系统配置的加载、验证和管理功能。
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    llm_model_name: str = "Qwen/Qwen3-8B"
    embedding_model_name: str = "google/embeddinggemma-300m"
    device: str = "auto"
    cache_dir: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None


@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    persist_directory: str = "./data/chromadb"
    collection_name: str = "rag_documents"
    embedding_function: Optional[str] = None


@dataclass
class RAGConfig:
    """RAG系统配置"""
    use_llama_index: bool = True
    max_context_length: int = 1500
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class GenerationConfig:
    """文本生成配置"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "./logs/rag_system.log"
    max_file_size: str = "10MB"
    backup_count: int = 5


class Config:
    """
    系统配置管理器
    
    负责加载、验证和管理所有系统配置。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or self._get_default_config_path()
        
        # 配置对象
        self.model = ModelConfig()
        self.vectordb = VectorDBConfig()
        self.rag = RAGConfig()
        self.generation = GenerationConfig()
        self.logging = LoggingConfig()
        
        # 加载配置
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """
        获取默认配置文件路径
        
        Returns:
            默认配置文件路径
        """
        # 尝试多个可能的配置文件位置
        possible_paths = [
            "./config/config.yaml",
            "./config.yaml",
            os.path.expanduser("~/.rag_system/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 返回默认路径
        return "./config/config.yaml"
    
    def load_config(self) -> None:
        """
        加载配置文件
        """
        try:
            if os.path.exists(self.config_path):
                logger.info(f"正在加载配置文件: {self.config_path}")
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                self._update_from_dict(config_data)
                logger.info("配置文件加载完成")
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                self.save_config()  # 保存默认配置
                
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.info("使用默认配置")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        从字典更新配置
        
        Args:
            config_data: 配置数据字典
        """
        # 更新模型配置
        if "model" in config_data:
            model_data = config_data["model"]
            for key, value in model_data.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # 更新向量数据库配置
        if "vectordb" in config_data:
            vectordb_data = config_data["vectordb"]
            for key, value in vectordb_data.items():
                if hasattr(self.vectordb, key):
                    setattr(self.vectordb, key, value)
        
        # 更新RAG配置
        if "rag" in config_data:
            rag_data = config_data["rag"]
            for key, value in rag_data.items():
                if hasattr(self.rag, key):
                    setattr(self.rag, key, value)
        
        # 更新生成配置
        if "generation" in config_data:
            generation_data = config_data["generation"]
            for key, value in generation_data.items():
                if hasattr(self.generation, key):
                    setattr(self.generation, key, value)
        
        # 更新日志配置
        if "logging" in config_data:
            logging_data = config_data["logging"]
            for key, value in logging_data.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
    
    def save_config(self) -> None:
        """
        保存配置到文件
        """
        try:
            # 确保配置目录存在
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建配置字典
            config_data = {
                "model": asdict(self.model),
                "vectordb": asdict(self.vectordb),
                "rag": asdict(self.rag),
                "generation": asdict(self.generation),
                "logging": asdict(self.logging)
            }
            
            # 保存到文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存到: {self.config_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def update_config(self, section: str, **kwargs) -> None:
        """
        更新配置项
        
        Args:
            section: 配置节名称 ("model", "vectordb", "rag", "generation", "logging")
            **kwargs: 要更新的配置项
        """
        try:
            config_obj = getattr(self, section)
            
            for key, value in kwargs.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logger.info(f"更新配置 {section}.{key} = {value}")
                else:
                    logger.warning(f"未知配置项: {section}.{key}")
            
        except AttributeError:
            logger.error(f"未知配置节: {section}")
            raise
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        获取完整配置字典
        
        Returns:
            配置字典
        """
        return {
            "model": asdict(self.model),
            "vectordb": asdict(self.vectordb),
            "rag": asdict(self.rag),
            "generation": asdict(self.generation),
            "logging": asdict(self.logging)
        }
    
    def validate_config(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 验证模型配置
            if not self.model.llm_model_name:
                logger.error("LLM模型名称不能为空")
                return False
            
            if not self.model.embedding_model_name:
                logger.error("嵌入模型名称不能为空")
                return False
            
            # 验证向量数据库配置
            if not self.vectordb.persist_directory:
                logger.error("向量数据库持久化目录不能为空")
                return False
            
            if not self.vectordb.collection_name:
                logger.error("集合名称不能为空")
                return False
            
            # 验证RAG配置
            if self.rag.max_context_length <= 0:
                logger.error("最大上下文长度必须大于0")
                return False
            
            if self.rag.top_k_retrieval <= 0:
                logger.error("检索文档数量必须大于0")
                return False
            
            if not (0 <= self.rag.similarity_threshold <= 1):
                logger.error("相似度阈值必须在0-1之间")
                return False
            
            # 验证生成配置
            if self.generation.max_new_tokens <= 0:
                logger.error("最大生成token数必须大于0")
                return False
            
            if not (0 <= self.generation.temperature <= 2):
                logger.error("温度参数必须在0-2之间")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def get_env_override(self, key: str, default: Any = None) -> Any:
        """
        获取环境变量覆盖值
        
        Args:
            key: 环境变量键名
            default: 默认值
            
        Returns:
            环境变量值或默认值
        """
        env_key = f"RAG_SYSTEM_{key.upper()}"
        return os.getenv(env_key, default)
    
    def apply_env_overrides(self) -> None:
        """
        应用环境变量覆盖
        """
        # 模型配置环境变量覆盖
        llm_model = self.get_env_override("llm_model")
        if llm_model:
            self.model.llm_model_name = llm_model
            
        embedding_model = self.get_env_override("embedding_model")
        if embedding_model:
            self.model.embedding_model_name = embedding_model
        
        device = self.get_env_override("device")
        if device:
            self.model.device = device
        
        # 向量数据库配置环境变量覆盖
        persist_dir = self.get_env_override("persist_directory")
        if persist_dir:
            self.vectordb.persist_directory = persist_dir
        
        collection_name = self.get_env_override("collection_name")
        if collection_name:
            self.vectordb.collection_name = collection_name
        
        logger.info("环境变量覆盖已应用")
