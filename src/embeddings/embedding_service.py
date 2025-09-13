"""
嵌入模型服务

提供google/embeddinggemma-300m嵌入模型的文本向量化服务。
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional, Dict, Any
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    嵌入模型服务类
    
    使用google/embeddinggemma-300m模型提供文本向量化服务。
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        初始化嵌入服务
        
        Args:
            model_name: HuggingFace模型名称
            device: 设备类型 ("auto", "cuda", "cpu")
            cache_dir: 模型缓存目录
            max_length: 最大序列长度
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """
        获取计算设备
        
        Args:
            device: 设备字符串
            
        Returns:
            torch设备对象
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self) -> None:
        """
        加载tokenizer和嵌入模型
        """
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"嵌入模型加载完成，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def encode_text(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
            show_progress: 是否显示进度
            
        Returns:
            文本向量数组，形状为 (n_texts, embedding_dim)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("嵌入模型未正确加载")
        
        # 处理输入
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # 批处理编码
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            if show_progress:
                logger.info(f"处理批次 {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            batch_embeddings = self._encode_batch(batch_texts)
            embeddings.append(batch_embeddings)
        
        # 合并结果
        all_embeddings = np.vstack(embeddings)
        
        # 归一化
        if normalize:
            all_embeddings = self._normalize_embeddings(all_embeddings)
        
        return all_embeddings
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            批次向量数组
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 获取最后一层隐藏状态
                last_hidden_states = outputs.last_hidden_state
                
                # 平均池化 (忽略padding tokens)
                attention_mask = inputs['attention_mask']
                embeddings = self._mean_pooling(last_hidden_states, attention_mask)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            raise
    
    def _mean_pooling(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        平均池化
        
        Args:
            hidden_states: 隐藏状态张量 (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            
        Returns:
            池化后的向量 (batch_size, hidden_size)
        """
        # 扩展attention_mask维度
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # 计算加权平均
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        Args:
            embeddings: 向量数组
            
        Returns:
            归一化后的向量数组
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        return embeddings / norms
    
    def compute_similarity(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        计算向量相似度
        
        Args:
            embeddings1: 第一组向量
            embeddings2: 第二组向量
            
        Returns:
            相似度矩阵
        """
        # 确保向量已归一化
        embeddings1 = self._normalize_embeddings(embeddings1)
        embeddings2 = self._normalize_embeddings(embeddings2)
        
        # 计算余弦相似度
        return np.dot(embeddings1, embeddings2.T)
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            向量维度
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        return self.model.config.hidden_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {}
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "embedding_dim": self.get_embedding_dimension(),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None
        }
    
    def benchmark(self, texts: List[str], num_runs: int = 3) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            texts: 测试文本列表
            num_runs: 运行次数
            
        Returns:
            性能指标字典
        """
        if not texts:
            return {}
        
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            _ = self.encode_text(texts, normalize=True)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        throughput = len(texts) / avg_time
        
        return {
            "avg_time_seconds": avg_time,
            "throughput_texts_per_second": throughput,
            "num_texts": len(texts),
            "num_runs": num_runs
        }
