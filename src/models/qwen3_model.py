"""
Qwen3-8B 4bit量化模型加载器

提供Qwen3-8B模型的4bit量化加载、推理和文本生成功能。
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Qwen3Model:
    """
    Qwen3-8B 4bit量化模型类
    
    支持4bit量化加载，提供高效的文本生成和推理功能。
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # 使用可用的Qwen模型
        device: str = "auto",
        max_memory: Optional[Dict[str, str]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        初始化Qwen3模型
        
        Args:
            model_name: HuggingFace模型名称
            device: 设备类型 ("auto", "cuda", "cpu")
            max_memory: 最大内存限制
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device
        self.max_memory = max_memory
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        self._setup_quantization_config()
        self._load_model()
        self._setup_generation_config()
    
    def _setup_quantization_config(self) -> None:
        """
        设置4bit量化配置
        """
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("4bit量化配置已设置")
    
    def _load_model(self) -> None:
        """
        加载tokenizer和模型
        """
        try:
            # 加载tokenizer
            logger.info(f"正在加载tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info(f"正在加载4bit量化模型: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map=self.device,
                max_memory=self.max_memory,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _setup_generation_config(self) -> None:
        """
        设置生成配置
        """
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        logger.info("生成配置已设置")
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未正确加载")
        
        # 更新生成配置
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            )
            
            # 移动到设备
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    **kwargs
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        对话模式生成
        
        Args:
            messages: 对话消息列表，格式: [{"role": "user", "content": "..."}]
            **kwargs: 生成参数
            
        Returns:
            模型回复
        """
        # 构建对话提示
        prompt = self._build_chat_prompt(messages)
        return self.generate_text(prompt, **kwargs)
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        构建对话提示格式
        
        Args:
            messages: 对话消息列表
            
        Returns:
            格式化的提示文本
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
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
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "quantization": "4bit",
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', None)
        }
    
    def clear_cache(self) -> None:
        """
        清理GPU缓存
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
