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

# 添加资源管理器和异常处理导入
try:
    from ..utils.resource_manager import ResourceManager
    from ..utils.exceptions import (
        ModelException, ErrorCode, handle_exceptions
    )
except ImportError:
    # 向后兼容
    ResourceManager = None
    ModelException = Exception
    ErrorCode = None
    handle_exceptions = lambda **kwargs: lambda f: f

logger = logging.getLogger(__name__)


class Qwen3Model:
    """
    Qwen3-8B 4bit量化模型类
    
    支持4bit量化加载，提供高效的文本生成和推理功能。
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_memory: Optional[Dict[str, str]] = None,
        cache_dir: Optional[str] = None,
        enable_resource_management: bool = True,
        lazy_loading: bool = False
    ):
        """
        初始化Qwen3模型

        Args:
            model_name: HuggingFace模型名称
            device: 设备类型 ("auto", "cuda", "cpu")
            max_memory: 最大内存限制
            cache_dir: 模型缓存目录
            enable_resource_management: 是否启用资源管理
            lazy_loading: 是否启用延迟加载模式
        """
        self.model_name = model_name
        self.device = device
        self.max_memory = max_memory
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading

        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self._model_loaded = False

        # 初始化资源管理器
        self.resource_manager = None
        if enable_resource_management and ResourceManager is not None:
            self.resource_manager = ResourceManager()
            logger.info("资源管理器已启用")

        self._setup_quantization_config()

        if not lazy_loading:
            # 立即加载模式
            self._load_model()
            self._setup_generation_config()
        else:
            logger.info("延迟加载模式已启用，模型将在首次使用时加载")
    
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
    
    @handle_exceptions(context="模型加载")
    def _load_model(self) -> None:
        """
        加载tokenizer和模型
        """
        if self._model_loaded:
            return

        try:
            # 使用资源管理器的内存管理上下文
            context_manager = (
                self.resource_manager.memory_management_context(cleanup_on_exit=False)
                if self.resource_manager else
                self._null_context()
            )

            with context_manager:
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

                self._model_loaded = True
                logger.info("模型加载完成")

        except torch.cuda.OutOfMemoryError as e:
            if ErrorCode:
                raise ModelException(
                    error_code=ErrorCode.MODEL_OUT_OF_MEMORY,
                    message=f"GPU内存不足，无法加载模型: {e}",
                    user_message="模型加载需要更多GPU内存，请尝试关闭其他程序或使用更小的模型",
                    cause=e
                )
            else:
                raise
        except Exception as e:
            if ErrorCode:
                raise ModelException(
                    error_code=ErrorCode.MODEL_LOAD_FAILED,
                    message=f"模型加载失败: {e}",
                    user_message="模型加载失败，请检查模型名称和网络连接",
                    cause=e
                )
            else:
                logger.error(f"模型加载失败: {e}")
                raise

    def _null_context(self):
        """空上下文管理器，用于向后兼容"""
        from contextlib import nullcontext
        return nullcontext()

    def _ensure_model_loaded(self) -> None:
        """
        确保模型已加载（延迟加载的关键方法）
        """
        if not self._model_loaded:
            logger.info("延迟加载：正在加载模型...")
            self._load_model()
            if self.generation_config is None:
                self._setup_generation_config()

    @property
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        Returns:
            模型是否已加载
        """
        return self._model_loaded
    
    def _setup_generation_config(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> None:
        """
        设置生成配置

        Args:
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
        """
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        logger.info("生成配置已设置")
    
    @handle_exceptions(context="文本生成")
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
        # 确保模型已加载
        self._ensure_model_loaded()

        if self.model is None or self.tokenizer is None:
            if ErrorCode:
                raise ModelException(
                    error_code=ErrorCode.MODEL_NOT_LOADED,
                    message="模型未正确加载",
                    user_message="模型尚未加载，请稍后重试"
                )
            else:
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

        except torch.cuda.OutOfMemoryError as e:
            if ErrorCode:
                raise ModelException(
                    error_code=ErrorCode.MODEL_OUT_OF_MEMORY,
                    message=f"文本生成时GPU内存不足: {e}",
                    user_message="生成文本时内存不足，请尝试减少生成长度或清理GPU内存",
                    cause=e
                )
            else:
                raise
        except Exception as e:
            if ErrorCode:
                raise ModelException(
                    error_code=ErrorCode.MODEL_INFERENCE_FAILED,
                    message=f"文本生成失败: {e}",
                    user_message="文本生成过程中出现错误，请稍后重试",
                    cause=e
                )
            else:
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
        # 确保模型已加载
        self._ensure_model_loaded()

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
        info = {
            "model_name": self.model_name,
            "lazy_loading": self.lazy_loading,
            "is_loaded": self.is_loaded,
            "quantization": "4bit"
        }

        if self.is_loaded and self.model is not None:
            info.update({
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
                "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', None)
            })
        else:
            info.update({
                "device": "未加载",
                "dtype": "未加载",
                "vocab_size": None,
                "max_position_embeddings": None
            })

        return info
    
    def clear_cache(self) -> None:
        """
        清理GPU缓存
        """
        if self.resource_manager:
            self.resource_manager.cleanup_gpu_memory()
            logger.info("通过资源管理器清理GPU缓存")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")

    def unload_model(self) -> bool:
        """
        卸载模型并清理资源

        Returns:
            是否成功卸载
        """
        success = True

        try:
            if self.resource_manager and self.model:
                success = self.resource_manager.unload_model(self.model)
            else:
                # 手动清理
                if self.model:
                    self.model.cpu()
                    del self.model
                    self.model = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 清理tokenizer
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            logger.info(f"模型卸载{'成功' if success else '部分成功'}")
            return success

        except Exception as e:
            logger.error(f"模型卸载失败: {e}")
            return False

    def get_resource_stats(self) -> Dict[str, Any]:
        """
        获取资源使用统计

        Returns:
            资源统计信息
        """
        if self.resource_manager:
            stats = self.resource_manager.get_resource_stats()
            return {
                "cpu_percent": stats.cpu_percent,
                "memory_percent": stats.memory_percent,
                "gpu_memory_percent": stats.gpu_memory_percent,
                "gpu_memory_used_gb": stats.gpu_memory_used,
                "gpu_memory_total_gb": stats.gpu_memory_total
            }
        else:
            # 基本统计信息
            stats = {}
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory
                stats["gpu_memory_percent"] = (allocated / total) * 100
                stats["gpu_memory_used_gb"] = allocated / 1024 / 1024 / 1024
                stats["gpu_memory_total_gb"] = total / 1024 / 1024 / 1024

            return stats

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.unload_model()
        except Exception:
            pass  # 忽略析构函数中的异常
