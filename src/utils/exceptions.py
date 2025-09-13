"""
异常处理模块

提供项目专用的异常类型定义和异常处理机制。
"""

import logging
import traceback
import sys
from typing import Any, Dict, Optional, Union, Callable, Type
from enum import Enum
from dataclasses import dataclass
import time
import torch

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """错误代码枚举"""
    # 通用错误
    UNKNOWN_ERROR = "E0001"
    INVALID_PARAMETER = "E0002"
    RESOURCE_NOT_FOUND = "E0003"
    PERMISSION_DENIED = "E0004"

    # 模型相关错误
    MODEL_LOAD_FAILED = "E1001"
    MODEL_NOT_LOADED = "E1002"
    MODEL_INFERENCE_FAILED = "E1003"
    MODEL_OUT_OF_MEMORY = "E1004"

    # 向量数据库错误
    VECTOR_DB_CONNECTION_FAILED = "E2001"
    VECTOR_DB_QUERY_FAILED = "E2002"
    VECTOR_DB_INSERT_FAILED = "E2003"

    # 文档处理错误
    DOCUMENT_PARSE_FAILED = "E3001"
    DOCUMENT_NOT_FOUND = "E3002"
    DOCUMENT_FORMAT_UNSUPPORTED = "E3003"

    # 安全相关错误
    AUTHENTICATION_FAILED = "E4001"
    AUTHORIZATION_FAILED = "E4002"
    INPUT_VALIDATION_FAILED = "E4003"

    # 网络和IO错误
    NETWORK_ERROR = "E5001"
    FILE_IO_ERROR = "E5002"
    TIMEOUT_ERROR = "E5003"


@dataclass
class ErrorContext:
    """错误上下文信息"""
    error_code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = None
    user_message: Optional[str] = None  # 面向用户的友好消息

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RAGException(Exception):
    """RAG系统基础异常"""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化RAG异常

        Args:
            error_code: 错误代码
            message: 错误消息
            details: 错误详情
            user_message: 用户友好消息
            cause: 原始异常
        """
        super().__init__(message)
        self.context = ErrorContext(
            error_code=error_code,
            message=message,
            details=details or {},
            user_message=user_message
        )
        self.cause = cause

    @property
    def error_code(self) -> ErrorCode:
        return self.context.error_code

    @property
    def user_message(self) -> str:
        """获取用户友好的错误消息"""
        return self.context.user_message or self.context.message

    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式"""
        result = {
            "error_code": self.error_code.value,
            "message": self.context.message,
            "user_message": self.user_message,
            "timestamp": self.context.timestamp,
            "details": self.context.details
        }

        if self.cause:
            result["cause"] = str(self.cause)

        return result


class ModelException(RAGException):
    """模型相关异常"""
    pass


class VectorDBException(RAGException):
    """向量数据库异常"""
    pass


class DocumentException(RAGException):
    """文档处理异常"""
    pass


class SecurityException(RAGException):
    """安全相关异常"""
    pass


class NetworkException(RAGException):
    """网络异常"""
    pass


class ExceptionHandler:
    """
    异常处理器

    提供统一的异常处理、记录和转换功能。
    """

    def __init__(self, log_level: int = logging.ERROR):
        """
        初始化异常处理器

        Args:
            log_level: 异常日志级别
        """
        self.log_level = log_level
        self._error_handlers: Dict[Type[Exception], Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """设置默认异常处理器"""
        # 内存不足异常
        self.register_handler(
            (RuntimeError, torch.cuda.OutOfMemoryError if 'torch' in sys.modules else RuntimeError),
            self._handle_memory_error
        )

        # 文件不存在异常
        self.register_handler(FileNotFoundError, self._handle_file_not_found)

        # 权限异常
        self.register_handler(PermissionError, self._handle_permission_error)

        # 网络异常
        self.register_handler(
            (ConnectionError, TimeoutError),
            self._handle_network_error
        )

    def register_handler(
        self,
        exception_types: Union[Type[Exception], tuple],
        handler: Callable[[Exception], RAGException]
    ) -> None:
        """
        注册异常处理器

        Args:
            exception_types: 异常类型或类型元组
            handler: 处理函数
        """
        if isinstance(exception_types, tuple):
            for exc_type in exception_types:
                self._error_handlers[exc_type] = handler
        else:
            self._error_handlers[exception_types] = handler

    def handle_exception(
        self,
        exception: Exception,
        context: Optional[str] = None
    ) -> RAGException:
        """
        处理异常

        Args:
            exception: 原始异常
            context: 异常上下文描述

        Returns:
            处理后的RAG异常
        """
        # 如果已经是RAG异常，直接返回
        if isinstance(exception, RAGException):
            self._log_exception(exception, context)
            return exception

        # 查找合适的处理器
        for exc_type, handler in self._error_handlers.items():
            if isinstance(exception, exc_type):
                try:
                    rag_exception = handler(exception)
                    self._log_exception(rag_exception, context)
                    return rag_exception
                except Exception as e:
                    logger.error(f"异常处理器执行失败: {e}")

        # 默认处理
        rag_exception = self._handle_unknown_error(exception, context)
        self._log_exception(rag_exception, context)
        return rag_exception

    def _handle_memory_error(self, exception: Exception) -> ModelException:
        """处理内存不足异常"""
        if "out of memory" in str(exception).lower():
            return ModelException(
                error_code=ErrorCode.MODEL_OUT_OF_MEMORY,
                message=f"GPU内存不足: {exception}",
                user_message="模型内存不足，请尝试减少批处理大小或清理GPU内存",
                cause=exception
            )
        else:
            return ModelException(
                error_code=ErrorCode.MODEL_INFERENCE_FAILED,
                message=f"模型推理失败: {exception}",
                user_message="模型运行时出现错误，请稍后重试",
                cause=exception
            )

    def _handle_file_not_found(self, exception: FileNotFoundError) -> DocumentException:
        """处理文件不存在异常"""
        return DocumentException(
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            message=f"文件未找到: {exception}",
            user_message="指定的文件不存在，请检查文件路径",
            cause=exception
        )

    def _handle_permission_error(self, exception: PermissionError) -> SecurityException:
        """处理权限异常"""
        return SecurityException(
            error_code=ErrorCode.PERMISSION_DENIED,
            message=f"权限不足: {exception}",
            user_message="没有足够的权限执行此操作",
            cause=exception
        )

    def _handle_network_error(self, exception: Exception) -> NetworkException:
        """处理网络异常"""
        if isinstance(exception, TimeoutError):
            error_code = ErrorCode.TIMEOUT_ERROR
            user_message = "操作超时，请检查网络连接并重试"
        else:
            error_code = ErrorCode.NETWORK_ERROR
            user_message = "网络连接失败，请检查网络设置"

        return NetworkException(
            error_code=error_code,
            message=f"网络错误: {exception}",
            user_message=user_message,
            cause=exception
        )

    def _handle_unknown_error(
        self,
        exception: Exception,
        context: Optional[str] = None
    ) -> RAGException:
        """处理未知异常"""
        context_msg = f" (上下文: {context})" if context else ""

        return RAGException(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message=f"未知错误{context_msg}: {exception}",
            user_message="系统遇到未知错误，请联系技术支持",
            details={
                "exception_type": type(exception).__name__,
                "context": context
            },
            cause=exception
        )

    def _log_exception(
        self,
        exception: RAGException,
        context: Optional[str] = None
    ) -> None:
        """记录异常信息"""
        context_msg = f" [上下文: {context}]" if context else ""

        log_message = (
            f"异常 {exception.error_code.value}{context_msg}: "
            f"{exception.context.message}"
        )

        # 添加详细信息
        if exception.context.details:
            log_message += f" | 详情: {exception.context.details}"

        # 添加原始异常信息
        if exception.cause:
            log_message += f" | 原因: {exception.cause}"

        logger.log(self.log_level, log_message)

        # 在调试模式下记录完整的堆栈跟踪
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"异常堆栈跟踪:\n{traceback.format_exc()}")


# 全局异常处理器实例
default_exception_handler = ExceptionHandler()


def handle_exceptions(
    func: Optional[Callable] = None,
    *,
    exception_handler: ExceptionHandler = None,
    context: Optional[str] = None,
    reraise: bool = True
):
    """
    异常处理装饰器

    Args:
        func: 被装饰的函数
        exception_handler: 异常处理器实例
        context: 异常上下文描述
        reraise: 是否重新抛出异常

    Usage:
        @handle_exceptions(context="模型加载")
        def load_model():
            # 可能抛出异常的代码
            pass
    """
    def decorator(f: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                handler = exception_handler or default_exception_handler
                rag_exception = handler.handle_exception(e, context or f.__name__)

                if reraise:
                    raise rag_exception
                else:
                    return {"error": rag_exception.to_dict()}

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def safe_execute(
    func: Callable,
    *args,
    exception_handler: ExceptionHandler = None,
    context: Optional[str] = None,
    default_return: Any = None,
    **kwargs
) -> Union[Any, Dict[str, Any]]:
    """
    安全执行函数

    Args:
        func: 要执行的函数
        *args: 函数位置参数
        exception_handler: 异常处理器
        context: 异常上下文
        default_return: 异常时的默认返回值
        **kwargs: 函数关键字参数

    Returns:
        函数返回值或错误信息
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = exception_handler or default_exception_handler
        rag_exception = handler.handle_exception(e, context or func.__name__)

        if default_return is not None:
            return default_return
        else:
            return {"error": rag_exception.to_dict()}


class ErrorReporter:
    """错误报告器"""

    def __init__(self, max_reports: int = 100):
        """
        初始化错误报告器

        Args:
            max_reports: 最大保存的错误报告数量
        """
        self.max_reports = max_reports
        self.reports: List[Dict[str, Any]] = []

    def report_error(self, exception: RAGException, additional_info: Dict[str, Any] = None) -> None:
        """
        报告错误

        Args:
            exception: RAG异常
            additional_info: 额外信息
        """
        report = exception.to_dict()
        if additional_info:
            report["additional_info"] = additional_info

        self.reports.append(report)

        # 保持最大数量限制
        if len(self.reports) > self.max_reports:
            self.reports = self.reports[-self.max_reports:]

    def get_error_summary(self) -> Dict[str, Any]:
        """
        获取错误摘要

        Returns:
            错误统计摘要
        """
        if not self.reports:
            return {"total_errors": 0}

        # 按错误代码统计
        error_counts = {}
        for report in self.reports:
            error_code = report.get("error_code", "unknown")
            error_counts[error_code] = error_counts.get(error_code, 0) + 1

        # 最近的错误
        recent_errors = self.reports[-10:]

        return {
            "total_errors": len(self.reports),
            "error_counts": error_counts,
            "recent_errors": recent_errors,
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }

    def clear_reports(self) -> None:
        """清空错误报告"""
        self.reports.clear()


# 全局错误报告器
global_error_reporter = ErrorReporter()