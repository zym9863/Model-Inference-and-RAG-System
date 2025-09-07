"""
日志系统模块

提供统一的日志记录功能。
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import colorlog


def setup_logger(
    name: str = "rag_system",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    format_string: Optional[str] = None,
    enable_color: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        max_file_size: 最大文件大小
        backup_count: 备份文件数量
        format_string: 日志格式字符串
        enable_color: 是否启用彩色输出
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 默认格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if enable_color:
        # 彩色格式
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
    else:
        # 普通格式
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析文件大小
        size_bytes = _parse_size(max_file_size)
        
        # 创建轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # 文件格式（不使用颜色）
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    解析大小字符串为字节数
    
    Args:
        size_str: 大小字符串，如 "10MB", "1GB"
        
    Returns:
        字节数
    """
    size_str = size_str.upper().strip()
    
    # 提取数字和单位
    import re
    match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?B?)', size_str)
    
    if not match:
        raise ValueError(f"无效的大小格式: {size_str}")
    
    number = float(match.group(1))
    unit = match.group(2)
    
    # 单位转换
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        '': 1  # 默认为字节
    }
    
    if unit not in multipliers:
        raise ValueError(f"不支持的单位: {unit}")
    
    return int(number * multipliers[unit])


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    日志记录器混入类
    
    为类提供便捷的日志记录功能。
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取类专用的日志记录器"""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """记录信息日志"""
        self.logger.info(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """记录警告日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """记录错误日志"""
        self.logger.error(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """记录调试日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def log_exception(self, message: str, *args, **kwargs) -> None:
        """记录异常日志"""
        self.logger.exception(message, *args, **kwargs)
