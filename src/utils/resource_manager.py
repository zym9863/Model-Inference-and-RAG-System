"""
资源管理器

提供GPU内存清理、模型卸载和资源监控功能。
"""

import torch
import gc
import psutil
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    MODEL = "model"


@dataclass
class ResourceStats:
    """资源统计信息"""
    cpu_percent: float
    memory_used: float
    memory_total: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None


class ResourceManager:
    """
    资源管理器

    提供GPU内存清理、模型卸载和资源监控功能。
    """

    def __init__(
        self,
        auto_cleanup_threshold: float = 85.0,  # GPU内存使用率阈值
        monitoring_interval: float = 10.0,      # 监控间隔（秒）
        enable_auto_cleanup: bool = True        # 是否启用自动清理
    ):
        """
        初始化资源管理器

        Args:
            auto_cleanup_threshold: 自动清理阈值（GPU内存使用百分比）
            monitoring_interval: 资源监控间隔时间
            enable_auto_cleanup: 是否启用自动清理
        """
        self.auto_cleanup_threshold = auto_cleanup_threshold
        self.monitoring_interval = monitoring_interval
        self.enable_auto_cleanup = enable_auto_cleanup

        # 资源监控
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._cleanup_callbacks: List[Callable] = []

        # GPU可用性检查
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            logger.info(f"检测到 {self.device_count} 个GPU设备")
        else:
            self.device_count = 0
            logger.info("未检测到GPU设备")

    def start_monitoring(self) -> None:
        """启动资源监控"""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("资源监控已在运行")
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("资源监控已启动")

    def stop_monitoring(self) -> None:
        """停止资源监控"""
        if self._monitoring_thread is None:
            return

        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        logger.info("资源监控已停止")

    def _monitoring_loop(self) -> None:
        """资源监控循环"""
        while not self._stop_monitoring.is_set():
            try:
                stats = self.get_resource_stats()

                # 检查GPU内存使用率
                if (self.enable_auto_cleanup and
                    stats.gpu_memory_percent is not None and
                    stats.gpu_memory_percent > self.auto_cleanup_threshold):

                    logger.warning(
                        f"GPU内存使用率过高: {stats.gpu_memory_percent:.1f}%，"
                        f"触发自动清理"
                    )
                    self.cleanup_gpu_memory()

                # 执行自定义清理回调
                for callback in self._cleanup_callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"清理回调执行失败: {e}")

            except Exception as e:
                logger.error(f"资源监控出错: {e}")

            # 等待下次检查
            self._stop_monitoring.wait(self.monitoring_interval)

    def add_cleanup_callback(self, callback: Callable[[ResourceStats], None]) -> None:
        """
        添加清理回调函数

        Args:
            callback: 回调函数，接收ResourceStats参数
        """
        self._cleanup_callbacks.append(callback)

    def cleanup_gpu_memory(self, device: Optional[int] = None) -> Dict[str, float]:
        """
        清理GPU内存

        Args:
            device: 指定GPU设备，None表示所有设备

        Returns:
            清理前后的内存使用情况
        """
        if not self.cuda_available:
            logger.warning("GPU不可用，跳过GPU内存清理")
            return {}

        before_stats = {}
        after_stats = {}

        devices = [device] if device is not None else list(range(self.device_count))

        for dev in devices:
            try:
                with torch.cuda.device(dev):
                    # 记录清理前状态
                    before_allocated = torch.cuda.memory_allocated(dev)
                    before_cached = torch.cuda.memory_reserved(dev)

                    # 执行清理
                    torch.cuda.empty_cache()

                    # 强制垃圾回收
                    gc.collect()

                    # 再次清理缓存
                    torch.cuda.empty_cache()

                    # 记录清理后状态
                    after_allocated = torch.cuda.memory_allocated(dev)
                    after_cached = torch.cuda.memory_reserved(dev)

                    freed_allocated = before_allocated - after_allocated
                    freed_cached = before_cached - after_cached

                    before_stats[f"gpu_{dev}"] = {
                        "allocated_mb": before_allocated / 1024 / 1024,
                        "cached_mb": before_cached / 1024 / 1024
                    }
                    after_stats[f"gpu_{dev}"] = {
                        "allocated_mb": after_allocated / 1024 / 1024,
                        "cached_mb": after_cached / 1024 / 1024,
                        "freed_allocated_mb": freed_allocated / 1024 / 1024,
                        "freed_cached_mb": freed_cached / 1024 / 1024
                    }

                    logger.info(
                        f"GPU {dev} 内存清理完成: "
                        f"释放分配内存 {freed_allocated / 1024 / 1024:.1f}MB, "
                        f"释放缓存内存 {freed_cached / 1024 / 1024:.1f}MB"
                    )

            except Exception as e:
                logger.error(f"清理GPU {dev} 内存失败: {e}")

        return {"before": before_stats, "after": after_stats}

    def cleanup_cpu_memory(self) -> None:
        """清理CPU内存"""
        try:
            # 强制垃圾回收
            collected = gc.collect()
            logger.info(f"CPU内存清理完成，回收对象数: {collected}")
        except Exception as e:
            logger.error(f"CPU内存清理失败: {e}")

    def unload_model(self, model: torch.nn.Module) -> bool:
        """
        卸载模型

        Args:
            model: 要卸载的PyTorch模型

        Returns:
            是否成功卸载
        """
        try:
            if model is None:
                return True

            # 将模型移到CPU
            model.cpu()

            # 删除模型引用
            del model

            # 清理GPU内存
            self.cleanup_gpu_memory()

            # CPU内存清理
            self.cleanup_cpu_memory()

            logger.info("模型卸载完成")
            return True

        except Exception as e:
            logger.error(f"模型卸载失败: {e}")
            return False

    def get_resource_stats(self) -> ResourceStats:
        """
        获取资源使用统计

        Returns:
            资源统计信息
        """
        # CPU和内存统计
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        stats = ResourceStats(
            cpu_percent=cpu_percent,
            memory_used=memory.used / 1024 / 1024 / 1024,  # GB
            memory_total=memory.total / 1024 / 1024 / 1024,  # GB
            memory_percent=memory.percent
        )

        # GPU统计
        if self.cuda_available and torch.cuda.device_count() > 0:
            try:
                device = 0  # 使用主GPU
                allocated = torch.cuda.memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory

                stats.gpu_memory_used = allocated / 1024 / 1024 / 1024  # GB
                stats.gpu_memory_total = total / 1024 / 1024 / 1024  # GB
                stats.gpu_memory_percent = (allocated / total) * 100

                # GPU利用率（需要nvidia-ml-py库，这里简化处理）
                stats.gpu_utilization = None

            except Exception as e:
                logger.debug(f"获取GPU统计失败: {e}")

        return stats

    def print_resource_stats(self) -> None:
        """打印资源使用统计"""
        stats = self.get_resource_stats()

        print("\n" + "="*50)
        print("资源使用统计")
        print("="*50)
        print(f"CPU使用率: {stats.cpu_percent:.1f}%")
        print(f"内存使用: {stats.memory_used:.2f}GB / {stats.memory_total:.2f}GB "
              f"({stats.memory_percent:.1f}%)")

        if stats.gpu_memory_total is not None:
            print(f"GPU内存: {stats.gpu_memory_used:.2f}GB / {stats.gpu_memory_total:.2f}GB "
                  f"({stats.gpu_memory_percent:.1f}%)")
        else:
            print("GPU: 不可用")
        print("="*50)

    @contextmanager
    def memory_management_context(self, cleanup_on_exit: bool = True):
        """
        内存管理上下文管理器

        Args:
            cleanup_on_exit: 退出时是否执行清理

        Usage:
            with resource_manager.memory_management_context():
                # 执行可能消耗大量内存的操作
                result = some_memory_intensive_operation()
        """
        # 记录进入时的内存状态
        initial_stats = self.get_resource_stats()
        logger.info(f"进入内存管理上下文，初始GPU内存使用: "
                   f"{initial_stats.gpu_memory_percent or 0:.1f}%")

        try:
            yield
        finally:
            if cleanup_on_exit:
                self.cleanup_gpu_memory()
                self.cleanup_cpu_memory()

                final_stats = self.get_resource_stats()
                logger.info(f"退出内存管理上下文，最终GPU内存使用: "
                           f"{final_stats.gpu_memory_percent or 0:.1f}%")

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        获取GPU设备信息

        Returns:
            GPU信息字典
        """
        if not self.cuda_available:
            return {"cuda_available": False}

        info = {
            "cuda_available": True,
            "device_count": self.device_count,
            "devices": []
        }

        for i in range(self.device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024 / 1024 / 1024,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                }
                info["devices"].append(device_info)
            except Exception as e:
                logger.error(f"获取GPU {i} 信息失败: {e}")

        return info

    def optimize_memory_usage(self) -> None:
        """内存使用优化"""
        try:
            # PyTorch内存分配器优化
            if self.cuda_available:
                # 设置内存分配策略
                torch.cuda.empty_cache()

                # 启用内存池
                if hasattr(torch.cuda, 'memory_pool'):
                    logger.info("启用CUDA内存池")

                # 设置内存碎片整理
                if hasattr(torch.cuda, 'memory_defrag'):
                    torch.cuda.memory_defrag()

            # CPU内存优化
            gc.set_threshold(700, 10, 10)  # 调整垃圾回收阈值

            logger.info("内存使用优化完成")

        except Exception as e:
            logger.error(f"内存优化失败: {e}")

    def __enter__(self):
        """进入上下文时启动监控"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时停止监控和清理"""
        self.stop_monitoring()
        if self.cuda_available:
            self.cleanup_gpu_memory()
        self.cleanup_cpu_memory()