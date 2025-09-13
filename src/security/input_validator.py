"""
输入验证器

提供用户输入的验证、清理和安全检查功能。
"""

import re
import html
import bleach
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecurityException(Exception):
    """安全相关异常"""
    pass


class InputValidator:
    """
    输入验证器

    负责验证和清理所有用户输入，防止注入攻击和恶意输入。
    """

    # 配置常量
    MAX_QUERY_LENGTH = 10000
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 1000
    MAX_BATCH_SIZE = 100

    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {
        '.txt', '.md', '.pdf', '.doc', '.docx',
        '.xls', '.xlsx', '.ppt', '.pptx', '.json',
        '.csv', '.html', '.xml', '.rtf'
    }

    # 危险字符和模式
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script标签
        r'javascript:',              # JavaScript协议
        r'vbscript:',               # VBScript协议
        r'on\w+\s*=',               # 事件处理器
        r'eval\s*\(',               # eval函数
        r'exec\s*\(',               # exec函数
        r'import\s+os',             # 系统导入
        r'__import__',              # 动态导入
        r'\.\./',                   # 路径遍历
        r'\.\.\\',                  # Windows路径遍历
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化输入验证器

        Args:
            config: 配置字典，可覆盖默认设置
        """
        self.config = config or {}
        self._load_config()
        self._compile_patterns()

    def _load_config(self) -> None:
        """加载配置"""
        self.max_query_length = self.config.get('max_query_length', self.MAX_QUERY_LENGTH)
        self.max_filename_length = self.config.get('max_filename_length', self.MAX_FILENAME_LENGTH)
        self.max_path_length = self.config.get('max_path_length', self.MAX_PATH_LENGTH)
        self.max_batch_size = self.config.get('max_batch_size', self.MAX_BATCH_SIZE)
        self.allowed_extensions = set(self.config.get('allowed_extensions', self.ALLOWED_EXTENSIONS))

    def _compile_patterns(self) -> None:
        """编译正则表达式模式"""
        self.dangerous_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.DANGEROUS_PATTERNS
        ]

    def validate_query(self, query: str) -> str:
        """
        验证查询文本

        Args:
            query: 用户查询文本

        Returns:
            清理后的查询文本

        Raises:
            SecurityException: 当输入不安全时
        """
        if not isinstance(query, str):
            raise SecurityException("查询必须是字符串类型")

        # 检查长度
        if len(query) == 0:
            raise SecurityException("查询不能为空")

        if len(query) > self.max_query_length:
            raise SecurityException(f"查询长度不能超过{self.max_query_length}字符")

        # 清理HTML
        cleaned_query = self._sanitize_html(query)

        # 检查危险模式
        self._check_dangerous_patterns(cleaned_query)

        # 移除控制字符
        cleaned_query = self._remove_control_chars(cleaned_query)

        logger.info(f"查询验证通过，长度: {len(cleaned_query)}")
        return cleaned_query.strip()

    def validate_file_path(self, file_path: str) -> str:
        """
        验证文件路径

        Args:
            file_path: 文件路径

        Returns:
            标准化的安全路径

        Raises:
            SecurityException: 当路径不安全时
        """
        if not isinstance(file_path, str):
            raise SecurityException("文件路径必须是字符串类型")

        # 检查长度
        if len(file_path) > self.max_path_length:
            raise SecurityException(f"路径长度不能超过{self.max_path_length}字符")

        # 使用pathlib标准化路径
        try:
            path = Path(file_path)
            resolved_path = path.resolve()
        except Exception as e:
            raise SecurityException(f"无效的文件路径: {e}")

        # 检查路径遍历
        if '..' in str(resolved_path):
            raise SecurityException("不允许路径遍历")

        # 检查绝对路径（可选：根据需求决定是否允许）
        if not path.is_absolute():
            logger.warning(f"使用相对路径: {file_path}")

        # 检查文件扩展名
        if path.suffix.lower() not in self.allowed_extensions:
            raise SecurityException(f"不支持的文件类型: {path.suffix}")

        # 检查文件名长度
        if len(path.name) > self.max_filename_length:
            raise SecurityException(f"文件名长度不能超过{self.max_filename_length}字符")

        logger.info(f"路径验证通过: {resolved_path}")
        return str(resolved_path)

    def validate_batch_input(self, items: List[Any]) -> List[Any]:
        """
        验证批量输入

        Args:
            items: 输入项列表

        Returns:
            验证后的项列表

        Raises:
            SecurityException: 当批量输入不安全时
        """
        if not isinstance(items, list):
            raise SecurityException("批量输入必须是列表类型")

        if len(items) == 0:
            raise SecurityException("批量输入不能为空")

        if len(items) > self.max_batch_size:
            raise SecurityException(f"批量大小不能超过{self.max_batch_size}项")

        validated_items = []
        for i, item in enumerate(items):
            try:
                if isinstance(item, str):
                    validated_item = self.validate_query(item)
                elif isinstance(item, dict) and 'text' in item:
                    item['text'] = self.validate_query(item['text'])
                    validated_item = item
                else:
                    validated_item = item
                validated_items.append(validated_item)
            except SecurityException as e:
                logger.warning(f"批量输入项{i}验证失败: {e}")
                # 根据策略决定是跳过还是抛出异常
                continue

        if not validated_items:
            raise SecurityException("批量输入中没有有效项")

        logger.info(f"批量验证完成，有效项: {len(validated_items)}/{len(items)}")
        return validated_items

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证元数据

        Args:
            metadata: 元数据字典

        Returns:
            清理后的元数据

        Raises:
            SecurityException: 当元数据不安全时
        """
        if not isinstance(metadata, dict):
            raise SecurityException("元数据必须是字典类型")

        clean_metadata = {}

        for key, value in metadata.items():
            # 验证键名
            if not isinstance(key, str):
                logger.warning(f"跳过非字符串键: {key}")
                continue

            if len(key) > 100:  # 限制键名长度
                logger.warning(f"跳过过长的键: {key}")
                continue

            # 清理键名
            clean_key = self._sanitize_string(key)

            # 验证值
            if isinstance(value, str):
                clean_value = self._sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            else:
                # 其他类型转为字符串并清理
                clean_value = self._sanitize_string(str(value))

            clean_metadata[clean_key] = clean_value

        logger.info(f"元数据验证完成，字段数: {len(clean_metadata)}")
        return clean_metadata

    def _sanitize_html(self, text: str) -> str:
        """
        清理HTML内容

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        # 使用bleach清理HTML
        allowed_tags = []  # 不允许任何HTML标签
        allowed_attributes = {}

        cleaned = bleach.clean(
            text,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )

        # HTML实体解码
        cleaned = html.unescape(cleaned)

        return cleaned

    def _check_dangerous_patterns(self, text: str) -> None:
        """
        检查危险模式

        Args:
            text: 要检查的文本

        Raises:
            SecurityException: 发现危险模式时
        """
        for pattern in self.dangerous_patterns:
            if pattern.search(text):
                raise SecurityException(f"检测到危险模式: {pattern.pattern}")

    def _remove_control_chars(self, text: str) -> str:
        """
        移除控制字符

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        # 保留换行符和制表符，移除其他控制字符
        cleaned = ''.join(
            char for char in text
            if ord(char) >= 32 or char in '\n\t\r'
        )
        return cleaned

    def _sanitize_string(self, text: str) -> str:
        """
        通用字符串清理

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # HTML清理
        cleaned = self._sanitize_html(text)

        # 移除控制字符
        cleaned = self._remove_control_chars(cleaned)

        # 限制长度
        if len(cleaned) > 1000:  # 通用长度限制
            cleaned = cleaned[:1000] + "..."

        return cleaned.strip()

    def is_safe_content(self, content: str) -> bool:
        """
        检查内容是否安全（不抛出异常的版本）

        Args:
            content: 要检查的内容

        Returns:
            内容是否安全
        """
        try:
            self.validate_query(content)
            return True
        except SecurityException:
            return False