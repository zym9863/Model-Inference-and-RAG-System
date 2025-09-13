"""
访问控制模块

提供基于角色的访问控制(RBAC)和文档权限管理功能。
"""

from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class Permission(Enum):
    """权限枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    QUERY = "query"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    MANAGE_USERS = "manage_users"
    VIEW_STATS = "view_stats"
    MANAGE_DOCUMENTS = "manage_documents"


@dataclass
class Role:
    """角色定义"""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    is_system_role: bool = False


@dataclass
class DocumentACL:
    """文档访问控制列表"""
    document_id: str
    owner_id: str
    public_read: bool = False
    public_write: bool = False
    user_permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    role_permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class AccessControlException(Exception):
    """访问控制异常"""
    pass


class AccessControl:
    """
    访问控制管理器

    实现基于角色的访问控制(RBAC)和细粒度的文档权限管理。
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化访问控制管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "./data/access_control.json"

        # 角色和权限存储
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> role_names
        self.document_acls: Dict[str, DocumentACL] = {}  # document_id -> ACL

        self._initialize_default_roles()
        self._load_config()

    def _initialize_default_roles(self) -> None:
        """初始化默认角色"""
        # 管理员角色
        admin_role = Role(
            name="admin",
            description="系统管理员，拥有所有权限",
            permissions={
                Permission.READ, Permission.WRITE, Permission.DELETE,
                Permission.ADMIN, Permission.QUERY, Permission.UPLOAD,
                Permission.DOWNLOAD, Permission.MANAGE_USERS,
                Permission.VIEW_STATS, Permission.MANAGE_DOCUMENTS
            },
            is_system_role=True
        )

        # 普通用户角色
        user_role = Role(
            name="user",
            description="普通用户，具有基本查询和上传权限",
            permissions={
                Permission.READ, Permission.QUERY,
                Permission.UPLOAD, Permission.DOWNLOAD
            },
            is_system_role=True
        )

        # 只读用户角色
        readonly_role = Role(
            name="readonly",
            description="只读用户，仅能查询和下载",
            permissions={
                Permission.READ, Permission.QUERY, Permission.DOWNLOAD
            },
            is_system_role=True
        )

        # 高级用户角色
        advanced_user_role = Role(
            name="advanced_user",
            description="高级用户，具有文档管理权限",
            permissions={
                Permission.READ, Permission.WRITE, Permission.QUERY,
                Permission.UPLOAD, Permission.DOWNLOAD,
                Permission.MANAGE_DOCUMENTS, Permission.VIEW_STATS
            },
            is_system_role=True
        )

        self.roles = {
            "admin": admin_role,
            "user": user_role,
            "readonly": readonly_role,
            "advanced_user": advanced_user_role
        }

        logger.info("默认角色初始化完成")

    def _load_config(self) -> None:
        """从文件加载配置"""
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 加载用户角色
                self.user_roles = {
                    user_id: set(roles)
                    for user_id, roles in data.get('user_roles', {}).items()
                }

                # 加载文档ACL
                for doc_id, acl_data in data.get('document_acls', {}).items():
                    acl = DocumentACL(
                        document_id=acl_data['document_id'],
                        owner_id=acl_data['owner_id'],
                        public_read=acl_data.get('public_read', False),
                        public_write=acl_data.get('public_write', False),
                        user_permissions={
                            user_id: {Permission(p) for p in perms}
                            for user_id, perms in acl_data.get('user_permissions', {}).items()
                        },
                        role_permissions={
                            role_name: {Permission(p) for p in perms}
                            for role_name, perms in acl_data.get('role_permissions', {}).items()
                        },
                        created_at=acl_data.get('created_at', time.time()),
                        updated_at=acl_data.get('updated_at', time.time())
                    )
                    self.document_acls[doc_id] = acl

                logger.info(f"访问控制配置加载完成")

            except Exception as e:
                logger.error(f"加载访问控制配置失败: {e}")

    def _save_config(self) -> None:
        """保存配置到文件"""
        try:
            config_path = Path(self.config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'user_roles': {
                    user_id: list(roles)
                    for user_id, roles in self.user_roles.items()
                },
                'document_acls': {}
            }

            for doc_id, acl in self.document_acls.items():
                data['document_acls'][doc_id] = {
                    'document_id': acl.document_id,
                    'owner_id': acl.owner_id,
                    'public_read': acl.public_read,
                    'public_write': acl.public_write,
                    'user_permissions': {
                        user_id: [p.value for p in perms]
                        for user_id, perms in acl.user_permissions.items()
                    },
                    'role_permissions': {
                        role_name: [p.value for p in perms]
                        for role_name, perms in acl.role_permissions.items()
                    },
                    'created_at': acl.created_at,
                    'updated_at': acl.updated_at
                }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info("访问控制配置已保存")

        except Exception as e:
            logger.error(f"保存访问控制配置失败: {e}")

    def assign_role(self, user_id: str, role_name: str) -> None:
        """
        为用户分配角色

        Args:
            user_id: 用户ID
            role_name: 角色名称

        Raises:
            AccessControlException: 角色不存在时
        """
        if role_name not in self.roles:
            raise AccessControlException(f"角色不存在: {role_name}")

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role_name)
        self._save_config()

        logger.info(f"为用户 {user_id} 分配角色: {role_name}")

    def revoke_role(self, user_id: str, role_name: str) -> None:
        """
        撤销用户角色

        Args:
            user_id: 用户ID
            role_name: 角色名称
        """
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]

        self._save_config()
        logger.info(f"撤销用户 {user_id} 的角色: {role_name}")

    def get_user_roles(self, user_id: str) -> Set[str]:
        """
        获取用户角色

        Args:
            user_id: 用户ID

        Returns:
            用户角色集合
        """
        return self.user_roles.get(user_id, set())

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        获取用户所有权限

        Args:
            user_id: 用户ID

        Returns:
            用户权限集合
        """
        permissions = set()
        user_roles = self.get_user_roles(user_id)

        for role_name in user_roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)

        return permissions

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        检查用户是否具有指定权限

        Args:
            user_id: 用户ID
            permission: 要检查的权限

        Returns:
            是否具有权限
        """
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions

    def check_document_permission(
        self,
        user_id: str,
        document_id: str,
        permission: Permission
    ) -> bool:
        """
        检查用户对特定文档的权限

        Args:
            user_id: 用户ID
            document_id: 文档ID
            permission: 要检查的权限

        Returns:
            是否具有权限
        """
        # 检查管理员权限
        if self.check_permission(user_id, Permission.ADMIN):
            return True

        # 获取文档ACL
        acl = self.document_acls.get(document_id)
        if not acl:
            # 如果没有ACL，使用默认权限检查
            return self.check_permission(user_id, permission)

        # 检查文档所有者
        if acl.owner_id == user_id:
            return True

        # 检查公共权限
        if permission == Permission.READ and acl.public_read:
            return True
        if permission == Permission.WRITE and acl.public_write:
            return True

        # 检查用户特定权限
        user_perms = acl.user_permissions.get(user_id, set())
        if permission in user_perms:
            return True

        # 检查角色权限
        user_roles = self.get_user_roles(user_id)
        for role_name in user_roles:
            role_perms = acl.role_permissions.get(role_name, set())
            if permission in role_perms:
                return True

        return False

    def create_document_acl(
        self,
        document_id: str,
        owner_id: str,
        public_read: bool = False,
        public_write: bool = False
    ) -> DocumentACL:
        """
        创建文档访问控制列表

        Args:
            document_id: 文档ID
            owner_id: 文档所有者ID
            public_read: 是否公开可读
            public_write: 是否公开可写

        Returns:
            创建的ACL对象
        """
        acl = DocumentACL(
            document_id=document_id,
            owner_id=owner_id,
            public_read=public_read,
            public_write=public_write
        )

        self.document_acls[document_id] = acl
        self._save_config()

        logger.info(f"创建文档ACL: {document_id}")
        return acl

    def grant_document_permission(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        role_name: Optional[str] = None,
        permission: Permission = Permission.READ
    ) -> None:
        """
        授予文档权限

        Args:
            document_id: 文档ID
            user_id: 用户ID（与role_name二选一）
            role_name: 角色名称（与user_id二选一）
            permission: 要授予的权限

        Raises:
            AccessControlException: 参数错误或文档不存在时
        """
        if not user_id and not role_name:
            raise AccessControlException("必须指定用户ID或角色名称")

        if user_id and role_name:
            raise AccessControlException("不能同时指定用户ID和角色名称")

        acl = self.document_acls.get(document_id)
        if not acl:
            raise AccessControlException(f"文档ACL不存在: {document_id}")

        if user_id:
            if user_id not in acl.user_permissions:
                acl.user_permissions[user_id] = set()
            acl.user_permissions[user_id].add(permission)
            logger.info(f"为用户 {user_id} 授予文档 {document_id} 的 {permission.value} 权限")

        if role_name:
            if role_name not in self.roles:
                raise AccessControlException(f"角色不存在: {role_name}")
            if role_name not in acl.role_permissions:
                acl.role_permissions[role_name] = set()
            acl.role_permissions[role_name].add(permission)
            logger.info(f"为角色 {role_name} 授予文档 {document_id} 的 {permission.value} 权限")

        acl.updated_at = time.time()
        self._save_config()

    def revoke_document_permission(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        role_name: Optional[str] = None,
        permission: Permission = Permission.READ
    ) -> None:
        """
        撤销文档权限

        Args:
            document_id: 文档ID
            user_id: 用户ID（与role_name二选一）
            role_name: 角色名称（与user_id二选一）
            permission: 要撤销的权限
        """
        acl = self.document_acls.get(document_id)
        if not acl:
            return

        if user_id and user_id in acl.user_permissions:
            acl.user_permissions[user_id].discard(permission)
            if not acl.user_permissions[user_id]:
                del acl.user_permissions[user_id]

        if role_name and role_name in acl.role_permissions:
            acl.role_permissions[role_name].discard(permission)
            if not acl.role_permissions[role_name]:
                del acl.role_permissions[role_name]

        acl.updated_at = time.time()
        self._save_config()

    def delete_document_acl(self, document_id: str) -> None:
        """
        删除文档ACL

        Args:
            document_id: 文档ID
        """
        if document_id in self.document_acls:
            del self.document_acls[document_id]
            self._save_config()
            logger.info(f"删除文档ACL: {document_id}")

    def get_accessible_documents(self, user_id: str, permission: Permission) -> List[str]:
        """
        获取用户可访问的文档列表

        Args:
            user_id: 用户ID
            permission: 权限类型

        Returns:
            可访问的文档ID列表
        """
        accessible_docs = []

        for doc_id, acl in self.document_acls.items():
            if self.check_document_permission(user_id, doc_id, permission):
                accessible_docs.append(doc_id)

        return accessible_docs

    def get_document_permissions(self, user_id: str, document_id: str) -> Set[Permission]:
        """
        获取用户对特定文档的所有权限

        Args:
            user_id: 用户ID
            document_id: 文档ID

        Returns:
            权限集合
        """
        permissions = set()

        # 检查每种权限
        for permission in Permission:
            if self.check_document_permission(user_id, document_id, permission):
                permissions.add(permission)

        return permissions

    def get_stats(self) -> Dict[str, Any]:
        """
        获取访问控制统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_roles': len(self.roles),
            'system_roles': sum(1 for role in self.roles.values() if role.is_system_role),
            'custom_roles': sum(1 for role in self.roles.values() if not role.is_system_role),
            'users_with_roles': len(self.user_roles),
            'total_documents': len(self.document_acls),
            'public_readable_docs': sum(1 for acl in self.document_acls.values() if acl.public_read),
            'public_writable_docs': sum(1 for acl in self.document_acls.values() if acl.public_write)
        }

    def create_custom_role(
        self,
        name: str,
        description: str,
        permissions: List[Permission]
    ) -> Role:
        """
        创建自定义角色

        Args:
            name: 角色名称
            description: 角色描述
            permissions: 权限列表

        Returns:
            创建的角色对象

        Raises:
            AccessControlException: 角色已存在时
        """
        if name in self.roles:
            raise AccessControlException(f"角色已存在: {name}")

        role = Role(
            name=name,
            description=description,
            permissions=set(permissions),
            is_system_role=False
        )

        self.roles[name] = role
        logger.info(f"创建自定义角色: {name}")

        return role

    def delete_custom_role(self, name: str) -> None:
        """
        删除自定义角色

        Args:
            name: 角色名称

        Raises:
            AccessControlException: 尝试删除系统角色时
        """
        if name not in self.roles:
            return

        role = self.roles[name]
        if role.is_system_role:
            raise AccessControlException(f"不能删除系统角色: {name}")

        # 从所有用户中移除该角色
        for user_id in list(self.user_roles.keys()):
            self.user_roles[user_id].discard(name)
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]

        # 从所有文档ACL中移除该角色
        for acl in self.document_acls.values():
            if name in acl.role_permissions:
                del acl.role_permissions[name]
                acl.updated_at = time.time()

        del self.roles[name]
        self._save_config()

        logger.info(f"删除自定义角色: {name}")