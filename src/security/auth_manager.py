"""
Token认证管理器

提供基于Token的用户认证和会话管理功能。
"""

import jwt
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class Session:
    """会话信息"""
    session_id: str
    user_id: str
    token: str
    expires_at: datetime
    created_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuthenticationException(Exception):
    """认证异常"""
    pass


class AuthorizationException(Exception):
    """授权异常"""
    pass


class AuthManager:
    """
    Token认证管理器

    提供JWT Token生成、验证和用户会话管理功能。
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expiry_minutes: int = 60,
        refresh_token_expiry_days: int = 7,
        user_store_path: Optional[str] = None
    ):
        """
        初始化认证管理器

        Args:
            secret_key: JWT签名密钥，如果不提供会自动生成
            algorithm: JWT签名算法
            token_expiry_minutes: 访问token过期时间（分钟）
            refresh_token_expiry_days: 刷新token过期时间（天）
            user_store_path: 用户数据存储路径
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = algorithm
        self.token_expiry_minutes = token_expiry_minutes
        self.refresh_token_expiry_days = refresh_token_expiry_days
        self.user_store_path = user_store_path or "./data/users.json"

        # 内存存储（生产环境应使用数据库）
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.refresh_tokens: Dict[str, Dict[str, Any]] = {}

        self._load_users()
        self._create_default_admin()

    def _generate_secret_key(self) -> str:
        """生成随机密钥"""
        return secrets.token_urlsafe(64)

    def _load_users(self) -> None:
        """从文件加载用户数据"""
        user_path = Path(self.user_store_path)
        if user_path.exists():
            try:
                with open(user_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for user_data in data.get('users', []):
                    user = User(
                        user_id=user_data['user_id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        roles=user_data['roles'],
                        created_at=datetime.fromisoformat(user_data['created_at']),
                        last_login=datetime.fromisoformat(user_data['last_login'])
                        if user_data.get('last_login') else None,
                        is_active=user_data.get('is_active', True)
                    )
                    self.users[user.user_id] = user

                logger.info(f"已加载 {len(self.users)} 个用户")

            except Exception as e:
                logger.error(f"加载用户数据失败: {e}")

    def _save_users(self) -> None:
        """保存用户数据到文件"""
        try:
            user_path = Path(self.user_store_path)
            user_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'users': []
            }

            for user in self.users.values():
                user_data = {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'roles': user.roles,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_active': user.is_active
                }
                data['users'].append(user_data)

            with open(user_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"用户数据已保存")

        except Exception as e:
            logger.error(f"保存用户数据失败: {e}")

    def _create_default_admin(self) -> None:
        """创建默认管理员用户"""
        admin_id = "admin"
        if admin_id not in self.users:
            admin_user = User(
                user_id=admin_id,
                username="admin",
                email="admin@localhost",
                roles=["admin", "user"],
                created_at=datetime.now()
            )
            self.users[admin_id] = admin_user
            self._save_users()
            logger.info("已创建默认管理员用户")

    def create_user(
        self,
        username: str,
        email: str,
        roles: List[str] = None
    ) -> User:
        """
        创建新用户

        Args:
            username: 用户名
            email: 邮箱
            roles: 用户角色列表

        Returns:
            创建的用户对象

        Raises:
            AuthenticationException: 用户已存在时
        """
        if not username or not email:
            raise AuthenticationException("用户名和邮箱不能为空")

        # 检查用户名是否存在
        for user in self.users.values():
            if user.username == username:
                raise AuthenticationException("用户名已存在")
            if user.email == email:
                raise AuthenticationException("邮箱已存在")

        # 生成用户ID
        user_id = hashlib.sha256(
            (username + email + str(time.time())).encode()
        ).hexdigest()[:16]

        # 创建用户
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or ["user"],
            created_at=datetime.now()
        )

        self.users[user_id] = user
        self._save_users()

        logger.info(f"创建用户: {username} ({user_id})")
        return user

    def authenticate_user(self, username: str, password: str = None) -> Optional[User]:
        """
        用户认证（简化版，实际应用中应验证密码）

        Args:
            username: 用户名
            password: 密码（当前版本忽略）

        Returns:
            认证成功的用户对象，失败返回None
        """
        # 简化版认证：仅检查用户是否存在且激活
        # 实际应用中应该验证密码哈希
        for user in self.users.values():
            if user.username == username and user.is_active:
                user.last_login = datetime.now()
                self._save_users()
                logger.info(f"用户认证成功: {username}")
                return user

        logger.warning(f"用户认证失败: {username}")
        return None

    def generate_token(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成访问token和刷新token

        Args:
            user: 用户对象
            ip_address: 客户端IP地址
            user_agent: 用户代理字符串

        Returns:
            包含access_token和refresh_token的字典
        """
        # 生成访问token
        access_token_payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'exp': datetime.utcnow() + timedelta(minutes=self.token_expiry_minutes),
            'iat': datetime.utcnow(),
            'type': 'access'
        }

        access_token = jwt.encode(
            access_token_payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        # 生成刷新token
        refresh_token_payload = {
            'user_id': user.user_id,
            'exp': datetime.utcnow() + timedelta(days=self.refresh_token_expiry_days),
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }

        refresh_token = jwt.encode(
            refresh_token_payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        # 创建会话
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            token=access_token,
            expires_at=datetime.utcnow() + timedelta(minutes=self.token_expiry_minutes),
            created_at=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        self.refresh_tokens[refresh_token] = {
            'user_id': user.user_id,
            'session_id': session_id
        }

        logger.info(f"为用户 {user.username} 生成token")

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': self.token_expiry_minutes * 60
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        验证访问token

        Args:
            token: JWT token

        Returns:
            token载荷

        Raises:
            AuthenticationException: token无效时
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            if payload.get('type') != 'access':
                raise AuthenticationException("无效的token类型")

            # 检查用户是否存在且激活
            user_id = payload.get('user_id')
            if user_id not in self.users or not self.users[user_id].is_active:
                raise AuthenticationException("用户不存在或已被禁用")

            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token已过期")
        except jwt.InvalidTokenError as e:
            raise AuthenticationException(f"无效的token: {e}")

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        使用刷新token获取新的访问token

        Args:
            refresh_token: 刷新token

        Returns:
            新的token信息

        Raises:
            AuthenticationException: 刷新token无效时
        """
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            if payload.get('type') != 'refresh':
                raise AuthenticationException("无效的token类型")

            user_id = payload.get('user_id')
            if user_id not in self.users:
                raise AuthenticationException("用户不存在")

            # 检查刷新token是否在存储中
            if refresh_token not in self.refresh_tokens:
                raise AuthenticationException("刷新token已失效")

            user = self.users[user_id]
            if not user.is_active:
                raise AuthenticationException("用户已被禁用")

            # 生成新的访问token
            return self.generate_token(user)

        except jwt.ExpiredSignatureError:
            raise AuthenticationException("刷新token已过期")
        except jwt.InvalidTokenError as e:
            raise AuthenticationException(f"无效的刷新token: {e}")

    def revoke_token(self, token: str) -> None:
        """
        撤销token

        Args:
            token: 要撤销的token（访问或刷新token）
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # 忽略过期时间
            )

            if payload.get('type') == 'refresh':
                # 撤销刷新token
                if token in self.refresh_tokens:
                    session_id = self.refresh_tokens[token]['session_id']
                    del self.refresh_tokens[token]

                    # 同时删除对应会话
                    if session_id in self.sessions:
                        del self.sessions[session_id]

                    logger.info(f"已撤销刷新token")

            elif payload.get('type') == 'access':
                # 撤销访问token（通过删除会话）
                user_id = payload.get('user_id')
                sessions_to_remove = [
                    sid for sid, session in self.sessions.items()
                    if session.user_id == user_id and session.token == token
                ]

                for session_id in sessions_to_remove:
                    del self.sessions[session_id]

                logger.info(f"已撤销访问token")

        except jwt.InvalidTokenError:
            logger.warning("尝试撤销无效token")

    def get_user_info(self, user_id: str) -> Optional[User]:
        """
        获取用户信息

        Args:
            user_id: 用户ID

        Returns:
            用户对象，不存在返回None
        """
        return self.users.get(user_id)

    def check_permission(self, user_id: str, required_roles: List[str]) -> bool:
        """
        检查用户权限

        Args:
            user_id: 用户ID
            required_roles: 需要的角色列表

        Returns:
            是否有权限
        """
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False

        # 检查是否有任一需要的角色
        return any(role in user.roles for role in required_roles)

    def list_active_sessions(self, user_id: str) -> List[Session]:
        """
        列出用户的活跃会话

        Args:
            user_id: 用户ID

        Returns:
            活跃会话列表
        """
        now = datetime.utcnow()
        active_sessions = []

        for session in self.sessions.values():
            if session.user_id == user_id and session.expires_at > now:
                active_sessions.append(session)

        return active_sessions

    def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            清理的会话数量
        """
        now = datetime.utcnow()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if session.expires_at <= now:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        # 同时清理过期的刷新token
        expired_refresh_tokens = []
        for refresh_token in self.refresh_tokens.keys():
            try:
                payload = jwt.decode(
                    refresh_token,
                    self.secret_key,
                    algorithms=[self.algorithm]
                )
            except jwt.ExpiredSignatureError:
                expired_refresh_tokens.append(refresh_token)

        for refresh_token in expired_refresh_tokens:
            del self.refresh_tokens[refresh_token]

        total_cleaned = len(expired_sessions) + len(expired_refresh_tokens)
        if total_cleaned > 0:
            logger.info(f"清理过期会话和token: {total_cleaned} 个")

        return total_cleaned

    def get_stats(self) -> Dict[str, Any]:
        """
        获取认证统计信息

        Returns:
            统计信息字典
        """
        now = datetime.utcnow()
        active_sessions = sum(
            1 for session in self.sessions.values()
            if session.expires_at > now
        )

        return {
            'total_users': len(self.users),
            'active_users': sum(1 for user in self.users.values() if user.is_active),
            'total_sessions': len(self.sessions),
            'active_sessions': active_sessions,
            'refresh_tokens': len(self.refresh_tokens)
        }