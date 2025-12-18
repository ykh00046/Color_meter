"""
User Models

SQLAlchemy models for users and audit logs.
"""

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from .database import Base


class UserRole(enum.Enum):
    """User role enumeration"""

    ADMIN = "admin"  # Full access (approve STD, manage users, etc.)
    ENGINEER = "engineer"  # Register STD, configure criteria
    INSPECTOR = "inspector"  # Run tests, view results
    VIEWER = "viewer"  # Read-only access


class User(Base):
    """
    User

    System users with role-based access control.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Authentication
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))  # bcrypt hash

    # Profile
    full_name = Column(String(200))
    role = Column(Enum(UserRole), nullable=False, default=UserRole.INSPECTOR, index=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    is_locked = Column(Boolean, nullable=False, default=False)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_login_at = Column(DateTime)
    password_changed_at = Column(DateTime)

    # Session info
    failed_login_attempts = Column(Integer, default=0)
    last_failed_login_at = Column(DateTime)

    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user", foreign_keys="AuditLog.user_id")

    __table_args__ = (
        Index("idx_user_username", "username"),
        Index("idx_user_role_active", "role", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, role={self.role.value if self.role else None})>"

    @hybrid_property
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN

    @hybrid_property
    def can_approve_std(self) -> bool:
        """Check if user can approve STD models"""
        return self.role in (UserRole.ADMIN, UserRole.ENGINEER)

    @hybrid_property
    def can_manage_users(self) -> bool:
        """Check if user can manage other users"""
        return self.role == UserRole.ADMIN

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value if self.role else None,
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }

        if include_sensitive:
            data.update(
                {
                    "failed_login_attempts": self.failed_login_attempts,
                    "last_failed_login_at": (
                        self.last_failed_login_at.isoformat() if self.last_failed_login_at else None
                    ),
                    "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
                }
            )

        return data


class AuditLog(Base):
    """
    Audit Log

    System activity log for compliance and debugging.
    """

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    # Action details
    action = Column(String(50), nullable=False, index=True)
    # Examples: 'std_register', 'std_approve', 'std_update', 'test_compare',
    #           'user_create', 'user_login', 'user_logout', 'criteria_update'

    target_type = Column(String(50), index=True)  # 'std_model', 'test_sample', 'user', etc.
    target_id = Column(Integer, index=True)  # ID of target object

    # Request details
    ip_address = Column(String(45))  # IPv4 or IPv6
    user_agent = Column(String(500))

    # Changes (JSON)
    changes = Column(JSON)
    # Example: {
    #   "before": {"sku_code": "SKU001", "is_active": true},
    #   "after": {"sku_code": "SKU001", "is_active": false},
    #   "changed_fields": ["is_active"]
    # }

    # Result
    success = Column(Boolean, nullable=False, default=True, index=True)
    error_message = Column(String(1000))

    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs", foreign_keys=[user_id])

    __table_args__ = (
        Index("idx_audit_action_created", "action", "created_at"),
        Index("idx_audit_user_created", "user_id", "created_at"),
        Index("idx_audit_target", "target_type", "target_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, user={self.user_id}, action={self.action}, "
            f"target={self.target_type}:{self.target_id})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "changes": self.changes,
            "success": self.success,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @staticmethod
    def create_log(
        user_id: Optional[int],
        action: str,
        target_type: Optional[str] = None,
        target_id: Optional[int] = None,
        changes: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> "AuditLog":
        """Factory method to create audit log"""
        return AuditLog(
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            changes=changes,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
        )
