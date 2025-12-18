"""
Database Configuration

Shared SQLAlchemy Base and database session management.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

# Shared declarative base
Base = declarative_base()

# Global engine and session factory (to be initialized)
engine = None
SessionLocal = None


def init_database(database_url: str = "sqlite:///./color_meter.db", echo: bool = False) -> None:
    """
    Initialize database engine and session factory.

    Args:
        database_url: SQLAlchemy database URL
        echo: Whether to echo SQL statements (for debugging)
    """
    global engine, SessionLocal

    # Create engine
    if database_url.startswith("sqlite"):
        # SQLite-specific configuration
        engine = create_engine(database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool, echo=echo)
    else:
        # PostgreSQL or other databases
        engine = create_engine(database_url, echo=echo)

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all tables in the database"""
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all tables in the database"""
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    Base.metadata.drop_all(bind=engine)


def get_session() -> Generator[Session, None, None]:
    """
    Get database session (dependency injection for FastAPI).

    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_session)):
            return db.query(Item).all()
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Session:
    """
    Get database session (for direct use).

    Usage:
        db = get_db()
        try:
            db.query(Item).all()
            db.commit()
        finally:
            db.close()
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    return SessionLocal()
