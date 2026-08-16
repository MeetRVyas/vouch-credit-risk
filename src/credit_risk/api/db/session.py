from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from credit_risk.api.config import Settings
from credit_risk.api.db.models import Base


def make_engine(settings: Settings):
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine) -> None:
    """Create tables if they don't exist. Fine for this project's scope
    (spec explicitly puts migrations/registry out of scope); swap for
    Alembic migrations if this ever needs real schema evolution."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session
