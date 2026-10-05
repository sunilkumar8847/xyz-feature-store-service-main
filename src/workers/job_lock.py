"""
feature-store-service/src/workers/job_lock.py

Mutual exclusion for materialization jobs.

Two jobs for the same tenant must never run at once (the API can be called twice, and
every API worker process runs its own scheduler). The lock also answers "is anyone
still running this job?" after a restart: a job recorded as RUNNING whose lock is free
was interrupted.

Lock scopes
  tenant job      shared lock on ALL  + exclusive lock on the tenant
  all-tenant job  exclusive lock on ALL
so a tenant job excludes another job for that tenant and any all-tenant job, while jobs
for different tenants run side by side.

PostgresJobLock uses session-level advisory locks: they live on one database connection
and PostgreSQL releases them when that connection ends, so a crashed process can never
leave a lock behind. InProcessJobLock has the same semantics inside one process (tests,
or a deployment with no Postgres lock available).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

ALL_TENANTS = "*"


def _lock_id(scope: str) -> int:
    """Stable signed 64-bit advisory-lock id for a scope name."""
    digest = hashlib.sha256(f"feature-store:materialization:{scope}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class JobLock:
    """Interface: acquire(tenant_id) -> handle or None; release(handle)."""

    async def acquire(self, tenant_id: Optional[str]):
        raise NotImplementedError

    async def release(self, handle) -> None:
        raise NotImplementedError


class PostgresJobLock(JobLock):
    def __init__(self, engine):
        self._engine = engine

    async def acquire(self, tenant_id: Optional[str]):
        conn = await self._engine.connect()
        try:
            if tenant_id is None:
                ok = await conn.scalar(
                    text("SELECT pg_try_advisory_lock(:k)"), {"k": _lock_id(ALL_TENANTS)})
            else:
                ok = await conn.scalar(
                    text("SELECT pg_try_advisory_lock_shared(:k)"), {"k": _lock_id(ALL_TENANTS)})
                if ok:
                    ok = await conn.scalar(
                        text("SELECT pg_try_advisory_lock(:k)"), {"k": _lock_id(tenant_id)})
            if not ok:
                await conn.close()   # closing the session drops any lock it did get
                return None
            return conn
        except Exception:
            await conn.close()
            raise

    async def release(self, handle) -> None:
        if handle is not None:
            try:
                await handle.execute(text("SELECT pg_advisory_unlock_all()"))
            finally:
                await handle.close()


class InProcessJobLock(JobLock):
    """Same exclusion rules, one process only. State is per instance."""

    def __init__(self):
        self._exclusive: Dict[str, bool] = {}
        self._shared_all = 0

    async def acquire(self, tenant_id: Optional[str]):
        if tenant_id is None:
            if self._exclusive.get(ALL_TENANTS) or self._shared_all or any(self._exclusive.values()):
                return None
            self._exclusive[ALL_TENANTS] = True
            return (ALL_TENANTS,)
        if self._exclusive.get(ALL_TENANTS) or self._exclusive.get(tenant_id):
            return None
        self._exclusive[tenant_id] = True
        self._shared_all += 1
        return (tenant_id,)

    async def release(self, handle: Optional[Tuple[str]]) -> None:
        if not handle:
            return
        scope = handle[0]
        self._exclusive.pop(scope, None)
        if scope != ALL_TENANTS:
            self._shared_all = max(self._shared_all - 1, 0)
