"""Search services: a request in, a response out, with indexes kept between requests.

A ``SearchService`` is built for one scope (usually a user), loads what it needs in ``open()``
and answers ``search(query)``. The ``registry`` keeps a ``persistent`` service open and reuses it
until its ``index_key()`` changes or the per-class LRU bound evicts it; a non-persistent service
is opened per request. ``run_search`` runs a search on the bounded search pool, so heavy queries
never occupy the event loop or more than ``SearchConfig.max_concurrent`` threads.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

from backend.config import SearchConfig


class SearchService[QueryT, ResultT]:
    """One search, for one scope."""

    name: ClassVar[str]
    persistent: ClassVar[bool] = False
    """Keep the opened instance (and its index) between requests."""
    max_instances: ClassVar[int] = 8
    """LRU bound on cached instances of this class (one per scope)."""

    def __init__(self, scope: Hashable) -> None:
        self.scope = scope

    def index_key(self) -> Hashable:
        """Changes whenever the opened index is out of date. Must be cheap (one small query).

        Only persistent services are asked. The default never changes.
        """
        return None

    def open(self) -> None:
        """Load indexes and models."""

    def search(self, query: QueryT) -> ResultT:
        """Answer one query. Must be safe to call from several threads at once."""
        raise NotImplementedError

    def close(self) -> None:
        """Release what ``open()`` loaded."""


class _Entry:
    def __init__(self, service: SearchService, key: Hashable) -> None:
        self.service, self.key = service, key


class Registry:
    """Open search services, keyed by class and scope."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scope_locks: dict[tuple[type, Hashable], threading.Lock] = {}
        self._entries: dict[type, OrderedDict[Hashable, _Entry]] = {}

    def acquire[ServiceT: SearchService](self, cls: type[ServiceT], scope: Hashable) -> ServiceT:
        """An open service: the cached one while its index key is current, otherwise a new one.

        A non-persistent service is returned open; the caller closes it (``run_search`` does).
        """
        if not cls.persistent:
            service = cls(scope)
            service.open()
            return service
        with self._lock:
            scope_lock = self._scope_locks.setdefault((cls, scope), threading.Lock())
        # One opener per scope: concurrent first requests share one load.
        with scope_lock:
            probe = cls(scope)
            key = probe.index_key()
            with self._lock:
                entries = self._entries.setdefault(cls, OrderedDict())
                entry = entries.get(scope)
                if entry is not None and entry.key == key:
                    entries.move_to_end(scope)
                    return entry.service  # type: ignore[return-value]
            probe.open()
            stale: list[SearchService] = []
            with self._lock:
                old = entries.pop(scope, None)
                if old is not None:
                    stale.append(old.service)
                entries[scope] = _Entry(probe, key)
                while len(entries) > cls.max_instances:
                    _, evicted = entries.popitem(last=False)
                    stale.append(evicted.service)
            # Searches still running on a replaced instance keep their own references.
            for service in stale:
                service.close()
            return probe

    def clear(self) -> None:
        """Close every cached service (tests, shutdown)."""
        with self._lock:
            services = [e.service for entries in self._entries.values() for e in entries.values()]
            self._entries.clear()
            self._scope_locks.clear()
        for service in services:
            service.close()


registry = Registry()

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _search_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=SearchConfig.load().max_concurrent, thread_name_prefix="search"
            )
        return _pool


def search_now[QueryT, ResultT](
    cls: type[SearchService[QueryT, ResultT]], scope: Hashable, query: QueryT
) -> ResultT:
    """Acquire the service and search, in the calling thread."""
    service = registry.acquire(cls, scope)
    try:
        return service.search(query)
    finally:
        if not cls.persistent:
            service.close()


async def run_search[QueryT, ResultT](
    cls: type[SearchService[QueryT, ResultT]], scope: Hashable, query: QueryT
) -> ResultT:
    """``search_now`` on the bounded search pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_search_pool(), search_now, cls, scope, query)


def shutdown() -> None:
    """Close cached services and stop the pool."""
    global _pool
    registry.clear()
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
