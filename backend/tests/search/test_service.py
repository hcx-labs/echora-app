import asyncio
import threading

import pytest

from backend.search.service import Registry, SearchService, run_search, shutdown


class Counting(SearchService[int, int]):
    name = "counting"
    persistent = True
    max_instances = 2
    keys: dict = {}
    opened: list = []
    closed: list = []

    def index_key(self):
        return self.keys.get(self.scope, 0)

    def open(self):
        self.opened.append(self.scope)
        self.base = self.keys.get(self.scope, 0) * 100

    def search(self, query):
        return self.base + query

    def close(self):
        self.closed.append(self.scope)


class OneShot(Counting):
    name = "one-shot"
    persistent = False


@pytest.fixture(autouse=True)
def fresh():
    Counting.keys, Counting.opened, Counting.closed = {}, [], []
    yield
    shutdown()


def test_persistent_service_is_reused_until_its_key_changes():
    registry = Registry()
    first = registry.acquire(Counting, "a")
    assert registry.acquire(Counting, "a") is first
    Counting.keys["a"] = 1
    second = registry.acquire(Counting, "a")
    assert second is not first
    assert second.search(5) == 105
    assert Counting.opened == ["a", "a"]
    assert Counting.closed == ["a"]


def test_lru_bound_evicts_the_least_recent_scope():
    registry = Registry()
    registry.acquire(Counting, "a")
    registry.acquire(Counting, "b")
    registry.acquire(Counting, "a")
    registry.acquire(Counting, "c")
    assert Counting.closed == ["b"]


def test_concurrent_first_requests_open_once():
    registry = Registry()
    barrier = threading.Barrier(4)

    def acquire():
        barrier.wait()
        registry.acquire(Counting, "a")

    threads = [threading.Thread(target=acquire) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert Counting.opened == ["a"]


def test_non_persistent_service_opens_and_closes_per_request():
    assert asyncio.run(run_search(OneShot, "a", 3)) == 3
    assert asyncio.run(run_search(OneShot, "a", 4)) == 4
    assert OneShot.opened == ["a", "a"]
    assert OneShot.closed == ["a", "a"]
