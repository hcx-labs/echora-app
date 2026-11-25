from collections.abc import Iterator, Sequence

import pytest

from core.config import EchoraConfig, setting
from core.errors import IntegrationError, JobCancelled, ModelUnavailableError, ProcessingError
from worker.processing import Item, ItemResult, ProcessingContext, Processor, run_stage


class _Config(EchoraConfig, env_prefix="ECHORA_TEST_RUNNER_"):
    scale: int = setting(2)


class _Song(Item):
    value: int


class _Doubler(Processor[_Config, _Song, int]):
    """Doubles values; odd values fail; zero is reused; a value of 99 is a bug."""

    name = "doubling"
    summary_key = "doubling"
    counters = ("doubled",)
    activity = "Doubling"

    def __init__(self, config: _Config, items: Sequence[_Song], *, crash_after: int | None = None):
        super().__init__(config)
        self.items = list(items)
        self.crash_after = crash_after
        self.events: list[str] = []
        self.stored: dict[str, int] = {}

    def plan(self, context: ProcessingContext) -> Sequence[_Song]:
        self.events.append("plan")
        return self.items

    def open(self, context: ProcessingContext, items: Sequence[_Song]) -> None:
        self.events.append(f"open:{len(items)}")

    def process(
        self, context: ProcessingContext, items: Sequence[_Song]
    ) -> Iterator[ItemResult[_Song, int]]:
        for index, item in enumerate(items):
            if self.crash_after is not None and index == self.crash_after:
                raise ModelUnavailableError("model gone")
            if item.value == 99:
                raise RuntimeError("bug")
            if item.value == 77:
                raise IntegrationError("service down")
            if item.value == 0:
                yield ItemResult(item=item, outcome="reused")
            elif item.value % 2:
                yield ItemResult.failed(item, ProcessingError("odd", stage="double"))
            else:
                yield ItemResult(
                    item=item, outcome="doubled", output=item.value * self.config.scale
                )

    def persist(self, context: ProcessingContext, item: _Song, output: int) -> None:
        if output == 8:
            raise ValueError("storage failed")
        self.stored[item.key] = output

    def close(self) -> None:
        self.events.append("close")


def _songs(*values: int) -> list[_Song]:
    return [_Song(key=f"t{i}", label=f"Song {i}", value=v) for i, v in enumerate(values)]


def test_nothing_planned_loads_nothing_and_reports_current():
    updates = []
    processor = _Doubler(_Config(), [])
    summary = run_stage(processor, ProcessingContext(publish=updates.append))
    assert processor.events == ["plan"]
    assert summary.as_dict() == {"total": 0, "doubled": 0, "failed": 0}
    assert updates == [
        {
            "phase": "planning",
            "message": "Already current",
            "completed": 0,
            "total": 0,
            "unit": "tracks",
        }
    ]


def test_each_outcome_is_counted_and_outputs_are_stored():
    processor = _Doubler(_Config(), _songs(2, 3, 0, 6))
    summary = run_stage(processor, ProcessingContext())
    assert processor.events == ["plan", "open:4", "close"]
    assert processor.stored == {"t0": 4, "t3": 12}
    assert summary.as_dict() == {"total": 4, "doubled": 2, "reused": 1, "failed": 1}


def test_a_storage_failure_fails_only_that_item():
    processor = _Doubler(_Config(), _songs(4, 2))
    summary = run_stage(processor, ProcessingContext())
    assert processor.stored == {"t1": 4}
    assert summary.as_dict() == {"total": 2, "doubled": 1, "failed": 1}


def test_progress_follows_each_item_with_the_running_summary():
    updates = []
    run_stage(_Doubler(_Config(), _songs(2, 3)), ProcessingContext(publish=updates.append))
    assert [(u["message"], u["completed"], u["total"]) for u in updates] == [
        ("Doubling Song 0", 1, 2),
        ("Doubling Song 1", 2, 2),
    ]
    assert updates[-1]["summary"] == {"total": 2, "doubled": 1, "failed": 1}


def test_a_domain_error_escaping_process_fails_the_remaining_items():
    processor = _Doubler(_Config(), _songs(2, 4, 6), crash_after=1)
    summary = run_stage(processor, ProcessingContext())
    assert processor.events[-1] == "close"
    assert summary.as_dict() == {"total": 3, "doubled": 1, "failed": 2}


def test_an_integration_failure_fails_the_job_after_closing():
    processor = _Doubler(_Config(), _songs(2, 77, 6))
    with pytest.raises(IntegrationError):
        run_stage(processor, ProcessingContext())
    assert processor.events[-1] == "close"
    assert processor.stored == {"t0": 4}


def test_a_storage_failure_fails_the_job_when_the_stage_says_so():
    class _Strict(_Doubler):
        storage_failure_fails_job = True

    processor = _Strict(_Config(), _songs(2, 4, 6))
    with pytest.raises(ValueError, match="storage failed"):
        run_stage(processor, ProcessingContext())
    assert processor.events[-1] == "close"


def test_a_bug_propagates_after_closing():
    processor = _Doubler(_Config(), _songs(2, 99))
    with pytest.raises(RuntimeError):
        run_stage(processor, ProcessingContext())
    assert processor.events[-1] == "close"


def test_cancellation_stops_between_items_and_closes():
    calls = []

    def check() -> None:
        calls.append(1)
        if len(calls) > 2:
            raise JobCancelled

    processor = _Doubler(_Config(), _songs(2, 4, 6))
    with pytest.raises(JobCancelled):
        run_stage(processor, ProcessingContext(check_cancelled=check))
    assert processor.stored == {"t0": 4}
    assert processor.events[-1] == "close"


def test_custom_summary_keys():
    class _Renderer(_Doubler):
        total_key = "tracks"
        failure_key = "errors"
        counters = ("rendered",)

    summary = run_stage(_Renderer(_Config(), _songs(3)), ProcessingContext())
    summary.extra["unavailable"] = "no GPU"
    assert summary.as_dict() == {"tracks": 1, "rendered": 0, "errors": 1, "unavailable": "no GPU"}


def test_missing_resource_is_named():
    with pytest.raises(KeyError, match="navidrome"):
        ProcessingContext().resource("navidrome")


def test_a_stage_without_a_total():
    class _Untotalled(_Doubler):
        total_key = None

    summary = run_stage(_Untotalled(_Config(), _songs(2)), ProcessingContext())
    assert summary.as_dict() == {"doubled": 1, "failed": 0}


def test_stopping_early_leaves_the_rest_uncounted():
    class _Switched(_Doubler):
        def process(self, context, items):
            yield ItemResult(item=items[0], outcome="doubled", output=items[0].value * 2)
            # The feature was switched off: the rest is not attempted, and not a failure.

    processor = _Switched(_Config(), _songs(2, 4, 6))
    summary = run_stage(processor, ProcessingContext())
    assert summary.as_dict() == {"total": 3, "doubled": 1, "failed": 0}
    assert processor.events[-1] == "close"


def test_persist_can_add_counters():
    class Matching(_Doubler):
        counters = ("doubled", "large")

        def persist(self, context, item, output):
            return ("large",) if output > 4 else None

    processor = Matching(_Config(), [_Song(key="a", value=2), _Song(key="b", value=4)])
    summary = run_stage(processor, ProcessingContext()).as_dict()
    # Every stored song counts as doubled; "large" is counted on top of it.
    assert summary == {"total": 2, "doubled": 2, "large": 1, "failed": 0}


def test_persist_decides_what_an_item_without_outcome_counts_as():
    class Deciding(_Doubler):
        def process(self, context, items):
            for item in items:
                yield ItemResult(item=item, outcome=None, output=item.value)

        def persist(self, context, item, output):
            return ("kept",) if output > 2 else ()

    processor = Deciding(_Config(), [_Song(key="a", value=2), _Song(key="b", value=4)])
    summary = run_stage(processor, ProcessingContext()).as_dict()
    assert summary == {"total": 2, "doubled": 0, "kept": 1, "failed": 0}
