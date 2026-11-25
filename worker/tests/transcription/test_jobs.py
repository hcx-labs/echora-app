from types import SimpleNamespace
from unittest.mock import Mock

import worker.operations as analysis_jobs
from core.config import reset_config_cache
from core.testing import stub_module


def setup_main(monkeypatch, library_helpers):
    main = SimpleNamespace(_load_connection=Mock(return_value=("url", "user", "password")))
    library_helpers(main)
    return SimpleNamespace(check=Mock(), report=Mock(), token="lease")


def test_default_batch_is_128(monkeypatch, library_helpers):
    ctx = setup_main(monkeypatch, library_helpers)
    monkeypatch.delenv("ECHORA_BATCH_SIZE", raising=False)
    reset_config_cache()
    jobs = SimpleNamespace(expand=Mock())
    stub_module(monkeypatch, "core.jobs.queue", jobs)
    analysis_jobs.execute(
        {
            "id": "p",
            "kind": "import",
            "user_id": "u",
            "payload": {"connection_id": "c", "track_ids": [str(i) for i in range(129)]},
        },
        ctx,
    )
    batches = jobs.expand.call_args.args[2]
    assert [len(b["track_ids"]) for b in batches] == [128, 1]


def test_lyrics_stage_precedes_karaoke_even_for_karaoke_backfill(monkeypatch, library_helpers):
    ctx = setup_main(monkeypatch, library_helpers)
    order = []

    def lyrics(*args, **kwargs):
        order.append("lyrics")
        return {}

    from worker.processing import Processor

    class Karaoke(Processor):
        name = summary_key = "karaoke"

        def plan(self, context):
            order.append("karaoke")
            return []

        def process(self, context, items):
            yield from ()

    stub_module(monkeypatch, "worker.lyrics.pipeline", SimpleNamespace(lyrics_stage=lyrics))
    stub_module(monkeypatch, "worker.kara.pipeline", SimpleNamespace(KaraokeProcessor=Karaoke))
    analysis_jobs.execute(
        {
            "kind": "analysis_batch",
            "user_id": "u",
            "payload": {
                "operation": "karaoke_backfill",
                "connection_id": "c",
                "track_ids": ["song"],
            },
        },
        ctx,
    )
    assert order == ["lyrics", "karaoke"]
