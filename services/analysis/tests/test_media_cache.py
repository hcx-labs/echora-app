from core.config import reset_config_cache
from types import SimpleNamespace

from echora_analysis.media_cache import StreamCacheWriter


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    def set(self, key, value, *, ex):
        self.values[key] = value
        self.expirations[key] = ex

    def exists(self, key):
        return key in self.values

    def append(self, key, value):
        self.values[key] += value

    def expire(self, key, seconds):
        if key not in self.values:
            return False
        self.expirations[key] = seconds
        return True

    def rename(self, source, destination):
        self.values[destination] = self.values.pop(source)
        self.expirations[destination] = self.expirations.pop(source)

    def delete(self, key):
        self.values.pop(key, None)
        self.expirations.pop(key, None)


def test_stream_writer_renews_staging_ttl_and_publishes_with_cache_ttl(monkeypatch):
    import echora_analysis.media_cache as media_cache

    monkeypatch.setenv("ECHORA_MEDIA_CACHE_TTL_SECONDS", "12")
    reset_config_cache()
    client = FakeRedis()
    writer = StreamCacheWriter(client, "cached", "audio/mpeg")

    writer.append(b"audio")
    assert client.values[writer.staging_key] == b"audio/mpeg\0audio"
    assert client.expirations[writer.staging_key] == 3600

    writer.commit()
    assert client.values["cached"] == b"audio/mpeg\0audio"
    assert client.expirations["cached"] == 12
    assert writer.enabled is False


def test_stream_writer_does_not_recreate_expired_staging_key(monkeypatch):
    import echora_analysis.media_cache as media_cache

    monkeypatch.setenv("ECHORA_MEDIA_CACHE_TTL_SECONDS", "12")
    reset_config_cache()
    client = FakeRedis()
    writer = StreamCacheWriter(client, "cached", "audio/mpeg")
    client.delete(writer.staging_key)

    writer.append(b"audio")
    assert writer.enabled is False
    assert writer.staging_key not in client.values
