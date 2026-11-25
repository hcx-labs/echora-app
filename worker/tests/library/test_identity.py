import uuid

from worker.library.ingest import CONTENT_NAMESPACE


def test_content_identity_is_stable() -> None:
    digest = "a" * 64
    assert uuid.uuid5(CONTENT_NAMESPACE, digest) == uuid.uuid5(CONTENT_NAMESPACE, digest)
    assert uuid.uuid5(CONTENT_NAMESPACE, digest) != uuid.uuid5(CONTENT_NAMESPACE, "b" * 64)
