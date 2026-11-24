from core.errors import EchoraError, InputUnavailableError, JobCancelled, ProcessingError, RemoteComputeError


def test_processing_errors_carry_their_stage_and_item():
    error = InputUnavailableError("no audio", stage="download", item="t1")
    assert isinstance(error, ProcessingError)
    assert (error.stage, error.item, str(error)) == ("download", "t1", "no audio")


def test_cancellation_is_not_an_exception():
    assert not issubclass(JobCancelled, Exception)


def test_remote_compute_is_an_integration_error():
    assert issubclass(RemoteComputeError, EchoraError)
