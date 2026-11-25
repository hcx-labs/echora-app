"""``/search``: hum, recording and concept search."""

from backend.api.endpoint import Category
from backend.api.v1.search.calibration_sample_delete import CalibrationSampleDelete
from backend.api.v1.search.calibration_sample_save import CalibrationSampleSave
from backend.api.v1.search.calibration_samples import CalibrationSamples
from backend.api.v1.search.concepts import ConceptSearch
from backend.api.v1.search.hum import HumSearchEndpoint
from backend.api.v1.search.hum_index_build import HumIndexBuild
from backend.api.v1.search.hum_index_status import HumIndexStatus
from backend.api.v1.search.recording import RecordingSearch
from backend.api.v1.search.recording_cancel import RecordingCancel
from backend.api.v1.search.recording_result import RecordingResult
from backend.api.v1.search.recording_status import RecordingStatus

# Static recording paths come before /recording/{job_id}.
CATEGORY = Category(
    prefix="/search",
    endpoints=[
        HumSearchEndpoint,
        HumIndexStatus,
        HumIndexBuild,
        RecordingStatus,
        RecordingSearch,
        CalibrationSamples,
        CalibrationSampleSave,
        CalibrationSampleDelete,
        RecordingResult,
        RecordingCancel,
        ConceptSearch,
    ],
)
