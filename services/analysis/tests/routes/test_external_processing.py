import pytest

from echora_analysis.routes.external_processing import SettingsUpdate


def test_settings_reject_malformed_tokens_and_gpus():
    with pytest.raises(ValueError):
        SettingsUpdate(token_id="as-not-an-id")
    with pytest.raises(ValueError):
        SettingsUpdate(gpu="RTX9000")
