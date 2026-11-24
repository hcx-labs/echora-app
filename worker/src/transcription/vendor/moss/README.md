# MOSS processor

`processor.py` is copied without changes from OpenMOSS/MOSS-Transcribe-Diarize, commit `61bc29cd4120be7b5d3b761b64cd5dff57263642`, path `moss_transcribe_diarize/processing_moss_transcribe_diarize.py`.

Upstream: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
License: Apache-2.0, included in LICENSE.

The explicit processor is required because the fine-tuned checkpoint's AutoProcessor configuration resolves to a tokenizer alone. Model architecture code comes from the configured, reviewed HF snapshot, with its revision pinned.
