"""The aligner's worker process: ``python -m worker.kara.aligner.worker``.

It keeps the decoder's models resident between songs. Each stdin line is one ``AlignRequest`` as
JSON; each reply is one JSON line on stdout: ``{"ok": true, "document": ..., "logs": ...}`` or
``{"ok": false, "error": ..., "traceback": ...}``. The decoder's own prints go to ``logs``.
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

from worker.kara.aligner.align import align
from worker.kara.aligner.request import AlignRequest


def run() -> None:
    protocol_out = sys.stdout
    for raw in sys.stdin:
        try:
            request = AlignRequest.model_validate_json(raw)
            logs = io.StringIO()
            with redirect_stdout(logs), redirect_stderr(logs):
                document = align(request)
            response = {"ok": True, "document": document, "logs": logs.getvalue()[-4000:]}
        except Exception as error:  # noqa: BLE001 - reported to the pipeline, which raises
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc()[-8000:],
            }
        protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    run()
