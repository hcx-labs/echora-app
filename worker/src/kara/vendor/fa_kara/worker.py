"""Persistent JSON-lines worker for FA-Kara.

The process keeps acoustic models resident between jobs. Each request supplies
ordinary ``main.py`` arguments and receives one JSON response on stdout.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import traceback

from main import main


def run() -> None:
    protocol_out = sys.stdout
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            argv = request.get("argv")
            if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                raise ValueError("worker request must contain a string argv list")
            logs = io.StringIO()
            with redirect_stdout(logs), redirect_stderr(logs):
                main(argv)
            response = {"ok": True, "logs": logs.getvalue()[-4000:]}
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc()[-8000:],
            }
        protocol_out.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    run()
