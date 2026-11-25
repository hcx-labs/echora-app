"""``PUT /settings/models/hum``."""

from __future__ import annotations

import psycopg
from fastapi import Depends
from psycopg.rows import dict_row

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._deps import admin_required
from backend.api.v1.settings._models import ModelSwitch, SwitchWithBacklog
from core.db.config import DatabaseConfig
from core.revisions import MELODY_CONTOUR_REVISION


class HumSwitch(Endpoint):
    """Turn melody contour extraction (hum search) on or off for syncs here or on Modal."""

    method = "PUT"
    path = "/models/hum"
    request = ModelSwitch
    response = SwitchWithBacklog

    def handle(
        self, request: ModelSwitch, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        column = "hum_modal_enabled" if request.location == "modal" else "hum_processing_enabled"
        with (
            psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                f"""INSERT INTO analysis_settings
                     (singleton, karaoke_bound_to_synced_lines, {column}, updated_at)
                   VALUES (true,false,%s,now()) ON CONFLICT (singleton) DO UPDATE
                   SET {column}=EXCLUDED.{column},
                       karaoke_bound_to_synced_lines=false, updated_at=now()""",
                (request.enabled,),
            )
            pending = 0
            if request.enabled:
                cursor.execute(
                    """SELECT count(DISTINCT ts.track_id) AS pending
                       FROM track_sources ts
                       WHERE ts.source_type='subsonic'
                         AND NOT EXISTS (
                           SELECT 1 FROM melody_contours mc
                           JOIN analysis_runs ar ON ar.id=mc.run_id
                           WHERE mc.track_id=ts.track_id
                             AND ar.model_name='melody_contour' AND ar.model_revision=%s
                         )""",
                    (MELODY_CONTOUR_REVISION,),
                )
                pending = int(cursor.fetchone()["pending"])
        return {"enabled": request.enabled, "location": request.location, "pending": pending}
