"""Project-wide local timezone.

All bot/dashboard timestamps — both displayed and persisted — use this
timezone instead of UTC. It is a *fixed* offset (no DST), so the wall-clock
representation of any instant is deterministic. To shift the whole project to
a different offset, change LOCAL_TZ here (and re-run scripts/migrate_tz.py to
relabel already-stored timestamps).
"""

from datetime import datetime, timezone, timedelta

# Fixed UTC+3 offset (e.g. Moscow / Gulf / East Africa — all DST-free).
LOCAL_TZ = timezone(timedelta(hours=3))

# Human-readable suffix used in formatted timestamp strings.
TZ_LABEL = "UTC+3"


def now() -> datetime:
    """Timezone-aware 'now' in the project local timezone (UTC+3)."""
    return datetime.now(LOCAL_TZ)
