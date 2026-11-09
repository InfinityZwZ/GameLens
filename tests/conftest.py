"""Legacy gold fixtures describe a UTC computer; do not depend on CI's timezone.

The system_local_time tests separately exercise positive/negative offsets,
calendar/DST boundaries, and the unmocked operating-system conversion.
"""
from datetime import timezone
import pytest
from gamelens.core import local_time


@pytest.fixture(autouse=True)
def stable_gold_fixture_clock(request, monkeypatch):
    if request.node.get_closest_marker('system_local_time'):
        return
    monkeypatch.setattr(local_time, '_system_localize', lambda moment:
        moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc))
