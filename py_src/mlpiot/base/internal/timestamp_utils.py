import calendar
from datetime import datetime


_NANOS_PER_MICROSECOND = 1000


def set_now(timestamp):
    dt = datetime.utcnow()
    timestamp.seconds = calendar.timegm(dt.utctimetuple())
    timestamp.nanos = dt.microsecond * _NANOS_PER_MICROSECOND
