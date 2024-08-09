from datetime import datetime, timedelta
from typing import Iterable, NewType, Tuple, Optional

import pytz

LocalizedDateTime = NewType("LocalizedDateTime", datetime)
NaiveDateTime = NewType("NaiveDateTime", datetime)


MIN_DATE = LocalizedDateTime(pytz.timezone("UTC").localize(datetime(1900, 1, 1)))
MAX_DATE = LocalizedDateTime(
    pytz.timezone("UTC").localize(datetime(9999, 12, 30, 23, 59, 59))
)


def localized_dtm(dtm: datetime) -> LocalizedDateTime:
    assert dtm.tzinfo is not None
    return LocalizedDateTime(dtm)


def localized_from_isoformat_with_default_tz(
    dtm_str: str, tz: pytz.BaseTzInfo
) -> LocalizedDateTime:
    dtm = datetime.fromisoformat(dtm_str.strip())

    if dtm.tzinfo is None:
        return LocalizedDateTime(tz.localize(dtm))
    else:
        return LocalizedDateTime(dtm)


def naive_from_isoformat(naive_str: str) -> NaiveDateTime:
    naive = datetime.fromisoformat(naive_str)
    assert naive.tzinfo is None

    return NaiveDateTime(naive)


def as_naive(
    localized: LocalizedDateTime, tz: pytz.BaseTzInfo
) -> Tuple[NaiveDateTime, bool]:
    assert localized.tzinfo is not None

    # localized remains a LocalizedDateTime after changing
    # the reference time zone
    localized = localized.astimezone(tz)

    naive = NaiveDateTime(localized.replace(tzinfo=None))
    is_dst = localized.dst() == timedelta(hours=1)

    return naive, is_dst


def localized_from_isoformat(
    localized_str: str, as_tz: Optional[pytz.BaseTzInfo] = None
) -> LocalizedDateTime:
    # pre py37
    # localized = iso8601.parse_date(localized_str)

    # to use instead when updated to py37
    localized = datetime.fromisoformat(localized_str)
    assert localized.tzinfo is not None

    if as_tz is not None:
        localized = localized.astimezone(as_tz)

    return LocalizedDateTime(localized)


def localize_datetime(
    naive: NaiveDateTime, is_dst: bool, time_zone: pytz.BaseTzInfo
) -> LocalizedDateTime:
    assert naive.tzinfo is None
    localized = time_zone.localize(naive, is_dst)
    return LocalizedDateTime(localized)


def increment(localized: LocalizedDateTime, delta: timedelta) -> LocalizedDateTime:
    assert localized.tzinfo is not None

    return LocalizedDateTime(
        (localized.astimezone(pytz.UTC) + delta).astimezone(localized.tzinfo)
    )


def iter_dt(
    start: LocalizedDateTime, inclusive_end: LocalizedDateTime, delta: timedelta
) -> Iterable[LocalizedDateTime]:
    assert inclusive_end >= start
    assert delta > timedelta(seconds=0)

    dt = start

    while dt <= inclusive_end:
        yield dt
        dt = increment(dt, delta)


def utcnow() -> LocalizedDateTime:
    now = localize_datetime(
        NaiveDateTime(datetime.utcnow()), is_dst=False, time_zone=pytz.utc
    )

    return now


def next_aligned(dt: LocalizedDateTime, delta: timedelta) -> LocalizedDateTime:
    return increment(_align(dt, delta), delta)


def _align(dt: LocalizedDateTime, delta: timedelta) -> LocalizedDateTime:
    # Alignment supported for intervals <= 1 hour which evenly divide 1 hour.
    # In this case, the proper alignment for every hour is independent.

    delta_secs = int(delta.total_seconds())
    assert delta_secs == delta.total_seconds()
    assert delta_secs <= 60 * 60
    assert 60 * 60 % delta_secs == 0

    hour_passed = dt - dt.replace(minute=0, second=0)
    extra_s = hour_passed.total_seconds() % delta.total_seconds()

    extra = timedelta(seconds=extra_s)

    return increment(dt, -extra)


def prev_aligned_on_or_before(
    dt: LocalizedDateTime, delta: timedelta
) -> LocalizedDateTime:
    return _align(dt, delta)


def is_aligned(dt: LocalizedDateTime, delta: timedelta) -> bool:
    return _align(dt, delta) == dt


def naive_datetime_or_none(
    localized_datetime: Optional[LocalizedDateTime],
) -> Optional[NaiveDateTime]:
    if localized_datetime is None:
        return None

    return NaiveDateTime(localized_datetime.replace(tzinfo=None))


def next_day_begining_timestamp(
    timestamp: LocalizedDateTime, time_zone: pytz.BaseTzInfo, n_days: int = 1
) -> LocalizedDateTime:
    timestamp_local = timestamp.astimezone(time_zone)

    return LocalizedDateTime(
        time_zone.localize(
            datetime.combine(timestamp_local.date(), datetime.min.time(), tzinfo=None)
            + timedelta(days=n_days)
        )
    )


def day_beginning(
    timestamp: LocalizedDateTime, timezone: pytz.BaseTzInfo
) -> LocalizedDateTime:
    local_timestamp = timestamp.astimezone(timezone)

    day_beginning_dtm = LocalizedDateTime(
        timezone.localize(
            datetime.combine(local_timestamp.date(), datetime.min.time(), tzinfo=None)
        )
    )

    return day_beginning_dtm
