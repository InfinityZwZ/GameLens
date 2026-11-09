"""Shared cache adequacy rules for bounded Steam collections."""


def cache_satisfies_target(collected_count: int, target_count: int, *, complete: bool = False) -> bool:
    """A cache is sufficient when it meets the request or the source is exhausted."""
    if type(collected_count) is not int or collected_count < 0:
        raise ValueError("collected_count must be a non-negative integer")
    if type(target_count) is not int or target_count < 1:
        raise ValueError("target_count must be a positive integer")
    return bool(complete or collected_count >= target_count)


def candidate_improves_cache(
    existing_count: int,
    candidate_count: int,
    target_count: int,
    *,
    existing_complete: bool = False,
    candidate_complete: bool = False,
    existing_error: str | None = None,
    candidate_error: str | None = None,
) -> bool:
    """Choose a refreshed snapshot without replacing a better cache with a failure."""
    existing_ok = cache_satisfies_target(existing_count, target_count, complete=existing_complete)
    candidate_ok = cache_satisfies_target(candidate_count, target_count, complete=candidate_complete)
    if bool(candidate_error) != bool(existing_error):
        return not candidate_error
    if candidate_ok != existing_ok:
        return candidate_ok
    if candidate_count != existing_count:
        return candidate_count > existing_count
    if candidate_complete != existing_complete:
        return candidate_complete
    return False
