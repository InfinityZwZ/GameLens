"""Retain audit history without using a superseded locator as current evidence.

Only an identical successful locator on the same source files may supersede an
earlier locator. Different queries, evidence reads and failed retries never do.
The newer files still have to pass the ordinary SHA256 verification gate.
"""
import json


def _locator_key(observation):
    if observation.get('tool') not in {'list_announcements', 'list_patches'} or not observation.get('ok'):
        return None
    arguments = observation.get('arguments')
    source = observation.get('data', {}).get('source', {})
    if not arguments or not source.get('path'):
        return None
    files = [source, *source.get('reference_files', [])]
    if any(not item.get('path') or not item.get('sha256') for item in files):
        return None
    return (observation['tool'], json.dumps(arguments, sort_keys=True, default=str),
            tuple(sorted(item['path'] for item in files)))


def superseded_locator_steps(observations):
    latest, superseded = {}, {}
    for observation in sorted(observations, key=lambda o: o.get('step', 0), reverse=True):
        key = _locator_key(observation)
        if key is None:
            continue
        if key in latest:
            superseded[observation['step']] = latest[key]
        else:
            latest[key] = observation['step']
    return superseded


def active_observations(observations):
    superseded = superseded_locator_steps(observations)
    return [o for o in observations if o.get('step') not in superseded]
