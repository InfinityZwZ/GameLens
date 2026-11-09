"""Opt-in exact read-only reuse over existing observations/receipts.

No fuzzy query merging or second cache store. A corpus/index change, missing
audit artifact, or old receipt without the identity metadata forces execution.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def file_digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ReadonlySnapshot:
    root: Path
    include_index: bool = False
    screening_model: str | None = None

    def __call__(self, args, state):
        appid = args.appid
        paths = {self.root / 'data/raw' / f'{appid}_{suffix}.json'
                 for suffix in ('reviews', 'patch_candidates', 'news')}
        paths.update((self.root/'data/raw/announcement_comments'/str(appid)).glob('*.json'))
        ledgers=list((self.root/'data/review_windows'/str(appid)).glob('*.json'))
        paths.update(ledgers)
        freshness=[]
        # An expired acquisition denominator must not survive read-only reuse.
        from gamelens.tools.review_coverage import ReviewCoverageTools
        coverage=ReviewCoverageTools(self.root)
        for ledger in sorted(ledgers):
            cached=coverage.read(appid,ledger.stem)
            freshness.append((ledger.stem,bool(cached)))
            if cached:paths.add(Path(cached[1]['path']))
        for key in ('reviews', 'patches', 'version_analysis'):
            if state.get('artifacts', {}).get(key):
                paths.add(Path(state['artifacts'][key]))
        if self.include_index:
            index = self.root / 'data/rag' / str(appid) / 'index.sqlite'
            if not index.is_file():
                return None
            paths.update((index, Path(str(index)+'-wal')))
        # Version the actual implementation, not an LLM judgement of equivalence.
        package = Path(__file__).resolve().parents[1]
        paths.update(package / relative for relative in (
            'tools/query_reuse.py', 'tools/investigation.py', 'tools/time_scope.py','tools/review_coverage.py','tools/steam_review_windows.py','core/query_contract.py',
            'rag/index.py', 'rag/embedding.py', 'rag/screening.py', 'rag/tools.py',
            'domain/models.py', 'domain/announcement_comments.py', 'tools/announcement_comments.py',
            'tools/steam_announcement_comments.py', 'core/local_time.py'))
        sources = [{'path':str(p.resolve()), 'sha256':file_digest(p)}
                   for p in sorted(paths, key=lambda p: str(p.resolve()))]
        from gamelens.core.local_time import timezone_signature
        return {'policy':'exact_readonly_local_time_v2', 'sources':sources, 'time_zone':timezone_signature(),
                'appid':state.get('appid'), 'goal':state.get('goal'),
                'patch_news_id':state.get('patch_news_id'),
                'window_days':state.get('window_days'),
                'profile':state.get('investigation_profile'),
                'query_contract':state.get('query_contract'),
                'review_window_freshness':freshness,
                'rag_mode':state.get('rag_mode', 'auto'),
                'screening_output_budget':os.getenv('LLM_SCREEN_MAX_TOKENS','4096') if self.include_index else None,
                'screening_model':self.screening_model}


def call_identity(tool, validated, state):
    if tool.readonly_snapshot is None:
        return None
    try:
        snapshot = tool.readonly_snapshot(validated, state)
        if snapshot is None:
            return None
        return {'identity':digest({'tool':tool.name,
                                  'schema':tool.args_model.model_json_schema(),
                                  'arguments':validated.model_dump(mode='json'),
                                  'snapshot':snapshot}),
                'snapshot_id':digest(snapshot)}
    except (OSError, ValueError, TypeError):
        return None  # Unavailable source cannot be used as a reusable identity.


def content_digest(data, artifacts):
    return digest({'data':{k:v for k,v in data.items() if k != 'execution'},
                   'artifacts':artifacts})


def stamp_result(result, identity):
    if identity is None:
        return
    artifacts = {k:('true' if v == 'true' else file_digest(v))
                 for k,v in result.artifacts.items()}
    result.data.setdefault('execution', {})['reuse'] = {
        **identity, 'content_digest':content_digest(result.data, result.artifacts),
        'artifact_hashes':artifacts}


def reusable(data, artifacts, identity):
    metadata = data.get('execution', {}).get('reuse', {})
    if not identity or metadata.get('identity') != identity['identity']:
        return False
    if metadata.get('content_digest') != content_digest(data, artifacts):
        return False
    try:
        current = {k:('true' if v == 'true' else file_digest(v)) for k,v in artifacts.items()}
        return all(v is not None for v in current.values()) and current == metadata.get('artifact_hashes')
    except OSError:
        return False


def prior_result(state, name, identity):
    for row in reversed(state.get('observations', [])):
        if row.get('tool') != name or not row.get('ok') or 'artifacts' not in row:
            continue
        if reusable(row.get('data', {}), row['artifacts'], identity):
            # JSON copy: never mutate the original checkpoint/observation.
            result = json.loads(json.dumps({k:row[k] for k in ('ok','summary','data','artifacts')}))
            result['data'].setdefault('execution', {}).update(
                reused_from_step=row['step'], handler_executed=False, elapsed_ms=0)
            return result
    return None


def identical_attempts(state, name, identity):
    if not identity:
        return 0
    return sum(o.get('tool') == name and
               o.get('data', {}).get('execution', {}).get('reuse', {}).get('identity') == identity['identity']
               for o in state.get('observations', []))
