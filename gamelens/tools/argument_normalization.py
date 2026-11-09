"""Explicit, auditable completion of unambiguous search arguments."""
from copy import deepcopy
import re


def normalize_search_arguments(arguments, state):
    result = deepcopy(arguments)
    changes = []
    terms = result.get('lexical_terms')
    if isinstance(terms,list) and all(isinstance(t,str) for t in terms):
        unique = list(dict.fromkeys(t.strip() for t in terms))
        limited = unique[:8]
        if limited != terms:
            result['lexical_terms'] = limited
            changes.append({'field':'lexical_terms','rule':'optional_ranking_terms_budget',
                'original_count':len(terms),'effective_count':len(limited),'omitted_terms':unique[8:]})
    from gamelens.rag.relevance import expand_lexical_terms
    expanded = expand_lexical_terms(str(result.get('query','')), result.get('lexical_terms') or [])
    if expanded != (result.get('lexical_terms') or []):
        original = list(result.get('lexical_terms') or [])
        result['lexical_terms'] = expanded
        changes.append({'field':'lexical_terms','rule':'deterministic_topic_expansion',
                        'original':original,'effective':expanded})
    goal = state.get('goal','')
    from gamelens.tools.time_scope import requested_directions
    directions=requested_directions(goal)
    after='after' in directions
    before='before' in directions
    filters = result.get('filters',{})
    if (result.get('patch_news_id') and after != before and isinstance(filters,dict) and
            filters.get('period','all') == 'all' and not filters.get('start_at') and not filters.get('end_at') and
            not re.search(r'全时段|所有版本|全部版本',goal)):
        # A patch ID by itself is never a time filter. Complete only a direction
        # literally stated by the user; never infer a game, date, ID or duration.
        result['filters'] = {**filters,'period':'after' if after else 'before'}
        changes.append({'field':'filters.period','rule':'explicit_user_direction',
                        'effective_value':result['filters']['period']})
    return result, changes
