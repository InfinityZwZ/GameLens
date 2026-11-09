"""Small, validated patches to a rejected final draft; never a bypass of gates."""
from typing import Literal, Any
from pydantic import BaseModel, ConfigDict, Field
from gamelens.core.grounding import GroundedClaim, ReviewCitation, NewsCitation, AnnouncementCommentCitation
from gamelens.tools.query_reuse import digest


class TextEdit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    field: Literal['final_answer','detailed_answer']
    old: str = Field(min_length=1,max_length=1000)
    new: str = Field(max_length=1800)


class FinalChanges(BaseModel):
    model_config = ConfigDict(extra='forbid')
    claims: list[GroundedClaim] | None = Field(default=None,max_length=64)
    fact_ids: list[str] | None = Field(default=None,max_length=64)
    citations: list[ReviewCitation] | None = Field(default=None,max_length=20)
    news_citations: list[NewsCitation] | None = Field(default=None,max_length=20)
    comment_citations: list[AnnouncementCommentCitation] | None = Field(default=None,max_length=20)
    scope_limited: bool | None = None


class FinalRepair(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['patch','replan']
    reason: str = Field(min_length=1,max_length=500)
    changes: FinalChanges = Field(default_factory=FinalChanges)
    text_edits: list[TextEdit] = Field(default_factory=list,max_length=8)
    action: str | None = Field(default=None,max_length=80)
    arguments: dict[str,Any] = Field(default_factory=dict)


def draft_signature(state):
    return digest({'goal':state.get('goal'),'appid':state.get('appid'),
        'profile':state.get('investigation_profile'),'window_days':state.get('window_days'),
        'patch_news_id':state.get('patch_news_id'),
        'sources':[{'step':o.get('step'),'tool':o.get('tool'),'source':o.get('data',{}).get('source')}
                   for o in state.get('observations',[]) if o.get('ok') and o.get('data',{}).get('source')]})


def apply_repair(draft, payload):
    from gamelens.core.runtime import PlannerDecision
    # Old compatible providers/mocks may still return a complete decision. It
    # passes the same validation and final gates; no extra fallback request.
    if 'mode' not in payload:
        return PlannerDecision.model_validate(payload)
    patch = FinalRepair.model_validate(payload)
    changes = patch.changes.model_dump(mode='json',exclude_none=True)
    if patch.mode == 'replan':
        if changes or patch.text_edits or not patch.action or patch.action == 'finish':
            raise ValueError('补查模式只允许注册工具行动，不同时修改或结束答案')
        return PlannerDecision(action=patch.action,reason=patch.reason,arguments=patch.arguments)
    if patch.action or patch.arguments:
        raise ValueError('答案补丁不能同时执行工具')
    result = {**draft, **changes, 'reason':patch.reason}
    for edit in patch.text_edits:
        original = result.get(edit.field) or ''
        if original.count(edit.old) != 1:
            raise ValueError('文字补丁须匹配原答案中唯一的原文片段')
        result[edit.field] = original.replace(edit.old,edit.new,1)
    for key in ('claims','fact_ids','citations','news_citations','comment_citations'):
        if key in changes and len(changes[key]) < len(draft.get(key, [])):
            if not (changes.get('scope_limited') is True and patch.text_edits):
                raise ValueError('不能仅删除错误事实或引用以通过门禁；须纠正、补查或修改文字明确范围限制')
    if not changes and not patch.text_edits:
        raise ValueError('修正补丁没有修改任何字段')
    return PlannerDecision.model_validate(result)
