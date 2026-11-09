"""Formatting is presentation only: retain facts, gates, and the JSON contract."""
import copy
import pytest

from gamelens.core.presentation import ANSWER_FORMAT_INSTRUCTION, reader_text
from gamelens.core.prompts import planner_prompt
from gamelens.core.synthesis import SYNTHESIS_PROMPT, expand_decision
from gamelens.core.investigation_checks import progress_limitation
from gamelens.core import answers


@pytest.mark.parametrize('profile', ['focused', 'extended'])
def test_all_planner_profiles_request_readable_markdown(profile):
    assert ANSWER_FORMAT_INSTRUCTION in planner_prompt(profile)
    assert '仍返回JSON' in planner_prompt(profile)
    assert ANSWER_FORMAT_INSTRUCTION in SYNTHESIS_PROMPT


def test_wording_and_synthesis_preserve_paragraphs_and_exact_citations():
    text = '现有反馈有限。\n\n**主要发现**\n\n- **崩溃**：有玩家提到闪退。\n- 暂时不能确认因果。\n\n**范围与限制**\n\n不代表全部玩家。'
    assert reader_text(text) == text
    quote = '开枪卡顿'
    result = expand_decision({'action': 'finish', 'reason': 'answer', 'final_answer': text,
        'review_evidence_ids': ['EV_R1'], 'news_evidence_ids': [], 'scope_limited': True},
        {'EV_R1': {'kind': 'review', 'citation': {'source_step': 1, 'review_id': 'r1', 'quote': quote}}})
    assert result.final_answer == text
    assert result.citations[0].quote == quote


@pytest.mark.parametrize('partial', [False, True])
def test_deterministic_statistics_are_formatted_without_changing_facts(monkeypatch, partial):
    monkeypatch.setattr(answers, 'verified_sources_current', lambda _: True)
    state = {'goal': '全部缓存评论数量与好评率', 'observations': [
        {'step': 1, 'tool': 'query_reviews', 'ok': True, 'data': {
            'filters': {'period': 'all'}, 'bounds': {},
            'stats': {'count': 5000, 'positive_count': 4166, 'negative_count': 834,
                      'positive_rate': .8332, 'negative_rate': .1668}}}]}
    original = copy.deepcopy(state)
    result = answers.statistical_answer(state, partial=partial)
    assert '\n\n- ' in result['answer']
    assert '\n\n**范围与限制**\n\n' in result['answer']
    for value in ('5000条', '4166条', '834条', '83.32%', '不代表Steam总体'):
        assert value in result['answer']
    assert len(result['claims']) == 5
    assert state == original
    if partial:
        assert '**调查未完成**' in result['answer']
        assert '恢复可能产生模型调用费用' in result['answer']


def test_unmet_aspects_are_separate_bullets_and_preserve_details():
    progress = {'enabled': True, 'gaps': [
        {'requirement': 'statistics', 'detail': '缺少更新前的评论样本'},
        {'requirement': 'announcement_check', 'detail': '没有找到目标公告'}],
        'budget_exhausted': True, 'needs_clarification': True}
    original = copy.deepcopy(progress)
    result = progress_limitation(progress)
    assert result.startswith('\n\n**调查范围未完整满足**\n\n- **精确统计**：')
    assert '\n- **官方公告核对**：没有找到目标公告' in result
    assert '未知不等于不存在' in result and '步数预算已用尽' in result and '请确认目标更新年份' in result
    assert progress == original


def test_no_empty_limitations_section():
    assert progress_limitation({'enabled': True, 'gaps': []}) == ''
