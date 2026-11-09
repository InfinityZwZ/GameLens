"""Validate typed claims against tool observations, not keyword matches."""
import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from gamelens.core.source_freshness import active_observations


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1,max_length=100)
    source_step: int = Field(ge=1)
    fact_path: list[str | int] = Field(min_length=1,max_length=12)
    value: Any


class ReviewCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_step: int = Field(ge=1)
    review_id: str = Field(min_length=1,max_length=100)
    quote: str = Field(min_length=1,max_length=500)


class NewsCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_step: int = Field(ge=1)
    news_id: str = Field(min_length=1,max_length=100)
    quote: str = Field(min_length=1,max_length=500)


class AnnouncementCommentCitation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_step: int = Field(ge=1)
    news_id: str = Field(min_length=1, max_length=100)
    comment_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=500)


def validate_comment_citations(observations, citations):
    steps = {o.get('step'): o for o in active_observations(observations) if o.get('ok')}
    errors = []
    for citation in citations:
        data = steps.get(citation['source_step'], {}).get('data', {})
        record = next((c for c in data.get('comments', []) if c.get('comment_id') == citation['comment_id'] and
            c.get('news_id') == citation['news_id']), None)
        if (not record or not data.get('binding_verified') or not citation.get('quote', '').strip() or
                record.get('topic_id') != data.get('mapping', {}).get('topic_id') or
                citation['quote'] not in record.get('text', '')):
            errors.append('公告留言引用/所属公告不匹配：' + citation['comment_id'])
    return errors


def validate_news_citations(observations,citations):
    steps={o.get("step"):o for o in active_observations(observations) if o.get("ok")}
    errors=[]
    for citation in citations:
        records=steps.get(citation["source_step"],{}).get("data",{}).get("announcements",[])
        found=next((n for n in records if n.get("news_id")==citation["news_id"]),None)
        if found is None or citation["quote"] not in (found.get("contents") or found.get("title","")):
            errors.append(f"公告引用不匹配：{citation['news_id']}")
    return errors


def resolve_path(value,path):
    for key in path:
        value = value[key]
    return value


def same_value(expected,actual):
    if isinstance(expected,bool) or isinstance(actual,bool): return type(expected)==type(actual) and expected==actual
    if isinstance(expected,(int,float)) and isinstance(actual,(int,float)):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected,actual,abs_tol=1e-6,rel_tol=1e-6)
    return type(expected)==type(actual) and expected==actual


def validate_grounding(observations,claims,citations):
    steps = {item.get("step"):item for item in active_observations(observations) if item.get("ok")}
    errors = []
    for claim in claims:
        try:
            observation = steps[claim["source_step"]]
            actual = resolve_path(observation.get("data",{}),claim["fact_path"])
            if not same_value(actual,claim["value"]): errors.append(f"数字/事实不匹配：{claim['label']}")
        except (KeyError,IndexError,TypeError): errors.append(f"不存在的事实来源：{claim['label']}")
    for citation in citations:
        observation = steps.get(citation["source_step"],{})
        data = observation.get("data",{})
        reviews = data.get("reviews",[]) + ([data["review"]] if "review" in data else [])
        found = next((r for r in reviews if r.get("review_id")==citation["review_id"]),None)
        if found is None or citation["quote"] not in found.get("text",""):
            errors.append(f"原文引用不匹配：{citation['review_id']}")
    return errors


def fact_catalog(observations):
    """Program-generated identifiers prevent the planner inventing JSON paths."""
    result=[]
    excluded={"execution","reviews","review","announcements","comments","hits","reference_files","path","sha256","text","contents","page_content_hashes","representative_after_evidence","time_standard","time_zone"}
    def walk(value,path):
        if isinstance(value,dict):
            for key in sorted(value):
                if key not in excluded and not any(part in str(key).lower() for part in ("api_key","apikey","password","secret","authorization","token")):
                    yield from walk(value[key],path+[key])
        elif isinstance(value,list):
            for index,item in enumerate(value[:20]): yield from walk(item,path+[index])
        elif value is None or isinstance(value,(bool,int,float,str)):
            if not isinstance(value,str) or len(value)<=500: yield path,value
    for observation in active_observations(observations):
        if not observation.get("ok"):continue
        for index,(path,value) in enumerate(walk(observation.get("data",{}),[])):
            if not path:continue
            result.append({"fact_id":f"S{observation['step']}F{index+1}","source_step":observation["step"],
                           "fact_path":path,"value":value,"label":".".join(map(str,path))[-100:]})
    return result


def resolve_fact_ids(observations,ids):
    catalog={fact["fact_id"]:fact for fact in fact_catalog(observations)}
    claims=[];errors=[]
    for id in dict.fromkeys(ids):
        if id not in catalog: errors.append(f"不存在的事实编号：{id}")
        else: claims.append({k:v for k,v in catalog[id].items() if k!="fact_id"})
    return claims,errors
