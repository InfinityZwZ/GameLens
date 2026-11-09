import json
import os
import time
import uuid
import re
from urllib.parse import urlsplit
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from gamelens.storage.telemetry import price_from_env, token_count
from gamelens.core.decision_trace import safe_audit
from gamelens.core.budget import BudgetExceeded, TaskBudgetStore

CALL_ID = ContextVar("llm_call_id", default=None)
OUTPUT_BUDGET = ContextVar("llm_output_budget", default=0)
CALL_PHASE = ContextVar("llm_call_phase", default="json")


# 始终从项目根目录加载配置；已存在的系统/终端环境变量优先。
from gamelens.core.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    def __init__(self, message, *, status_code=None, error_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def error_description(error=None, status=None):
    if error is not None:
        name = type(error).__name__
        detail = {'ReadTimeout':'读取响应超时（不是额度报错）',
                  'ConnectTimeout':'连接超时','WriteTimeout':'上传请求超时',
                  'PoolTimeout':'连接池等待超时'}.get(name,'网络/连接异常')
        return name, detail
    detail = {408:'供应商请求超时',429:'限流或配额限制，需查看供应商错误码',
              500:'供应商内部错误',502:'网关错误',503:'服务暂不可用',504:'网关响应超时',
              401:'认证失败',403:'访问权限不足',402:'计费或余额限制'}.get(status,'接口请求错误')
    return f'HTTP{status}',detail


class OpenAICompatibleClient:
    """最小化的 OpenAI-compatible Chat Completions 客户端。"""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        telemetry_sink=None,
        request_budget: TaskBudgetStore | None = None,
        budget_thread_id: str | None = None,
    ) -> None:
        self.model = model.strip()
        self.telemetry_sink = telemetry_sink
        self.request_budget = request_budget
        self.budget_thread_id = budget_thread_id
        try:
            self.pricing = price_from_env(os.environ)
        except ValueError as error:
            raise LLMConfigurationError(str(error)) from error
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        trust_env = os.getenv('LLM_TRUST_ENV', 'true').strip().lower()
        if trust_env not in {'true', 'false'}:
            raise LLMConfigurationError('LLM_TRUST_ENV须为true或false')
        self.trust_env = trust_env == 'true'
        # Auto tuning is restricted to officially documented mixed-thinking models.
        known_qwen = bool(re.fullmatch(r'qwen3\.(?:8-(?:max|flash)|7-(?:max|flash|plus))(?:-(?:\d{4}|\d{4}-\d{2}-\d{2}))?', self.model.lower()))
        provider = (urlsplit(self.base_url).hostname or '').endswith('.aliyuncs.com')
        thinking = os.getenv('LLM_ENABLE_THINKING', 'auto').strip().lower()
        streaming = os.getenv('LLM_STREAMING', 'auto').strip().lower()
        if thinking not in {'auto', 'true', 'false'} or streaming not in {'auto', 'true', 'false'}:
            raise LLMConfigurationError('LLM_ENABLE_THINKING与LLM_STREAMING须为auto/true/false')
        self.enable_thinking = (False if provider and known_qwen else None) if thinking == 'auto' else thinking == 'true'
        self.streaming = (provider and known_qwen) if streaming == 'auto' else streaming == 'true'
        try:
            self.timeout = (
                timeout
                if timeout is not None
                else float(os.getenv("LLM_TIMEOUT", "180"))
            )
            self.max_retries = (
                max_retries
                if max_retries is not None
                else int(os.getenv("LLM_MAX_RETRIES", "3"))
            )
            self.max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")) or 4096
            self.request_deadline = float(os.getenv('LLM_REQUEST_DEADLINE', '240'))
            self.stream_read_timeout = float(os.getenv('LLM_STREAM_READ_TIMEOUT', '30'))
        except ValueError as error:
            raise LLMConfigurationError(
                "LLM_TIMEOUT必须是数字，重试次数与输出预算必须是整数"
            ) from error

        if not self.model:
            raise LLMConfigurationError("LLM模型名不能为空")
        if not self.api_key:
            raise LLMConfigurationError(
                "缺少LLM_API_KEY或OPENAI_API_KEY环境变量"
            )
        if self.timeout <= 0:
            raise LLMConfigurationError("LLM_TIMEOUT必须大于0")
        if self.max_retries < 0:
            raise LLMConfigurationError("LLM_MAX_RETRIES不能小于0")
        if not 0 < self.request_deadline <= 1800:
            raise LLMConfigurationError('LLM_REQUEST_DEADLINE须在0..1800秒之间')
        if not 0 < self.stream_read_timeout <= 180:
            raise LLMConfigurationError('LLM_STREAM_READ_TIMEOUT须在0..180秒之间')
        if not 0 <= self.max_output_tokens <= 16384:
            raise LLMConfigurationError("LLM_MAX_OUTPUT_TOKENS必须在0..16384；0表示安全默认4096")

    def _post_with_retry(
        self,
        client: httpx.Client,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        fallback: bool = False,
        deadline: float | None = None,
    ) -> httpx.Response:
        """对超时、网络中断、限流和服务端错误进行有限重试。"""
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        deadline = deadline if deadline is not None else time.monotonic() + self.request_deadline

        # Repeating an entire final answer after a read stall wastes time and can
        # duplicate supplier billing. Evidence/checkpoints survive for recovery.
        attempts = 1 if CALL_PHASE.get() == 'synthesis' else self.max_retries + 1
        for attempt in range(attempts):
            started = time.monotonic()
            remaining = deadline - started
            if remaining <= 0:
                raise LLMResponseError('LLM请求总等待预算已耗尽，停止重试；已有证据已保留。',
                                       status_code=408,error_code='retry_deadline_exceeded')
            if self.request_budget is not None:
                if not self.budget_thread_id:
                    raise LLMConfigurationError("启用任务预算时必须提供budget_thread_id")
                try:
                    self.request_budget.checkpoint_active(self.budget_thread_id)
                    self.request_budget.reserve_llm_request(self.budget_thread_id)
                except BudgetExceeded as error:
                    raise LLMResponseError(str(error), error_code=error.code) from error
            event = {
                "id": uuid.uuid4().hex,
                "kind": "request",
                "call_id": CALL_ID.get(),
                "model": self.model,
                "phase": CALL_PHASE.get(),
                "streaming": self.streaming,
                "thinking_enabled": self.enable_thinking,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "retry": attempt > 0,
                "fallback": fallback and attempt == 0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "estimated_cost": None,
                "currency": self.pricing["currency"],
            }
            self._emit(event)
            try:
                timeout = httpx.Timeout(min(self.timeout,remaining),connect=min(30,remaining),
                                        write=min(30,remaining),pool=min(30,remaining))
                if payload.get('stream'):
                    timeout = httpx.Timeout(min(self.stream_read_timeout,remaining),connect=min(30,remaining),
                                            write=min(30,remaining),pool=min(30,remaining))
                    response = self._stream_response(client, endpoint, headers, payload, timeout, event, deadline, started)
                else:
                    response = client.post(endpoint, headers=headers, json=payload, timeout=timeout)
            except LLMResponseError as error:
                event.update(status='failed',error_type=type(error).__name__,error_code=error.error_code,
                             latency_ms=round((time.monotonic()-started)*1000,1))
                self._emit(event)
                raise
            except httpx.DecodingError as error:
                error.status_code=event.get('http_status')
                error.error_code='response_decoding_failed'
                event.update(status='failed',error_type='DecodingError',error_code='response_decoding_failed',
                             error_detail='响应解压/解码失败，未自动重试',latency_ms=round((time.monotonic()-started)*1000,1))
                self._emit(event)
                raise
            except httpx.TransportError as error:
                label,detail = error_description(error=error)
                event.update(status="failed", error_type=label, error_detail=detail, latency_ms=round((time.monotonic()-started)*1000, 1))
                self._emit(event)
                print(f'LLM请求失败 [{label}]：{detail}；本次等待{event["latency_ms"]/1000:.1f}s。',flush=True)
                if attempt >= attempts - 1 or event.get('content_chars',0):
                    raise
            except Exception as error:
                event.update(status='failed',error_type=type(error).__name__,error_code='unexpected_request_error',
                             error_detail='请求处理异常，未自动重试',latency_ms=round((time.monotonic()-started)*1000,1))
                self._emit(event)
                raise
            else:
                provider_code = None
                try:
                    body = response.json()
                    error_body = body.get('error',{}) if isinstance(body,dict) else {}
                    if isinstance(error_body,dict) and isinstance(error_body.get('code'),str):
                        provider_code = safe_audit(error_body['code'])[:100]
                    usage = body.get("usage", {}) if isinstance(body, dict) else {}
                    if not isinstance(usage, dict):
                        usage = {}
                except ValueError:
                    usage = {}
                input_tokens = token_count(usage.get("prompt_tokens", usage.get("input_tokens")))
                output_tokens = token_count(usage.get("completion_tokens", usage.get("output_tokens")))
                details = usage.get('completion_tokens_details',{})
                if isinstance(details,dict):
                    event['reasoning_tokens'] = token_count(details.get('reasoning_tokens'))
                event.update(
                    status="succeeded" if response.is_success else "failed",
                    http_status=response.status_code,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    latency_ms=round((time.monotonic()-started)*1000, 1),
                )
                rates = self.pricing
                if input_tokens is not None and output_tokens is not None and rates["input_per_million"] is not None and rates["output_per_million"] is not None:
                    event["estimated_cost"] = (input_tokens*rates["input_per_million"] + output_tokens*rates["output_per_million"])/1_000_000
                    event["pricing"] = dict(rates)
                if not response.is_success:
                    label,detail = error_description(status=response.status_code)
                    if provider_code:
                        code_lower=provider_code.lower()
                        if any(k in code_lower for k in ('insufficient_quota','balance','arrears','billing')):
                            detail='供应商明确返回配额/余额限制'
                        elif any(k in code_lower for k in ('rate','throttl')):
                            detail='供应商明确返回速率限制'
                    event.update(error_type=label,error_detail=detail,provider_error_code=provider_code)
                self._emit(event)
                if not response.is_success:
                    print(f'LLM请求失败 [{event["error_type"]}]：{event["error_detail"]}。'+
                          (f'供应商错误码：{provider_code}' if provider_code else ''),flush=True)
                if (
                    response.status_code not in retryable_statuses
                    or attempt >= attempts - 1
                ):
                    return response

            wait_seconds = min(2**attempt, 8)
            if time.monotonic() + wait_seconds >= deadline:
                raise LLMResponseError('LLM请求总等待预算已耗尽，停止重试；已有证据已保留。',
                                       status_code=408,error_code='retry_deadline_exceeded')
            print(
                "LLM请求暂时失败，"
                f"{wait_seconds}秒后进行第{attempt + 2}次尝试（总等待上限{self.request_deadline:g}s）……",
                flush=True,
            )
            time.sleep(wait_seconds)

        raise RuntimeError("LLM请求重试流程异常结束")

    def _stream_response(self, client, endpoint, headers, payload, timeout, event, deadline, started):
        """Buffer validated SSE; publish counters only, never answer/reasoning text."""
        parts, usage, finish = [], {}, None
        done = False
        last_notice = started
        characters = reasoning_characters = 0

        def consume(raw):
            nonlocal done, usage, finish, characters, reasoning_characters, last_notice
            if raw.strip() == '[DONE]':
                done = True
                return
            try:
                chunk = json.loads(raw)
            except ValueError as error:
                raise LLMResponseError('流式响应包含无效事件JSON',error_code='invalid_stream') from error
            if not isinstance(chunk,dict) or chunk.get('error'):
                raise LLMResponseError('流式响应返回错误事件',error_code='invalid_stream')
            if isinstance(chunk.get('usage'),dict):
                usage = chunk['usage']
            choices = chunk.get('choices',[])
            if not isinstance(choices,list):
                raise LLMResponseError('流式响应choices格式无效',error_code='invalid_stream')
            for choice in choices:
                if not isinstance(choice,dict) or choice.get('index',0) != 0:
                    continue
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta',{})
                if not isinstance(delta,dict):
                    raise LLMResponseError('流式响应delta格式无效',error_code='invalid_stream')
                content = delta.get('content')
                if content is not None and not isinstance(content,str):
                    raise LLMResponseError('流式正文不是字符串',error_code='invalid_stream')
                if content:
                    parts.append(content); characters += len(content)
                    event.setdefault('first_content_ms',round((time.monotonic()-started)*1000,1))
                thought = delta.get('reasoning_content')
                if isinstance(thought,str):
                    reasoning_characters += len(thought)  # Never retain private reasoning.
            if characters + reasoning_characters > 200000:
                raise LLMResponseError('流式响应超过安全大小上限',error_code='response_too_large')
            event.update(content_chars=characters,reasoning_chars=reasoning_characters)
            now = time.monotonic()
            if now-last_notice >= 1 or 'first_content_ms' in event and event.get('status') != 'streaming':
                event.update(status='streaming',elapsed_ms=round((now-started)*1000,1))
                self._emit(event); last_notice = now

        with client.stream('POST',endpoint,headers=headers,json=payload,timeout=timeout) as response:
            event['first_byte_ms'] = round((time.monotonic()-started)*1000,1)
            event.update(http_status=response.status_code,
                         response_content_type=safe_audit(response.headers.get('content-type',''))[:200],
                         response_content_encoding=safe_audit(response.headers.get('content-encoding',''))[:100])
            request_id=response.headers.get('x-request-id') or response.headers.get('x-dashscope-request-id')
            if request_id:event['provider_request_id']=safe_audit(request_id)[:200]
            if not response.is_success or 'text/event-stream' not in response.headers.get('content-type',''):
                response.read()  # Some compatible providers ignore stream and return JSON.
                decoded_headers={k:v for k,v in response.headers.items() if k.lower() not in {'content-encoding','content-length'}}
                return httpx.Response(response.status_code,content=response.content,headers=decoded_headers,request=response.request)
            fragments = []
            for line in response.iter_lines():
                if time.monotonic() >= deadline:
                    raise LLMResponseError('流式请求总等待预算已耗尽',status_code=408,error_code='retry_deadline_exceeded')
                if line.startswith('data:'):
                    fragments.append(line[5:].lstrip())
                elif not line and fragments:
                    consume('\n'.join(fragments)); fragments=[]
                    if done:
                        break
            if fragments and not done:
                consume('\n'.join(fragments))
            if not done or finish is None:
                raise LLMResponseError('流式响应未正常结束，已有片段不能当完整答案',error_code='incomplete_stream')
            data = {'choices':[{'message':{'content':''.join(parts)},'finish_reason':finish}], 'usage':usage}
            return httpx.Response(200,json=data,request=response.request)

    def _emit(self, event):
        if self.telemetry_sink is not None:
            self.telemetry_sink(dict(event))

    def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        call_id = uuid.uuid4().hex
        token = CALL_ID.set(call_id)
        started = time.monotonic()
        event = {"id": call_id, "call_id": call_id, "kind": "call", "model": self.model, "status": "pending",
                 "phase":CALL_PHASE.get(), "streaming":self.streaming,"thinking_enabled":self.enable_thinking,
                 "started_at":datetime.now(timezone.utc).isoformat(),
                 "prompt_chars": sum(len(m.get("content", "")) for m in messages),
                 "output_budget": OUTPUT_BUDGET.get() or self.max_output_tokens or None}
        self._emit(event)
        try:
            result = self._generate_json(messages)
            event["status"] = "succeeded"
            return result
        except Exception as error:
            code = getattr(error,'error_code',None)
            detail = {'output_truncated':'达到输出预算，响应被截断；需要调整阶段预算或缩短输出',
                      'invalid_json':'模型返回的正文不是有效JSON',
                      'invalid_response':'供应商返回格式缺少预期字段',
                      'retry_deadline_exceeded':'客户端重试等待预算已耗尽，已有证据已保留',
                      'llm_request_budget_exhausted':'任务级LLM HTTP请求预算已耗尽，未发送本次请求',
                      'active_time_budget_exhausted':'任务级活跃执行时间预算已耗尽'}.get(code)
            if code in {'invalid_stream','incomplete_stream','response_too_large'}:
                detail = '流式响应格式/结束状态异常，未展示或接受未校验的片段'
            if detail is None:
                _,detail = error_description(error=error if isinstance(error,httpx.TransportError) else None,
                                            status=getattr(error,'status_code',None))
            event.update(status="failed", error_type=type(error).__name__,
                         error_code=code,error_detail=detail,http_status=getattr(error,'status_code',None))
            print(f'LLM调用失败 [{code or type(error).__name__}]：{detail}；已保留已有证据。',flush=True)
            raise
        finally:
            event["latency_ms"] = round((time.monotonic()-started)*1000, 1)
            self._emit(event)
            CALL_ID.reset(token)

    def _generate_json(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        budget = OUTPUT_BUDGET.get() or self.max_output_tokens
        if budget:
            payload["max_tokens"] = budget
        if self.enable_thinking is not None:
            payload['enable_thinking'] = self.enable_thinking
        if self.streaming:
            payload.update(stream=True,stream_options={'include_usage':True})

        with httpx.Client(timeout=self.timeout, trust_env=self.trust_env) as client:
            deadline = time.monotonic() + self.request_deadline
            response = self._post_with_retry(
                client,
                endpoint,
                headers,
                payload,
                deadline=deadline,
            )

            # 某些兼容服务不支持 response_format，自动降级重试一次。
            try:
                rejection = response.json() if response.status_code == 400 else None
            except ValueError:
                rejection = None
            format_rejection = rejection == {} or (isinstance(rejection,dict) and
                any(term in json.dumps(rejection).lower() for term in ('response_format','json_object','json_schema')))
            if response.status_code == 400 and format_rejection:
                payload.pop("response_format", None)
                response = self._post_with_retry(
                    client,
                    endpoint,
                    headers,
                    payload,
                    fallback=True,
                    deadline=deadline,
                )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                response_detail = safe_audit(response.text.strip()[:1000])
                raise LLMResponseError(
                    f"LLM接口返回HTTP {response.status_code}: "
                    f"{response_detail or '响应正文为空'}", status_code=response.status_code,
                    error_code=f'http_{response.status_code}',
                ) from error
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            if data["choices"][0].get("finish_reason") == "length":
                raise LLMResponseError("模型输出达到长度上限，不能把截断内容当完整结果；可调整当前阶段输出预算。",error_code='output_truncated')
            if data["choices"][0].get("finish_reason") == 'content_filter':
                raise LLMResponseError('供应商拒绝生成正文',error_code='content_filtered')
        except (KeyError, IndexError, TypeError) as error:
            raise LLMResponseError("LLM响应缺少消息正文",error_code='invalid_response') from error

        if not isinstance(content, str):
            raise LLMResponseError("LLM响应正文不是字符串")

        cleaned_content = content.strip()
        if cleaned_content.startswith("```"):
            lines = cleaned_content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_content = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned_content)
        except json.JSONDecodeError as error:
            raise LLMResponseError("LLM没有返回有效JSON",error_code='invalid_json') from error

        if not isinstance(parsed, dict):
            raise LLMResponseError("LLM JSON最外层必须是对象")

        return parsed
