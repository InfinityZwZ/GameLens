import gzip,json,time,socket
import pytest,httpx
from gamelens.core.llm import OpenAICompatibleClient
from gamelens.storage.telemetry import TelemetryStore,summarize

@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*a,**k):raise AssertionError('No real network in tests')
    monkeypatch.setattr(socket.socket,'connect',denied)

@pytest.mark.parametrize('status,compressed',[(200,True),(400,True),(200,False)])
def test_json_fallback_preserves_decoded_body_status_and_safe_headers(status,compressed):
    body=json.dumps({'error':{'code':'invalid_request'}} if status==400 else {'choices':[], 'usage':{'prompt_tokens':7}}).encode()
    def handle(req):return httpx.Response(status,content=gzip.compress(body) if compressed else body,
        headers={'Content-Type':'application/json',**({'Content-Encoding':'gzip'} if compressed else {}),'x-request-id':'request-1','authorization':'secret'},request=req)
    model=OpenAICompatibleClient('test',api_key='test');event={};start=time.monotonic()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        response=model._stream_response(client,'https://example.invalid',{}, {},10,event,start+10,start)
    assert response.status_code==status and response.content==body
    assert response.headers['x-request-id']=='request-1'
    assert event['http_status']==status and event['response_content_type']=='application/json'
    assert 'secret' not in json.dumps(event)

@pytest.mark.parametrize('kind',['decode','unknown'])
def test_exception_finalizes_request_and_never_retries(tmp_path,kind):
    store=TelemetryStore(tmp_path/'events.sqlite');calls=[]
    def handle(req):
        calls.append(req)
        if kind=='unknown':raise RuntimeError('unexpected')
        return httpx.Response(200,stream=httpx.ByteStream(b'invalid gzip'),headers={'content-type':'text/event-stream','content-encoding':'gzip','x-request-id':'bad-1'},request=req)
    model=OpenAICompatibleClient('test',api_key='test',max_retries=2,telemetry_sink=lambda e:store.record('t',e));start=time.monotonic()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.DecodingError if kind=='decode' else RuntimeError):
            model._post_with_retry(client,'https://example.invalid',{}, {'stream':True},deadline=start+10)
    events=store.events('t');assert len(calls)==1
    assert events[-1]['status']=='failed' and events[-1]['latency_ms']>=0
    if kind=='decode':assert events[-1]['error_code']=='response_decoding_failed' and events[-1]['http_status']==200
    assert summarize(events)['pending_requests']==0 and summarize(events)['failed_requests']==1

def test_compressed_sse_still_validates_finish_and_usage():
    chunks=[{'choices':[{'index':0,'delta':{'content':'{}'},'finish_reason':'stop'}]}, {'choices':[],'usage':{'prompt_tokens':3,'completion_tokens':2}}]
    body=(''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode()
    def handle(req):return httpx.Response(200,content=gzip.compress(body),headers={'content-type':'text/event-stream','content-encoding':'gzip'},request=req)
    model=OpenAICompatibleClient('test',api_key='test');start=time.monotonic()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        response=model._stream_response(client,'https://example.invalid',{}, {},10,{},start+10,start)
    assert response.json()['usage']=={'prompt_tokens':3,'completion_tokens':2}
    assert response.json()['choices'][0]['message']['content']=='{}'

def test_corrupt_error_body_retains_http_status_for_batch_stop(tmp_path):
    def handle(req):return httpx.Response(400,stream=httpx.ByteStream(b'invalid gzip'),headers={'content-encoding':'gzip'},request=req)
    model=OpenAICompatibleClient('test',api_key='test',max_retries=0)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.DecodingError) as caught:model._post_with_retry(client,'https://example.invalid',{}, {'stream':True})
    assert getattr(caught.value,'status_code',None)==400
