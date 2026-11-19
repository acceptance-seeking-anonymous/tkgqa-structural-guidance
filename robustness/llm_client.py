"""llm_client.py -- small provider-agnostic client for the two API-served LLM roles of the robustness probes:
the R1 rewrite generator and the LLM judge (R1 and R2). Any endpoint that implements the de-facto standard
/v1/chat/completions interface works (a hosted API, a gateway, or a local vLLM / TGI server).

Configuration (environment variables, or KEY=VALUE lines in the file named by LLM_CONFIG; see configs/llm.env.example):
  LLM_API_BASE      base URL of the chat-completions API, e.g. http://localhost:8000/v1
  LLM_API_KEY       bearer token (leave empty for a local server without authentication)
  GEN_MODEL_ID      model ID of the R1 rewrite generator
  JUDGE_MODEL_ID    model ID of the judge
  GEMINI_API_KEY    key of the Google Gemini API used by the R3 generator (robustness/r3/local/r3_pipeline.py reads it
                    with setting(); the Gemini calls use the provider's generateContent endpoint)
  LLM_TEMPERATURE   optional; omitted from the request when unset (provider default)
  LLM_TIMEOUT       request timeout in seconds (default 120); LLM_RETRIES (default 5)

complete(requests, max_concurrency=16) takes a list of {"prompt", "max_tokens", optional "model"} dicts and returns,
position by position, {"text", "model", "usage": {"input_tokens", "output_tokens"}, "stop_reason"} or {"error": ...}.
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PLACEHOLDER = '<SET_TO_MODEL_ID_REPORTED_IN_PAPER>'


def _load_config():
    path = os.environ.get('LLM_CONFIG')
    if path and os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setting(name, default=None):
    _load_config()
    return os.environ.get(name, default)


def model_id(role):
    """role: 'gen' or 'judge'"""
    v = setting({'gen': 'GEN_MODEL_ID', 'judge': 'JUDGE_MODEL_ID'}[role], PLACEHOLDER)
    if not v or v == PLACEHOLDER:
        raise RuntimeError(f'set {"GEN_MODEL_ID" if role == "gen" else "JUDGE_MODEL_ID"} (see configs/llm.env.example)')
    return v


def _one(req):
    base = setting('LLM_API_BASE')
    if not base:
        raise RuntimeError('set LLM_API_BASE (see configs/llm.env.example)')
    body = {'model': req['model'], 'messages': [{'role': 'user', 'content': req['prompt']}],
            'max_tokens': int(req.get('max_tokens', 300))}
    temp = req.get('temperature', setting('LLM_TEMPERATURE'))
    if temp not in (None, ''):
        body['temperature'] = float(temp)
    headers = {'Content-Type': 'application/json'}
    key = setting('LLM_API_KEY')
    if key:
        headers['Authorization'] = f'Bearer {key}'
    url = base.rstrip('/') + '/chat/completions'
    last = None
    for attempt in range(int(setting('LLM_RETRIES', '5'))):
        try:
            r = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(r, timeout=float(setting('LLM_TIMEOUT', '120'))) as resp:
                d = json.loads(resp.read().decode('utf-8'))
            ch = d['choices'][0]
            u = d.get('usage') or {}
            return {'text': (ch.get('message') or {}).get('content'), 'model': d.get('model', req['model']),
                    'usage': {'input_tokens': u.get('prompt_tokens', 0), 'output_tokens': u.get('completion_tokens', 0)},
                    'stop_reason': ch.get('finish_reason')}
        except Exception as e:          # network / rate-limit errors: exponential back-off
            last = e
            time.sleep(min(60, 2 ** attempt))
    return {'error': f'{type(last).__name__}: {last}', 'text': None, 'model': req['model'], 'usage': {}}


def complete(requests, max_concurrency=16, role=None):
    """requests: list of dicts with 'prompt', 'max_tokens' and optionally 'model'; if 'model' is missing it is taken
    from the role ('gen' or 'judge')."""
    reqs = []
    for q in requests:
        q = dict(q)
        if 'model' not in q:
            q['model'] = model_id(role or 'judge')
        reqs.append(q)
    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as ex:
        return list(ex.map(_one, reqs))
