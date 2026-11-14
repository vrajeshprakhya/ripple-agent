"""Run a chat model over benchmark problems through any OpenAI-compatible endpoint.

The same code serves both arms of the A/B: OpenRouter for a quick base-model run, or a Modal (vLLM
or similar) endpoint serving the base checkpoint and the checkpoint plus LoRA adapter. Everything
that could differ between arms is fixed here and recorded in the output: system prompt, decoding
settings, and no tools.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request

SYSTEM = ("You are an expert electrical engineer. Solve the problem and reply with only the final answer "
          "in exactly the requested form, with its SI unit.")
# Ripple as served by GMICloud reasons in tokens it never returns (neither content nor the reasoning
# field), about 2,300 for one RC question, then answers in a few characters. 160 and then 1024
# tokens ended most replies before the answer, so those runs measured the budget, not the model.
# Unused budget costs nothing. Both A/B arms must use this same value.
DECODING = {"temperature": 0.0, "max_tokens": 8192}


def _post(url: str, key: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(problem: dict, *, base_url: str, model: str, key: str, openrouter: bool, retries: int = 6) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": problem["prompt"]}], **DECODING}
    if openrouter:
        body["provider"] = {"order": ["GMICloud", "PrimeIntellect"], "allow_fallbacks": True}
    last = ""
    for attempt in range(retries):
        t0 = time.time()
        try:
            d = _post(base_url.rstrip("/") + "/chat/completions", key, body)
            msg = d["choices"][0]["message"]
            return {"id": problem["id"], "response": msg.get("content") or "", "latency_s": round(time.time() - t0, 2),
                    "provider": d.get("provider"), "usage": d.get("usage"),
                    "finish_reason": d["choices"][0].get("finish_reason")}
        except urllib.error.HTTPError as e:
            last = f"HTTPError: {e}"
            if 400 <= e.code < 500 and e.code != 429:  # a malformed request will not improve by retrying
                break
            time.sleep(min(30, 2 ** (attempt + 1)))
        except (urllib.error.URLError, KeyError, IndexError, TimeoutError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(30, 2 ** (attempt + 1)))
    return {"id": problem["id"], "response": "", "error": last}


def run(problems: list[dict], *, base_url: str, model: str, key: str, openrouter: bool = False,
        workers: int = 6) -> list[dict]:
    with cf.ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(ask, p, base_url=base_url, model=model, key=key, openrouter=openrouter) for p in problems]
        out = [f.result() for f in futs]
    order = {p["id"]: i for i, p in enumerate(problems)}
    return sorted(out, key=lambda r: order[r["id"]])
