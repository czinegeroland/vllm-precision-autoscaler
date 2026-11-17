"""lm_eval CLI with one fix for vLLM's OpenAI server: at most 4 stop sequences.

lm-eval's local-completions sends the task's `until` list plus the EOS token as `stop`;
HumanEval+ ends up with 5 and vLLM rejects the request (400, "at most 4 items").
vLLM stops at EOS anyway, so the EOS entry is dropped; if a task still has more than 4,
the first 4 are sent and every generation is cut at the earliest of all of them.

    python3 bench/lm_eval_vllm.py <normal lm_eval arguments>
"""
import sys

import lm_eval.models.openai_completions as oc
from lm_eval.__main__ import cli_evaluate

API = oc.LocalCompletionsAPI
_orig_payload = API._create_payload
_FULL_STOP: list[str] = []


def _payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kw):
    p = _orig_payload(self, messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos, **kw)
    stop = p.get("stop") if generate else None
    if isinstance(stop, list) and len(stop) > 4:
        if eos in stop:
            stop = [s for s in stop if s != eos]
        if len(stop) > 4:
            _FULL_STOP[:] = stop
            stop = stop[:4]
        p["stop"] = stop
    return p


_orig_parse = API.parse_generations


def _parse(outputs, **kw):
    res = _orig_parse(outputs, **kw)
    if not _FULL_STOP:
        return res
    out = []
    for text in res:
        if text is not None:
            cut = min((i for i in (text.find(s) for s in _FULL_STOP) if i >= 0), default=-1)
            text = text[:cut] if cut >= 0 else text
        out.append(text)
    return out


API._create_payload = _payload
API.parse_generations = staticmethod(_parse)

if __name__ == "__main__":
    sys.argv[0] = "lm_eval"
    cli_evaluate()
