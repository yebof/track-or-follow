"""Worker process of the typed-local backend (design_v2_delta IF-1): runs in the isolated `.venv-ng` environment.

It loads `flock-io/this-that-model-1.0` once (CPU, pinned Hugging Face revision) through the `thisthat` package and
answers JSON-lines requests on stdin, one per line, in order:

    request:  {"id": <int>, "state": <str>, "questions": [{"text": <str>, "options": [<str>, ...]}, ...]}
    response: {"id": <int>, "probs": [[p, ...], ...], "n_tokens": <int>, "state_tokens": <int>,
               "truncated": <bool>, "wall_s": <float>, "maxrss_gb": <float>}
    or        {"id": <int>, "error": "<type>: <message>"}

All questions of a request are answered in one forward pass (`TypedDecider.decide`, state-first layout); nothing is
generated, so an answer outside the declared options is unrepresentable. The first line written is the ready record
(`{"ready": true, "model", "revision", "threads", "dtype", "load_s", ...}`); its `revision` is the one actually
loaded (the commit of the local snapshot directory, cross-checked with the model config's commit hash when the
library records one), never an echo of `--revision`. Only protocol lines go to stdout: the model libraries' own
output is redirected to stderr. `--mock` answers with a deterministic hash-based distribution, imports neither torch
nor thisthat, and says so in the ready record (`"mock": true`; tests of the transport and the adapter).

This file must stay importable by a bare interpreter: standard library only at module level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time


def _maxrss_gb() -> float:
    """Peak resident set of this process in GB: VmHWM (per address space, reset by exec) where /proc exists; else
    ru_maxrss, which on Linux also counts the forking parent's pages before exec (an overestimate)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1e6  # kB
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kilobytes


class MockDecider:
    """Deterministic stand-in: p over the options is a softmax of hash bytes of (state, question, option)."""

    def __init__(self, revision: str):
        self.revision = revision  # nothing is loaded: the ready record says mock

    def answer(self, state: str, questions: list[dict]) -> tuple[list[list[float]], int, int]:
        import math

        out = []
        for q in questions:
            z = []
            for o in q["options"]:
                h = hashlib.sha256(json.dumps([state, q["text"], o]).encode()).digest()
                z.append(3.0 * (h[0] / 255.0 - 0.5) * 2)
            e = [math.exp(v) for v in z]
            out.append([v / sum(e) for v in e])
        n_state = len(state.split())
        return out, n_state + sum(len(q["text"].split()) + sum(len(o.split()) for o in q["options"])
                                  for q in questions), n_state


class ThisThatDecider:
    def __init__(self, model: str, revision: str, threads: int, dtype: str, max_state_tokens: int):
        import torch

        torch.set_num_threads(threads)
        from huggingface_hub import snapshot_download
        from thisthat import Question, TypedDecider
        from thisthat.prompt import build

        self.torch, self.Question, self.build = torch, Question, build
        self.max_state_tokens = max_state_tokens
        # load tokenizer AND weights from the pinned local snapshot (`from_pretrained(name, revision=...)` would pin
        # only the weights: thisthat loads the tokenizer without the revision); never fetch at run time
        path = snapshot_download(model, revision=revision, local_files_only=True)
        self.snapshot = os.path.normpath(path)
        self.revision = os.path.basename(self.snapshot)  # the Hub cache names a snapshot directory by its commit
        if self.revision != revision:
            raise RuntimeError(f"snapshot {path} is revision {self.revision}, not the pinned {revision}")
        self.decider = TypedDecider.from_pretrained(path, device="cpu", dtype=dtype)
        loaded = getattr(getattr(self.decider.model, "config", None), "_commit_hash", None)
        if loaded and loaded != self.revision:  # transformers records the commit of the files it read
            raise RuntimeError(f"model config reports commit {loaded}, snapshot {self.revision}")
        self.threads = torch.get_num_threads()
        self.dtype = str(self.decider.model.dtype)

    def answer(self, state: str, questions: list[dict]) -> tuple[list[list[float]], int, int]:
        qs = [self.Question(q["text"], q["options"]) for q in questions]
        tok = self.decider.tokenizer
        built = self.build(tok, state, qs, max_state_tokens=self.max_state_tokens)
        state_tokens = len(tok.encode("Context:\n" + state, add_special_tokens=False))
        ds = self.decider.decide(state, qs, max_state_tokens=self.max_state_tokens)
        return [list(d.probabilities) for d in ds], len(built["ids"]), state_tokens


def ready_record(a: argparse.Namespace, dec, info: dict, load_s: float) -> dict:
    """The first protocol line: what was loaded (the decider's revision, not the `--revision` argument)."""
    return {"ready": True, "model": a.model, "revision": dec.revision, "requested_revision": a.revision,
            "mock": a.mock, "load_s": round(load_s, 3), "pid": os.getpid(), "max_state_tokens": a.max_state_tokens,
            "maxrss_gb": round(_maxrss_gb(), 3), **info}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flock-io/this-that-model-1.0")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-state-tokens", type=int, default=32768)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args(argv)
    proto = os.fdopen(os.dup(1), "w", buffering=1)  # protocol channel; everything else printed goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    t0 = time.perf_counter()
    try:
        if a.mock:
            dec, info = MockDecider(a.revision), {"threads": 0, "dtype": "mock", "package": "mock"}
        else:
            dec = ThisThatDecider(a.model, a.revision, a.threads, a.dtype, a.max_state_tokens)
            import thisthat

            info = {"threads": dec.threads, "dtype": dec.dtype, "package": f"thisthat {thisthat.__version__}"}
    except Exception as e:  # noqa: BLE001 (reported to the client, which abstains)
        proto.write(json.dumps({"ready": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
        return 1
    proto.write(json.dumps(ready_record(a, dec, info, time.perf_counter() - t0)) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            t1 = time.perf_counter()
            probs, n_tokens, state_tokens = dec.answer(req["state"], req["questions"])
            out = {"id": rid, "probs": probs, "n_tokens": n_tokens, "state_tokens": state_tokens,
                   "truncated": state_tokens > a.max_state_tokens, "wall_s": round(time.perf_counter() - t1, 4),
                   "maxrss_gb": round(_maxrss_gb(), 3)}
        except Exception as e:  # noqa: BLE001 (one bad request abstains; the worker keeps serving)
            out = {"id": rid, "error": f"{type(e).__name__}: {e}"}
        proto.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
