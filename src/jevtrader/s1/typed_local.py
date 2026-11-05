"""`typed-local`: the open, non-generative typed decision model as an S1 backend (design_v2_delta IF-1; bank prompt
`ng-v1`; user decision D-1).

Model: `flock-io/this-that-model-1.0` at Hugging Face revision NG_REVISION (MIT; 1.88 B parameters), run on CPU through
the `thisthat` package (github.com/FLock-io/this-that-model at commit NG_PACKAGE_COMMIT, installed from that commit's
source archive, SHA-256 NG_ARCHIVE_SHA256) in the isolated environment `.venv-ng`, so the main environment's
dependencies do not change. The model runs in a worker process (`s1/ng_worker.py`) that this adapter starts once and
talks to over JSON lines; the worker stays up across event loops (runner calls), and requests are serialized.

Mapping (bank `prompts.ng-v1`; identical `(state, questions)` as every backend):
- state -> `state_text(state)`, the emu-v1 STATE block (JSON, indent 1);
- one thisthat `Question` per bank question, in render order; text = `_text(q.instructions)` (a Score's
  instructions object carries the question and its `judge_from`);
- options = exactly the strings emu-v1 renders after each letter (`backends.option_strings`): noul 'yes -- <true>',
  'no -- <false>'; choice '<name> -- <definition>'; score 'level k -- <level text>';
- all questions of a request in one `decide` call, i.e. one forward pass ("batching per request"); the returned
  distribution is mapped back to the option labels (`labels(q)` order);
- no retries (a malformed answer is unrepresentable); a worker error, a crash or a timeout -> probs None for every
  question of the request (abstain, counted, never cached), and the worker is restarted on the next request;
- the state is never truncated (`max_state_tokens` far above any state; the worker reports `truncated` otherwise);
- request key: backend id, the pinned `repo@revision`, the prompt version, `cfg_hash` (bank prompt, package commit,
  layout, temperature, dtype, max_state_tokens, parser) and the bank hash; cost $0 and no ledger entry; wall time,
  the model tokenizer's token count, the state's token count and the thread count go to `requests.parquet`
  (`backend_meta`).
- pin: the worker reports the revision it actually loaded (the snapshot directory's commit, and the model config's
  commit hash where the library records one), not the revision it was asked for; the adapter asserts it equals the
  pin (`ModelPinError`), and a mock worker never passes for the real model;
- the worker runs third-party code: it gets an allowlisted environment (`WORKER_ENV_KEYS`: paths, locale, Hugging
  Face cache location, thread settings), never the parent's API keys.
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jevtrader.s1.backends import ROOT, S1Backend, _bank_prompt, _text, config_hash, option_strings, state_text
from jevtrader.s1.types import Answer, JSONContent, Question, S1Response, labels, normalize

log = logging.getLogger("jevtrader.s1")

NG_REPO = "flock-io/this-that-model-1.0"
NG_REVISION = "3d927195c4f9845efe66c5715883a7a0f42b1239"  # docs/ng/this-that-model-1.0_hf_api_2026-09-26.json `sha`
NG_MODEL = f"{NG_REPO}@{NG_REVISION}"
NG_PACKAGE_COMMIT = "542d445efa5f68b14bfbd1f8ed25aacd8379d839"  # github.com/FLock-io/this-that-model main, 2026-09-26
NG_ARCHIVE_SHA256 = "bc9ec14cfae5381f85907bd853e2d0d41c52576be28f0f581afb08679d4545c0"  # <commit>.tar.gz from GitHub
WORKER = Path(__file__).with_name("ng_worker.py")
DEFAULT_PYTHON = ROOT / ".venv-ng" / "bin" / "python"
PARSER = "typed-local-v1"
# the only parent environment variables the worker sees (it needs no credential: the model loads offline)
WORKER_ENV_KEYS = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ",
                   "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
                   "TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "KMP_AFFINITY",
                   "KMP_BLOCKTIME")


def worker_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The worker's environment: the allowlisted parent variables, offline Hugging Face, no tokenizer threads."""
    env = {k: os.environ[k] for k in WORKER_ENV_KEYS if k in os.environ}
    return {**env, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", **(extra or {})}


class WorkerError(RuntimeError):
    pass


def ng_question(q: Question) -> dict[str, Any]:
    """A bank question as the thisthat `Question` payload (text, options in `labels(q)` order)."""
    return {"text": _text(q.instructions), "options": option_strings(q)}


def ng_payload(state: JSONContent, questions: Mapping[str, Question]) -> dict[str, Any]:
    return {"state": state_text(state), "questions": [ng_question(q) for q in questions.values()]}


class WorkerProcess:
    """One `ng_worker.py` subprocess, spoken to over JSON lines from worker threads (loop-independent, so the model
    stays loaded across `generate()` calls). Requests are serialized; a timeout kills the worker."""

    def __init__(self, python: str | Path, args: list[str], startup_timeout: float = 900.0,
                 request_timeout: float = 600.0, env: Mapping[str, str] | None = None, log_path: Path | None = None):
        self.python, self.args = str(python), list(args)
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.env = worker_env(env)
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.info: dict[str, Any] = {}
        self._buf = b""
        self._lock = threading.Lock()
        self._next_id = 0
        self.restarts = 0

    # ------------------------------------------------------------------ process
    def _start(self) -> None:
        err = open(self.log_path, "ab") if self.log_path is not None else subprocess.DEVNULL  # noqa: SIM115
        try:
            # -I (isolated): the script's directory must not shadow the standard library (s1/types.py vs `types`)
            self.proc = subprocess.Popen([self.python, "-I", str(WORKER), *self.args], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=err, env=self.env, cwd=str(ROOT))
        finally:
            if err is not subprocess.DEVNULL:
                err.close()  # the child holds its own descriptor
        self._buf = b""
        ready = self._read_line(self.startup_timeout)
        if not ready.get("ready"):
            self.close()
            raise WorkerError(f"typed-local worker failed to start: {ready.get('error', ready)}")
        self.info = ready
        log.info("typed-local worker ready: %s", {k: ready.get(k) for k in ("revision", "threads", "dtype", "load_s")})

    def _read_line(self, timeout: float) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close()
                raise TimeoutError(f"typed-local worker gave no answer within {timeout:.0f} s")
            r, _, _ = select.select([fd], [], [], min(left, 1.0))
            if r:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    code = self.proc.poll()
                    self.close()
                    raise WorkerError(f"typed-local worker exited (code {code})")
                self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def request(self, payload: dict) -> dict:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                if self.proc is not None:
                    self.restarts += 1
                self._start()
            self._next_id += 1
            rid = self._next_id
            assert self.proc is not None and self.proc.stdin is not None
            try:
                self.proc.stdin.write((json.dumps({"id": rid, **payload}) + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self.close()
                raise WorkerError(f"typed-local worker pipe failed: {e}") from e
            out = self._read_line(self.request_timeout)
            if out.get("id") != rid:
                self.close()
                raise WorkerError(f"typed-local worker answered request {out.get('id')}, expected {rid}")
            return out

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is not None:
            try:
                if p.stdin:
                    p.stdin.close()
                p.terminate()
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                p.kill()


class TypedDecisionLocal(S1Backend):
    backend_id = "typed-local"
    prompt_version = "ng-v1"
    uses_ledger = False  # $0 API cost; CPU time is budgeted in the run manifest instead (design_v2_delta §6)
    needs_api_key = False

    def __init__(self, model: str = NG_MODEL, *, python: str | Path | None = None, threads: int = 12,
                 dtype: str = "float32", max_state_tokens: int = 32768, request_timeout: float = 600.0,
                 startup_timeout: float = 900.0, package_commit: str = NG_PACKAGE_COMMIT, mock: bool = False,
                 transport: Any = None, log_path: str | Path | None = None, **kw):
        kw.setdefault("degraded_retries", 0)  # no retries: a malformed answer is unrepresentable
        kw.setdefault("max_concurrency", 1)  # one CPU worker; requests are serialized anyway
        kw.pop("rpm", None)
        super().__init__(model, **kw)
        self.prompt = _bank_prompt(self.prompt_version)
        if self.prompt["model"] != model:
            raise ValueError(f"{self.prompt_version} pins {self.prompt['model']!r}; got model {model!r}")
        repo, _, revision = model.partition("@")
        if not revision:
            raise ValueError("typed-local needs a pinned '<repo>@<revision>' model id")
        self.repo, self.revision = repo, revision
        self.threads, self.dtype, self.max_state_tokens = int(threads), str(dtype), int(max_state_tokens)
        self.package_commit, self.mock = package_commit, bool(mock)
        self._cfg_hash = config_hash({
            "prompt_version": self.prompt_version, "prompt": self.prompt, "package_commit": package_commit,
            "layout": "state_first", "temperature": 1.0, "dtype": self.dtype, "max_state_tokens": self.max_state_tokens,
            "parser": PARSER, "mock": self.mock,
        })
        args = ["--model", repo, "--revision", revision, "--threads", str(self.threads), "--dtype", self.dtype,
                "--max-state-tokens", str(self.max_state_tokens)] + (["--mock"] if self.mock else [])
        py = Path(python) if python else DEFAULT_PYTHON
        py = py if py.is_absolute() else ROOT / py
        self._transport = transport if transport is not None else WorkerProcess(
            py, args, startup_timeout=startup_timeout, request_timeout=request_timeout,
            log_path=Path(log_path) if log_path else None)
        self.failures = 0  # requests that abstained because the worker failed or timed out

    @property
    def cfg_hash(self) -> str:
        return self._cfg_hash

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"package_commit": self.package_commit, "threads": self.threads, "dtype": self.dtype,
                  "worker": getattr(self._transport, "info", {}), "failures": self.failures})
        return d

    def _cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        return 0.0

    def estimate_cost(self, state, questions) -> float:
        return 0.0

    async def _ask(self, state: JSONContent, questions: Mapping[str, Question]) -> S1Response:
        import asyncio

        payload = ng_payload(state, questions)
        t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(self._transport.request, payload)
            if "error" in out:
                raise WorkerError(out["error"])
            probs = out["probs"]
            if len(probs) != len(questions):
                raise WorkerError(f"{len(probs)} answers for {len(questions)} questions")
        except (WorkerError, TimeoutError, OSError, ValueError, KeyError) as e:
            self.failures += 1
            log.warning("typed-local: request abstains (%s: %s)", type(e).__name__, e)
            answers = {name: Answer(q.kind, None) for name, q in questions.items()}
            return S1Response(answers=answers, model=self.model, backend=self.backend_id, response_model=self.model,
                              degraded=list(questions), meta={"error": f"{type(e).__name__}: {e}",
                                                              "wall_s": round(time.perf_counter() - t0, 4)})
        answers, degraded = {}, []
        for (name, q), p in zip(questions.items(), probs):
            labs = labels(q)
            dist = normalize(dict(zip(labs, (float(x) for x in p)))) if len(p) == len(labs) else None
            if dist is None:
                degraded.append(name)
            answers[name] = Answer(q.kind, dist)
        info = getattr(self._transport, "info", {}) or {}
        # the revision the worker loaded (ng_worker reports the snapshot's commit, not its argument); a worker that
        # reports nothing or runs the mock model does not pass for the pinned model
        served = f"{info.get('model', '?')}@{info.get('revision', '?')}"
        if bool(info.get("mock", False)) != self.mock:
            served += f" (mock={bool(info.get('mock', False))})"
        meta = {"wall_s": out.get("wall_s"), "n_tokens": out.get("n_tokens"), "state_tokens": out.get("state_tokens"),
                "truncated": out.get("truncated"), "threads": info.get("threads"), "maxrss_gb": out.get("maxrss_gb")}
        return S1Response(answers=answers, model=self.model, backend=self.backend_id,
                          input_tokens=int(out.get("n_tokens") or 0), output_tokens=0, response_model=served,
                          degraded=degraded, meta=meta)

    def close(self) -> None:
        if hasattr(self._transport, "close"):
            self._transport.close()
