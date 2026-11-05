"""Generative LLM baselines: the TA-style pipeline "TA-lite" (B14) and Mono-CoT (B15) (design_v1 §13.3; design_v2_delta
FR-7, QB-9, RP-1, RB-17, PE-12).

Both read the same blinded point-in-time evidence as the typed system, from the bank prompts `ta-lite-v1` and
`mono-cot-v1` (question_bank_v1.json, prompts are ours: decision O6), on the pinned snapshot `gpt-4.1-mini-2025-04-14`
at temperature 0, and every output goes through the one fixed parser `gen-parse-v1` (`baselines/parse.py`).

TA-lite follows the roles of xiao2024tradingagents (analysts, bull/bear debate, trader, risk manager; p.5-6) in 8
capped calls per stock-week: 4 analysts (fundamentals, technical, news, macro and policy) in parallel, one bull and
one bear researcher in parallel (one debate round), the trader (reason-first JSON, QB-9), and the risk manager. Per-role
output caps are the bank's `max_tokens`. The final decision is the risk output if valid, else the trader's; both
invalid: HOLD with an abstaining p (counted). Its native book is EW over BUY names (at most 20, by p then symbol hash).

Mono-CoT is one call: generative chain of thought, then a final JSON line with a verbalized p (`p_outperform`).

Inputs (`build_inputs`), all blinded and point in time (cutoff c_t):
- technical: the T state (`programs.technical.TechnicalProgram.states`);
- fundamentals: the F-B state (`programs.fundamentals.FundamentalsProgram.states`);
- news: EXACTLY the H reader's evidence (`h_evidence`): the items of `programs.news.ItemSource` (the news analyst's
  window, ticker filter, D0, duplicate rule keeping the earliest copy, empty headlines dropped, the 40 most recent,
  blinding), rendered as `programs.readers.HolisticReader.news_items` renders them ({news_items: [{headline, summary,
  published}]});
- macro and policy: the M state (`programs.macro.MacroProgram.state`) plus the headlines of the week's P1 items
  (`programs.policy.policy_candidates`: market-tagged or > 3-ticker items matching the policy lexicon, the 60 with the
  smallest sha256(id)), D0-sanitized, then blinded with the item's tagged firms as P1 does.

Calls go through `GenLLM`: the response model is asserted equal to the pin, every call is charged to the spend
ledger under its tag before any check can raise, and responses are cached by a content hash of (model, messages,
max_tokens, temperature), so replays are free and identical. The budget guard (and the ledger check) run AFTER a call
has its concurrency slot and RESERVE the call's projected cost until it settles, so concurrent calls cannot all pass
the check at spent = 0.

A DEGRADED response (empty content, or a finish reason other than `stop`/`length`, e.g. `content_filter` or a
transient empty 200) is never cached: the call is retried up to `degraded_retries` times (every attempt is charged),
and if it is still degraded the Call is returned with `degraded=True` and is not cached, so a rerun asks again. A
degraded entry found in an older cache is treated as a miss. `result_frames` reports degraded calls separately from
invalid outputs (a degraded call's empty text also parses as invalid).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from jevtrader.baselines.parse import (
    PARSER_SHA256,
    PARSER_VERSION,
    Parsed,
    TaFinal,
    last_json_object,
    parse_analyst,
    parse_mono_cot,
    ta_final,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL = "gpt-4.1-mini-2025-04-14"
CACHE_PATH = ROOT / "data/cache/gen_v1.sqlite"
ROLES = ("fundamentals", "technical", "news", "macro_policy")
POLICY_CAP = 60  # policy items per week (design_v1 §5.5)
OK_FINISH = ("stop", "length")  # any other finish reason (content_filter, tool_calls, "", ...) is degraded
DEGRADED_RETRIES = 2  # as the S1 backends: a degraded answer is retried twice


def is_degraded(text: str | None, finish_reason: str | None) -> bool:
    """Empty content or an abnormal finish reason: not a model answer, so never cached (module doc)."""
    return not (text or "").strip() or str(finish_reason or "") not in OK_FINISH


class BudgetExceeded(RuntimeError):
    pass


def _bank_prompt(pid: str) -> dict:
    from jevtrader.programs.bank_v1 import load_bank

    return load_bank().prompts[pid]


def _fill(template: str, **kw: str) -> str:
    """Placeholder replacement in ONE pass over the template (the templates carry literal JSON braces, so never
    str.format): a placeholder-like string inside an inserted value (untrusted news, a model's output) is never
    expanded by a later replacement."""
    if not kw:
        return template
    pat = re.compile(r"\{(" + "|".join(re.escape(k) for k in sorted(kw, key=len, reverse=True)) + r")\}")
    return pat.sub(lambda m: kw[m.group(1)], template)


def _as_text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------------------------------------ the LLM client
@dataclass
class Call:
    role: str
    request_key: str
    text: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    usd: float
    latency_s: float
    cached: bool
    response_model: str
    finish_reason: str
    max_tokens: int
    degraded: bool = False  # still degraded after the retries (not cached; counted apart from invalid outputs)
    attempts: int = 1  # paid attempts behind this Call (tokens and USD are summed over them)


class _GenCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS gen (key TEXT PRIMARY KEY, response TEXT)")
        self._db.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM gen WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, rec: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO gen VALUES (?, ?)", (key, json.dumps(rec)))
            self._db.commit()


class GenLLM:
    """Pinned OpenAI chat client for the generative baselines (pin assertion, ledger, cache, optional USD budget)."""

    backend_id = "gen-openai"

    def __init__(self, model: str = MODEL, *, client=None, ledger=None, cache: bool | Path = True, tag: str = "",
                 budget_usd: float | None = None, max_concurrency: int = 32, degraded_retries: int = DEGRADED_RETRIES):
        if model != MODEL:
            raise ValueError(f"the generative baselines pin {MODEL!r}; got {model!r}")
        self.model, self.tag, self.ledger, self.budget_usd = model, tag, ledger, budget_usd
        self.degraded_retries = int(degraded_retries)
        self.degraded_calls = 0  # Calls returned degraded after the retries
        self._client = client
        self._cache = _GenCache(Path(cache) if isinstance(cache, (str, Path)) else CACHE_PATH) if cache else None
        self.spent = 0.0
        self.reserved = 0.0  # projected cost of the calls in flight (checked with `spent`)
        self.calls = 0
        self._budget_lock = threading.Lock()
        self.max_concurrency = max_concurrency
        self._sems: dict[int, asyncio.Semaphore] = {}  # one per event loop (a GenLLM may serve several asyncio.run)

    @property
    def client(self):
        if self._client is None:
            import os

            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=120.0, max_retries=4)
        return self._client

    def key(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        blob = json.dumps({"model": self.model, "system": system, "user": user, "max_tokens": max_tokens,
                           "temperature": temperature}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    async def _once(self, system: str, user: str, max_tokens: int, temperature: float) -> dict:
        """One paid request: reserve, call, settle, charge, assert the pin. Returns the response record."""
        from jevtrader.s1.backends import ModelPinError, estimate_tokens
        from jevtrader.s1.pricing import cost_usd

        projected = cost_usd(self.model, estimate_tokens(system + user) + 12, max_tokens)
        sem = self._sems.setdefault(id(asyncio.get_running_loop()), asyncio.Semaphore(self.max_concurrency))
        async with sem:
            with self._budget_lock:  # check and reserve together (no await in between)
                if self.budget_usd is not None and self.spent + self.reserved + projected > self.budget_usd:
                    raise BudgetExceeded(f"{self.tag}: {self.spent:.4f} spent + {self.reserved:.4f} in flight + "
                                         f"{projected:.4f} projected > {self.budget_usd}")
                if self.ledger is not None:
                    self.ledger.check(self.reserved + projected, self.tag)
                self.reserved += projected
            try:
                t0 = time.monotonic()
                r = await self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": user}],
                    temperature=temperature, max_completion_tokens=max_tokens)
                latency = time.monotonic() - t0
            except BaseException:
                with self._budget_lock:
                    self.reserved -= projected
                raise
        u = getattr(r, "usage", None)
        if u is None or getattr(u, "prompt_tokens", None) is None:
            n_in, n_out, n_cached = estimate_tokens(system + user) + 12, max_tokens, 0
        else:
            d = getattr(u, "prompt_tokens_details", None)
            n_in, n_out = int(u.prompt_tokens), int(u.completion_tokens or 0)
            n_cached = int(getattr(d, "cached_tokens", 0) or 0) if d is not None else 0
        usd = cost_usd(self.model, n_in, n_out, n_cached)
        with self._budget_lock:  # settle the reservation
            self.reserved -= projected
            self.spent += usd
            self.calls += 1
        if self.ledger is not None:  # booked before any check can raise: the call is paid for
            await self.ledger.acharge(self.backend_id, self.model, self.tag, n_in, n_out, usd)
        if getattr(r, "model", None) != self.model:
            raise ModelPinError(f"asked {self.model}, answered by {getattr(r, 'model', None)}")
        ch = r.choices[0]
        return {"text": ch.message.content or "", "input_tokens": n_in, "cached_tokens": n_cached,
                "output_tokens": n_out, "usd": usd, "response_model": r.model,
                "finish_reason": str(getattr(ch, "finish_reason", "") or ""), "latency_s": latency}

    async def complete(self, role: str, system: str, user: str, max_tokens: int, temperature: float = 0.0) -> Call:
        key = self.key(system, user, max_tokens, temperature)
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            if not is_degraded(hit["text"], hit["finish_reason"]):  # an older cache may hold a degraded entry
                return Call(role, key, hit["text"], hit["input_tokens"], hit["cached_tokens"], hit["output_tokens"],
                            hit["usd"], 0.0, True, hit["response_model"], hit["finish_reason"], max_tokens)
        recs = []
        for _ in range(1 + max(self.degraded_retries, 0)):
            rec = await self._once(system, user, max_tokens, temperature)
            recs.append(rec)
            if not is_degraded(rec["text"], rec["finish_reason"]):
                break
        rec = recs[-1]
        bad = is_degraded(rec["text"], rec["finish_reason"])
        tot = {k: sum(x[k] for x in recs) for k in ("input_tokens", "cached_tokens", "output_tokens", "usd",
                                                     "latency_s")}
        if bad:
            self.degraded_calls += 1
        elif self._cache is not None:
            self._cache.put(key, {k: v for k, v in rec.items() if k != "latency_s"})
        return Call(role, key, rec["text"], tot["input_tokens"], tot["cached_tokens"], tot["output_tokens"], tot["usd"],
                    tot["latency_s"], False, rec["response_model"], rec["finish_reason"], max_tokens, bad, len(recs))


# ------------------------------------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class StockWeekInputs:
    symbol: str
    t: pd.Timestamp
    code_name: str
    fundamentals: str
    technical: str
    news: str
    macro_policy: str
    sources: dict = field(default_factory=dict)

    def block(self, role: str) -> str:
        return getattr(self, role)


def h_evidence(ctx, symbol: str, bank=None) -> dict:
    """{news_items: [{headline, summary, published}]}: the H reader's evidence for the stock-week, built by the same
    code (`ItemSource` items rendered by `HolisticReader.news_items`), so TA-lite, Mono-CoT, H and the typed news
    analyst read the same items."""
    from jevtrader.programs.news import ItemSource
    from jevtrader.programs.readers import HolisticReader

    reader = HolisticReader(bank)
    rows, _ = reader.news_items(ctx, symbol, ItemSource(ctx, reader.bank).items(symbol))
    return {"news_items": rows}


def policy_headlines(ctx, cap: int = POLICY_CAP) -> list[str]:
    """Headlines of the week's P1 items (`programs.policy.policy_candidates`), D0-sanitized, then blinded with the
    item's tagged firms as P1 renders them; oldest first."""
    from jevtrader.programs import render
    from jevtrader.programs.policy import policy_candidates

    start = ctx.prev_cutoff_utc if ctx.prev_cutoff_utc is not None else ctx.cutoff_utc - pd.Timedelta(days=7)
    cand = policy_candidates(ctx.news.t, start, ctx.cutoff_utc, cap)
    year = int(pd.Timestamp(ctx.t).year)
    out = []
    for _, it in cand.sort_values("created_at", kind="stable").iterrows():
        h0 = render.d0(it["headline"], render.HEADLINE_MAX)
        out.append(render.blind_clean(h0, ctx.blinder, decision_year=year, cutoff=ctx.cutoff_utc,
                                      tagged=tuple(it["tags"])))
    return out


def build_inputs(ctx, symbol: str, bank=None) -> StockWeekInputs:
    """The four input blocks of one stock-week, blinded and point in time (module doc)."""
    from jevtrader.programs import render
    from jevtrader.programs.bank_v1 import load_bank
    from jevtrader.programs.fundamentals import FundamentalsProgram
    from jevtrader.programs.macro import MacroProgram
    from jevtrader.programs.technical import TechnicalProgram

    bank = bank or load_bank()
    sources = {}
    tech = TechnicalProgram(bank).states(ctx, [symbol]).get(symbol)
    fund = FundamentalsProgram(bank).states(ctx, [symbol]).get(symbol)
    macro_state, _ = MacroProgram(bank).state(ctx)
    firm = ctx.firms[symbol]
    focal = render.focal_company(bank, ctx.blinder.code_name(symbol), firm.industry, firm.size_rank)
    news = h_evidence(ctx, symbol, bank)
    sources["news"] = "programs.news.ItemSource + programs.readers.HolisticReader.news_items"
    macro = {"indicators": macro_state.get("indicators"), "calendar": macro_state.get("calendar"),
             "policy_news_headlines_this_week": policy_headlines(ctx)}
    sources.update(technical="programs.technical" if tech else "not available",
                   fundamentals="programs.fundamentals" if fund else "not available",
                   macro_policy="programs.macro + programs.policy.policy_candidates")
    return StockWeekInputs(
        symbol=symbol, t=pd.Timestamp(ctx.t), code_name=focal["code_name"],
        fundamentals=_as_text(fund[0]) if fund else render.NOT_AVAILABLE,
        technical=_as_text(tech[0]) if tech else render.NOT_AVAILABLE,
        news=_as_text({"focal_company": focal, **news}), macro_policy=_as_text(macro), sources=sources)


# ------------------------------------------------------------------------------------------------ TA-lite
@dataclass
class TaLiteResult:
    symbol: str
    t: pd.Timestamp
    calls: list[Call]
    stances: dict[str, Parsed]
    final: TaFinal

    @property
    def invalid_counts(self) -> dict[str, int]:
        return {"analyst_stance": sum(not p.valid for p in self.stances.values()),
                "trader": int(not self.final.trader.valid), "risk": int(not self.final.risk.valid),
                "final": int(self.final.invalid)}


class TaLite:
    name = "TA-lite"
    prompt_id = "ta-lite-v1"

    def __init__(self, llm: GenLLM, prompt: dict | None = None):
        self.llm = llm
        self.p = prompt if prompt is not None else _bank_prompt(self.prompt_id)
        if self.p["decoding"]["model"] != llm.model or self.p["parser_version"] != PARSER_VERSION:
            raise ValueError("ta-lite-v1 pins another model or parser than this code")
        if self.p["calls_per_stock_week"] != 8:
            raise ValueError("TA-lite makes exactly 8 calls per stock-week")

    def _call(self, role: str, user: str, kind: str):
        return self.llm.complete(role, self.p["system"], user, int(self.p["max_tokens"][kind]),
                                 float(self.p["decoding"]["temperature"]))

    @staticmethod
    def reports_text(reports: dict[str, str]) -> str:
        return "\n\n".join(f"[{name}]\n{text.strip()}" for name, text in reports.items())

    async def run(self, x: StockWeekInputs) -> TaLiteResult:
        p = self.p
        analysts = await asyncio.gather(*[
            self._call(role, _fill(p["analyst_template"], role=p["roles"][role]["role"], input=x.block(role)),
                       "analyst") for role in ROLES])
        reports = {role: c.text for role, c in zip(ROLES, analysts)}
        rep = self.reports_text(reports)
        bull, bear = await asyncio.gather(self._call("bull", _fill(p["bull_template"], reports=rep), "bull"),
                                          self._call("bear", _fill(p["bear_template"], reports=rep), "bear"))
        trader = await self._call("trader", _fill(p["trader_template"], reports=rep, bull=bull.text.strip(),
                                                  bear=bear.text.strip()), "trader")
        obj = last_json_object(trader.text)
        proposal = json.dumps(obj, ensure_ascii=False) if obj is not None else trader.text.strip()
        risk = await self._call("risk", _fill(p["risk_template"], trader_json=proposal, reports=rep), "risk")
        calls = [*analysts, bull, bear, trader, risk]
        return TaLiteResult(x.symbol, x.t, calls, {r: parse_analyst(reports[r]) for r in ROLES},
                            ta_final(trader.text, risk.text))


def native_book(finals: dict[str, TaFinal], k: int = 20) -> dict[str, float]:
    """TA-lite's native book: equal weight over BUY names, at most k (by p_outperform, abstentions last, then the
    symbol hash of decide.select)."""
    from jevtrader.decide.select import tiebreak_key

    buys = [(s, f.p_outperform) for s, f in finals.items() if f.decision == "BUY"]
    buys.sort(key=lambda sp: (sp[1] is None, -(sp[1] or 0.0), tiebreak_key(sp[0])))
    top = [s for s, _ in buys[:k]]
    return {s: 1.0 / len(top) for s in top} if top else {}


# ------------------------------------------------------------------------------------------------ Mono-CoT
@dataclass
class MonoCoTResult:
    symbol: str
    t: pd.Timestamp
    call: Call
    parsed: Parsed

    @property
    def p_outperform(self) -> float | None:
        return self.parsed.value["p_outperform"] if self.parsed.valid else None


class MonoCoT:
    name = "Mono-CoT"
    prompt_id = "mono-cot-v1"

    def __init__(self, llm: GenLLM, prompt: dict | None = None):
        self.llm = llm
        self.p = prompt if prompt is not None else _bank_prompt(self.prompt_id)
        if self.p["decoding"]["model"] != llm.model or self.p["parser_version"] != PARSER_VERSION:
            raise ValueError("mono-cot-v1 pins another model or parser than this code")

    async def run(self, x: StockWeekInputs) -> MonoCoTResult:
        dec = self.p["decoding"]
        user = _fill(self.p["user_template"], fundamentals=x.fundamentals, technicals=x.technical, news=x.news,
                     macro_policy=x.macro_policy)
        c = await self.llm.complete("mono", self.p["system"], user, int(dec["max_tokens"]), float(dec["temperature"]))
        return MonoCoTResult(x.symbol, x.t, c, parse_mono_cot(c.text))


# ------------------------------------------------------------------------------------------------ tables
def result_frames(results: list[TaLiteResult | MonoCoTResult], bank_hash: str | None = None,
                  tag: str = "") -> dict[str, pd.DataFrame]:
    """`answers` (one row per stock-week and reader; `degraded_calls` apart from `invalid`), `requests` (one per
    call; the requests.parquet columns that apply, plus `degraded` and `attempts`) and `outputs` (raw text per call,
    for faithfulness and stating-vs-doing checks)."""
    ans, req, outs = [], [], []
    for r in results:
        reader = "TA-lite" if isinstance(r, TaLiteResult) else "Mono-CoT"
        prompt = TaLite.prompt_id if isinstance(r, TaLiteResult) else MonoCoT.prompt_id
        calls = r.calls if isinstance(r, TaLiteResult) else [r.call]
        if isinstance(r, TaLiteResult):
            f = r.final
            ans.append({"t": r.t, "symbol": r.symbol, "reader": reader, "decision": f.decision,
                        "p_outperform": f.p_outperform, "source": f.source, "invalid": f.invalid,
                        "degraded_calls": sum(c.degraded for c in calls),
                        **{f"stance_{k}": v.value for k, v in r.stances.items()},
                        **{f"invalid_{k}": v for k, v in r.invalid_counts.items()}})
        else:
            ans.append({"t": r.t, "symbol": r.symbol, "reader": reader, "decision": None,
                        "p_outperform": r.p_outperform, "source": "mono", "invalid": not r.parsed.valid,
                        "degraded_calls": int(r.call.degraded)})
        for c in calls:
            req.append({"t": r.t, "symbol": r.symbol, "reader": reader, "role": c.role, "request_key": c.request_key,
                        "backend": GenLLM.backend_id, "model_pinned": MODEL, "response_model": c.response_model,
                        "prompt_version": prompt, "parser_version": PARSER_VERSION,
                        "parser_sha256": PARSER_SHA256, "bank_hash": bank_hash, "input_tokens": c.input_tokens,
                        "cached_tokens": c.cached_tokens, "output_tokens": c.output_tokens, "usd": c.usd,
                        "latency_s": c.latency_s, "cached": c.cached, "finish_reason": c.finish_reason,
                        "degraded": c.degraded, "attempts": c.attempts, "max_tokens": c.max_tokens, "tag": tag})
            outs.append({"request_key": c.request_key, "role": c.role, "text": c.text})
    return {"answers": pd.DataFrame(ans), "requests": pd.DataFrame(req), "outputs": pd.DataFrame(outs)}

