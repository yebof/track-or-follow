"""Planning model for docs/design/design_v1.md: windows, LLM cost, wall-clock and power (planning estimates only).

These numbers size the experiment matrix of design v1. They are NOT paper results; the paper's cost table is
regenerated from measured `requests.parquet` rows (CLAUDE.md rule 2).

Inputs (all measured or archived, see provenance):
- per-request-type token counts of the question bank (docs/design/proposals/jev_native_prototype/token_counts.json,
  o200k proxy tokenizer, measured 2026-09-25);
- news volume, measured by `measure_news()` from data/processed/news_table.parquet (complete download) on the PIT
  top-100 universe: company items tagged with <= 3 tickers, created in [c_{t-1}, c_t), capped at 40 per stock-week;
  broad items (MARKET-tagged or > 3 tickers) matching the policy pre-filter lexicon of design_v1 section 5.5
  (cached in results/plan/news_volume.json; `--refresh` recomputes);
- prices: gpt-4.1-mini $0.40 in / $0.10 cached in / $1.60 out per 1M tokens (docs/pricing/openai_model_gpt-4.1-mini.html);
  Jev $0.042 per 1M input tokens, output free (docs/jev/pages/models);
- evaluation_first section 12 rows for generators whose profile does not depend on the bank (TA-lite, Mono-CoT, S2).
- trading calendar from data/raw/bars/etf_daily_2018_2026.parquet.

Run: .venv/bin/python experiments/plan/plan_v1.py  -> results/plan/plan_v1.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOK = json.loads((ROOT / "docs/design/proposals/jev_native_prototype/token_counts.json").read_text())

P_IN, P_CACHED, P_OUT = 0.40e-6, 0.10e-6, 1.60e-6  # gpt-4.1-mini
P_JEV = 0.042e-6
SYS_PROMPT_TOK = 60
POLICY_CAP = 60  # P1 items per week (design_v1 section 5.5)
# policy pre-filter lexicon (design_v1 section 5.5); matched case-insensitively on headline + summary
POLICY_LEXICON = (r"\b(?:tariffs?|fed|fomc|federal reserve|powell|congress|senate|house|white house|president|"
                  r"administration|regulators?|sec|ftc|doj|antitrust|sanctions?|export controls?|elections?|shutdown|"
                  r"debt ceiling|war|missiles?|ceasefire|subsid(?:y|ies)|tax(?:es)?|treasury|trade deal|opec|"
                  r"rate cuts?|rate hikes?|inflation)\b")


def _decision_fridays() -> pd.Series:
    d = pd.read_parquet(ROOT / "data/raw/bars/etf_daily_2018_2026.parquet", columns=["date"]).date
    s = pd.Series(sorted(pd.to_datetime(d.unique())))
    iso = s.dt.isocalendar()
    return s.groupby([iso.year.values, iso.week.values]).max().sort_values().reset_index(drop=True)


def measure_news(refresh: bool = False) -> dict:
    """Measured news volume per window (planning input; F18 of the feasibility review)."""
    cache = ROOT / "results/plan/news_volume.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    t = pd.read_parquet(ROOT / "data/processed/news_table.parquet")
    u = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    u["date"] = pd.to_datetime(u.date)
    fr = _decision_fridays()
    fr = fr[fr >= "2019-07-01"].reset_index(drop=True)
    cuts = pd.Series([pd.Timestamp(x.date()).tz_localize("America/New_York").replace(hour=16).tz_convert("UTC")
                      for x in fr])
    t["wk"] = np.searchsorted(cuts.values, t.created_at.values, side="right")  # created in [c_{k-1}, c_k)
    comp = t[(t.symbol != "MARKET") & (t.n_tags <= 3)]
    cnt = comp.groupby(["wk", "symbol"]).size()
    recon = sorted(u.date.unique())
    rows = []
    for k, dt in enumerate(fr):
        r = [x for x in recon if x <= dt]
        if not r:
            continue
        mem = u[u.date == r[-1]].symbol
        c = cnt.reindex(pd.MultiIndex.from_product([[k], mem])).fillna(0).clip(upper=40).values
        rows.append((dt, k, c.mean(), (c > 0).mean(), float(np.median(c))))
    df = pd.DataFrame(rows, columns=["d", "k", "mean", "cover", "median"]).set_index("d")
    broad = t[(t.symbol == "MARKET") | (t.n_tags > 3)].drop_duplicates("id")
    txt = (broad.headline.fillna("") + " " + broad.summary.fillna("")).str.lower()
    pol = broad[txt.str.contains(POLICY_LEXICON, regex=True)].groupby("wk").size()
    out = {}
    for name, a, b in (("W_clean", "2024-07-05", "2026-09-11"), ("W_pre_slice", "2023-04-07", "2024-05-31"),
                       ("full", "2019-07-05", "2026-09-11")):
        x = df.loc[a:b]
        out[name] = {"weeks": len(x), "items_per_stock_week": round(float(x["mean"].mean()), 3),
                     "share_with_item": round(float(x.cover.mean()), 3),
                     "median_items_per_stock_week": float(x["median"].median()),
                     "policy_prefilter_items_per_week": round(float(pol.reindex(x.k).fillna(0).mean()), 1)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


NEWS = measure_news("--refresh" in sys.argv)
ITEMS_CLEAN = NEWS["W_clean"]["items_per_stock_week"]
COVER_CLEAN = NEWS["W_clean"]["share_with_item"]
ITEMS_FULL = NEWS["full"]["items_per_stock_week"]
ITEMS_PRE = NEWS["W_pre_slice"]["items_per_stock_week"]
P1_RATE = min(POLICY_CAP, NEWS["W_clean"]["policy_prefilter_items_per_week"]) / 100


def t(name: str, field: str) -> float:
    return next(v[field] for k, v in TOK.items() if k.startswith(name))


def profile(items: float) -> dict:
    """Design-v1 bank per stock-week: request rates -> tokens (emulator), cacheable prefix, questions."""
    rates = {
        "N0": items, "N1": items, "N2": 1.6, "N3": 0.9, "T": 1.0, "FB": 17 / 52,
        "FA": 5 / 52,  # one 10-K per year, 5 exposure groups
        "P1": P1_RATE, "P2": 1 / 100, "M": 1 / 100,
    }
    n_q = {"N0": 6, "N1": 13, "N2": 5, "N3": 4, "T": 6, "FB": 5, "FA": 5, "P1": 18, "P2": 2, "M": 18}
    breakdown = {k: {"rate_per_stock_week": round(r, 3), "emu_in_per_request": t(k, "emu_in"),
                     "emu_out_per_request": t(k, "emu_out"), "jev_payload_per_request": t(k, "jev"),
                     "emu_in_per_stock_week": round(r * t(k, "emu_in"), 1)} for k, r in rates.items()}
    tin = sum(r * t(k, "emu_in") for k, r in rates.items())
    tout = sum(r * t(k, "emu_out") for k, r in rates.items())
    tjev = sum(r * t(k, "jev") for k, r in rates.items())
    cacheable = 0.0
    for k, r in rates.items():  # question-first fixed layout; >= 1024-token static prefix, 128-token blocks
        q = t(k, "questions") + SYS_PROMPT_TOK
        if q >= 1024:
            cacheable += r * (q // 128) * 128
    return {"requests": sum(rates.values()), "in": tin, "out": tout, "jev_payload": tjev, "cacheable": cacheable,
            "questions": sum(r * n_q[k] for k, r in rates.items()), "breakdown": breakdown}


def usd(p: dict, cached: bool = True) -> float:
    c = p["cacheable"] if cached else 0.0
    return (p["in"] - c) * P_IN + c * P_CACHED + p["out"] * P_OUT


def windows() -> dict:
    fridays = _decision_fridays()

    def n(a, b):
        return int(((fridays >= a) & (fridays <= b)).sum())

    return {"W_pre_slice(2023-04-07..2024-05-31)": n("2023-04-07", "2024-05-31"),
            "W_buffer(2024-06-07..2024-06-28)": n("2024-06-07", "2024-06-28"),
            "W_clean(2024-07-05..2026-09-11)": n("2024-07-05", "2026-09-11"),
            "W_release(2025-04-18..2026-09-11)": n("2025-04-18", "2026-09-11"),
            "W_pre(2020-01-03..2024-05-31)": n("2020-01-03", "2024-05-31")}


def power(T: int) -> dict:
    z2 = 1.96 + 0.8416  # two-sided 5%, power 80%
    z_holm = 2.128 + 0.8416  # one-sided at alpha/3 (Holm worst case), power 80%
    out = {}
    # C1: paired binary attack success, n pairs, discordance pi_d, week-cluster design effect 2
    for n_pairs in (1400, 2800):
        for pi_d in (0.2, 0.4):
            se = math.sqrt(2 * pi_d / n_pairs)
            out[f"C1_MDE_n{n_pairs}_pid{pi_d}"] = round(z_holm * se, 3)
    # C2: weekly mean of item-level Spearman; ~260 kept items per week (6.13 x 100 x 0.84 x ~0.5 kept)
    items_wk = ITEMS_CLEAN * 100 * 0.5  # about half of the items are kept (about_focal and new_development)
    se_wk = 1 / math.sqrt(items_wk - 1)
    for between in (0.0, 0.05):
        se = math.sqrt(se_wk**2 + between**2) / math.sqrt(T)
        out[f"C2_MDE_between_sd{between}"] = round(z_holm * se, 4)
    # C3: paired Delta-IC, sigma_IC 0.12 (U100 idiosyncratic signal), correlation rho between the two systems
    for rho in (0.5, 0.8):
        sd = 0.12 * math.sqrt(2 * (1 - rho))
        out[f"C3_MDE_rho{rho}"] = round(z_holm * sd / math.sqrt(T), 4)
    out["IC_MDE_single_sigma0.12"] = round(z2 * 0.12 / math.sqrt(T), 4)
    out["IR_MDE_approx"] = round(z2 / math.sqrt(T / 52), 2)
    return out


def main() -> None:
    pc, pf, pp = profile(ITEMS_CLEAN), profile(ITEMS_FULL), profile(ITEMS_PRE)
    sw_clean = 115 * 100
    s50, s30 = 115 * 50, 115 * 30
    sw_c, sw_f = usd(pc), usd(pf)
    n1_cached = (t("N1", "questions") + SYS_PROMPT_TOK) // 128 * 128
    n1_item = (t("N1", "emu_in") - n1_cached) * P_IN + n1_cached * P_CACHED + t("N1", "emu_out") * P_OUT
    n0_item = t("N0", "emu_in") * P_IN + t("N0", "emu_out") * P_OUT
    n3_req = t("N3", "emu_in") * P_IN + t("N3", "emu_out") * P_OUT
    n2_req = t("N2", "emu_in") * P_IN + t("N2", "emu_out") * P_OUT
    holistic_req = 1900 * P_IN + 10 * P_OUT
    sentiment_req = 850 * P_IN + 40 * P_OUT
    talite_sw = 52.56 / s50  # evaluation_first section 12, outputs capped per role
    typed_attack = n0_item + n1_item + n2_req + n3_req  # attacked item re-asked through N0..N3
    nodef_attack = n1_item + 1500 * P_IN + 20 * P_OUT  # N1 on raw text + N3 with raw headlines
    verb_n1_item = t("N1", "emu_in") * P_IN + 13 * 5 * 9 * P_OUT  # ~9 output tokens per option (pilot)

    rows = {
        # --- P0
        "DEV: 2 x 500 W_dev stock-weeks (bank + prompt shake-down)": 1000 * usd(pf),
        "W_pre slice: S30 x 60 wk blinded + unblinded; S30 W_clean unblinded": 2 * 30 * 60 * 0.95 * usd(pp) + s30 * 0.95 * sw_c,
        "MAIN: W_clean 115 wk x U100 + 2 pseudo-forward weeks": (sw_clean + 200) * sw_c,
        "Mono-S1 holistic (one request, same evidence), U100": sw_clean * holistic_req,
        "Headline sentiment (S1 Choice per headline), U100": sw_clean * sentiment_req,
        "TA-lite NL multi-agent trader, S50 (capped outputs)": 52.56,
        "Typed pipeline without defenses (clean), S50": s50 * (ITEMS_CLEAN * n1_item + 0.9 * (1500 * P_IN)),
        "C1 static red-team: promotion 400 sw x A1-A7 x {typed, nodef, holistic, sentiment}":
            400 * 7 * (typed_attack + nodef_attack + holistic_req + sentiment_req) * 1.15,  # A7 extra weeks
        "C1 bound check (rank > K+J): 100 sw x A1-A7": 100 * 7 * (typed_attack + holistic_req + sentiment_req),
        "C1 demotion: 200 held sw x {A2-neg, A6-neg, A8}": 200 * 3 * (typed_attack + holistic_req + sentiment_req),
        "C1 TA-lite under attack: 100 S50 sw x 4 families": 100 * 4 * talite_sw,
        "C1 adaptive black-box: 150 items x 64 queries, typed + holistic, proposer": 150 * 64 * (
            n0_item + n1_item + holistic_req + 500 * P_IN + 60 * P_OUT),
        "Faithfulness T2/T3 (+ null arms), TA-lite T4, Mono-S1 deletion": 8.0,
        "Emulator fidelity V1 (one question per call) + V2 re-ask floor": 5.0,
        "Probes D1 (monthly window rule + all weeks) / D2 re-identification": 3.0,
        "Memory ablations: M-epi off (N3 re-ask), L0 off (T re-ask)": sw_clean * (0.84 * n3_req + t("T", "emu_in") * P_IN),
        "Live + shadow books (5 weeks)": 8.0,
        # --- P1
        "P1: S2 (gpt-4.1-mini CoT) + V verification, all S50 stock-weeks": s50 * (4.5e3 * P_IN + 450 * P_OUT + t("V", "emu_in") * P_IN),
        "P1: verbalized backend on a 6,000-item N1 sample": 6000 * verb_n1_item,
        "P1: Mono-CoT single LLM, S50": 8.8,
        "P1: X-Web prevalence, 10k pre-cutoff items (N0 + guard Nouls)": 10_000 * (n0_item + 500 * P_IN),
    }
    planned = sum(rows.values())
    rerun = (sw_clean + 200) * usd(pc, cached=False)
    req_main = sw_clean * pc["requests"]
    out = {
        "news_volume_measured": NEWS,
        "windows_decisions": windows(),
        "token_profile_by_request_clean": pc["breakdown"],
        "profile_per_stock_week_clean": {k: round(v, 1) for k, v in pc.items() if k != "breakdown"} | {
            "usd_cached": round(sw_c, 5), "usd_uncached": round(usd(pc, False), 5),
            "jev_usd_payload": round(pc["jev_payload"] * P_JEV, 6)},
        "cost_rows_usd": {k: round(v, 1) for k, v in rows.items()},
        "total_planned_usd": round(planned, 0),
        "total_x1.3_usd": round(1.3 * planned, 0),
        "plus_one_uncached_MAIN_rerun_usd": round(1.3 * planned + rerun, 0),
        "jev_if_key": {
            "MAIN_requests": round(req_main),
            "MAIN_usd_payload": round(sw_clean * pc["jev_payload"] * P_JEV, 1),
            "MAIN_usd_with_280tok_per_question_overhead": round(sw_clean * (pc["jev_payload"] + 280 * pc["questions"]) * P_JEV, 1),
            "MAIN_hours_1200rpm": round(req_main / 1200 / 60, 1), "MAIN_hours_300rpm": round(req_main / 300 / 60, 1),
        },
        "emulator_MAIN_hours_conc64_lat1.5s": round(req_main / (64 / 1.5) / 3600, 1),
        "power": power(115),
        "power_W_release": power(73),
    }
    dst = ROOT / "results/plan/plan_v1.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
