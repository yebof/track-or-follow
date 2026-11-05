# qu2026clqt: CLQT: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of LLM Portfolio-Management Agents

## Verified metadata
- Title: CLQT: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of LLM Portfolio-Management Agents [source: arxiv (abs page citation_title)]
- Authors: Bo Qu, Mingguang Chen [source: arxiv]
- Venue: arXiv preprint [source: arxiv].
  - The arXiv comment reads "53 pages, 14 figures, 10 tables" and lists v3 changes; no journal-ref or DOI is given [source: arxiv].
  - Crossref bibliographic search (2026-09-25) returned no matching record; the top 3 hits are unrelated papers.
  - A re-query of the arXiv export API (2026-09-25) returns v3 with no journal_ref or DOI [source: arxiv API].
  - DBLP timed out or returned a bot-challenge page, and OpenAlex and Semantic Scholar returned HTTP 429 on 2026-09-25. No peer-reviewed venue is claimed.
- Year: 2026 (v1 2026-06-29; v2 2026-08-02; v3 2026-09-12) [source: arxiv]
- DOI / arXiv: none / arXiv:2606.29771 (v3). ACM classes: I.2.11; I.2.1; J.1 [source: arxiv]
- Metadata files:
  - literature/meta/qu2026clqt.arxiv_abs.html (abs page with citation_* meta tags, saved when the export API first returned HTTP 429);
  - literature/meta/qu2026clqt.arxiv_api.xml (arXiv export API record, fetched 2026-09-25 during verification);
  - literature/meta/qu2026clqt.crossref_search.json.

## PDF
- Path: literature/pdf/qu2026clqt.pdf (arXiv v3, 12 Sep 2026; downloaded from arxiv.org/pdf/2606.29771v3)
- Text: literature/text/qu2026clqt.txt ; pages: 53

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - The authors argue that ranking LLM trading agents by returns over one fixed window certifies neither process soundness nor edge durability. One period's return is dominated by the realized market path, and apparent alpha can dissolve once look-ahead bias and costs are controlled (p.1).
  - Table 1 (p.4) maps twelve prior closed-loop trading benchmarks on seven features: point-in-time data, cost modeling, portfolio-level evaluation, multi-asset support, cross-round memory, tool-orchestration evaluation and a live track. No prior benchmark covers all seven (the best cover 5/7); CLQT credits itself 7/7. Its multi-asset mark reflects substrate support only, since the campaign trades US equities alone (p.4).
  - Their stance is "diagnosis before ranking" (p.2). A model should be called superior only if it dominates the capability axes consistently across sub-periods (p.2).
- Method / architecture:
  - The loop has five stages (p.8): Intelligence Acquisition (IA), Multi-Source Signal Synthesis (MSS), Mandate-Aware Portfolio Construction (MAPC), Cost-Optimal Order Execution (COE), and Attribution & Lesson Extraction (ALE).
  - Six agents run the loop (p.9): Research, Analyst, PM, Risk Officer, Execution and Reflection. All quantitative analysis goes through 19 typed MCP tools (p.8). Every call is schema-checked, hashed and timed (p.7-9).
  - Two scaffolds run the same model (p.10):
    - "structured": the six-agent committee with role-restricted tools and a Risk Officer gate.
    - "autonomous": one orchestrator with all tools, a mandate and an output schema.
  - The SCOUT stage ranks candidates by a composite z-score, 0.50·momentum (12-1) + 0.30·earnings surprise + 0.20·60-day correlation with SPY (p.10-11). The weights are "an ablation default, not a derived optimum" (p.11).
  - Costs (p.11): four impact models (fixed bps, square-root, Almgren–Chriss, structural) are selectable per cost tier. The LOW tier's commission and spread legs are validated against live broker paper fills; its impact leg is parameterized, not fitted.
  - Strategy consistency (p.12): C_t = 0.25·(1−style drift) + 0.25·turnover consistency + 0.25·signal adherence + 0.25·mandate alignment, with a drift warning when C_t < 0.7.
  - Memory has three tiers (p.12-13):
    - working memory: the current observation plus the last N = 3 rounds;
    - episodic memory, decaying at α = 0.95 per round;
    - semantic memory, at α = 0.98, consolidated every 12 rounds.
  - MAPC has four layers (p.13): an optimizer proposal, a budget-neutral agent overlay, constrained projection, and a Risk Officer gate. It records pre- and post-projection deltas separately.
  - TimeGate enforces timestamp(d) ≤ t_as_of for all data, including memory retrieval (p.15). Each DecisionRound is sealed into a recompute-verifiable hash chain. The authors scope it as integrity of a provided store, not non-repudiation (p.15).
  - The five-axis scorecard APM-CS (p.18-19) is computed from audit fields:
    - D1 Coherence = 0.5·signal–action agreement + 0.5·held-out LLM judge;
    - D2 Acuity = Spearman(proposed Δ, informative composite) − Spearman(proposed Δ, noise composite);
    - D3 Composure;
    - D4 Discipline;
    - D5 Reliability = completion·(1−parse-fail)·(1−truncation).
  - The judge is held out of the scored cohort: minimax-m3 for backtest, GLM-5.2 for live. The two judges have Spearman ρ = 0.65 and are within one point on 97% of shared rounds (p.20).
- Data, universe, evaluation period, baselines:
  - Universe (p.16): derived from the S&P 100, point-in-time membership. There are about 14 equal-weighted core holdings plus about 58 candidates, 72 symbols in total.
  - Data sources (Table 2, p.10):
    - prices: Yahoo Finance (live: Yahoo Finance or a broker snapshot);
    - fundamentals: Finnhub in backtest; Finnhub / SEC EDGAR live;
    - news: Alpha Vantage / Polygon / RSS / NewsAPI;
    - macro: 12 FRED series.
  - Backtest (p.22):
    - 26 bi-weekly rounds from 2025-06-16 to 2026-06-12, $1M NAV, LOW cost tier;
    - context capped at 32k and output budget 8192;
    - five seeds per reliable configuration; qwen3·structured has a single showcase seed.
  - Backtest LLMs (p.22): qwen3-235b, gemini-2.5-flash-lite, deepseek-chat-v3.1, gpt-5-mini and claude-haiku-4.5. The authors state each training cutoff precedes the start, with qwen3 (Jun'25) inside a "≤1-month grace".
  - Live track (p.33):
    - 18 valid NYSE days (2026-06-15 → 2026-07-10), daily rebalance, on a commercial broker's paper account;
    - six newer models (minimax-m3, deepseek-v4-pro, qwen3.7-plus, gpt-5.4-mini, gemini-3.5-flash, claude-sonnet-4.6) in both modes, i.e., 12 virtual $1M ledgers;
    - an end-of-day flatten.
  - Costs are modeled (Table A1, p.47):
    - LOW tier (default): spread 2.0, commission 0.5, slippage 1.0 bps, borrow 30 ann. bps, square-root impact;
    - HIGH tier: 5.0/2.0/5.0/100 with Almgren–Chriss impact.
  - Baselines (p.17): SPY, IEF, 60/40, EW-Universe, buy-and-hold, equal-weight (rebalanced), risk-parity and 12-1 momentum.
  - Sharpe and Sortino use a zero risk-free rate (p.18).
  - Ablation grid (p.22): anchor gemini-2.5-flash-lite·structured; six module-off variants, two cost tiers, and four cluster knockouts (p.29).
- Key results (copy numbers EXACTLY, with page):
  - Table 5 (p.23), backtest, means over seeds:

    | Model · mode | Sharpe | CumRet | Beat/8 | D1 | D5 | APM-CS |
    |---|---|---|---|---|---|---|
    | deepseek-chat-v3.1 · structured | 2.99 | +28.6% | 6.8 | 0.45 | 0.99 | 72.9 |
    | claude-haiku-4.5 · structured | 3.21 | +23.4% | 6.0 | 0.23 | 1.00 | 71.0 |
    | gemini-2.5-flash-lite · structured | 2.31 | +24.8% | 6.0 | 0.38 | 0.61 | 47.4 |
    | qwen3-235b · structured ★ | 9.82 | +13.1% | 8.0 | 0.33 | 0.19 | 30.7 |

  - The qwen3·structured Sharpe of 9.82 comes from only 5/26 completed rounds. The authors call it "a survivorship artifact of a self-selected window, not a strategy" (p.23-24).
  - Repeated-run noise band: ±0.42 Sharpe (p.27).
  - No module-off variant is separable on Sharpe: every run range straddles or sits inside that band (p.27-28), even though two means exceed ±0.42. Examples (Table 6, p.27): no-reflection +0.09 [−0.16, +0.39]; no-memory +0.52 [+0.24, +0.72]; no-world-macro +0.66 [+0.33, +1.06]. All module-off means are positive, which the authors call "mechanistically implausible" (p.28).
  - The HIGH cost tier (≈527 bps/round, Sharpe −1.06) is judged mis-calibrated (p.27-28).
  - A "bare-workflow" agent with no information, learning or quant modules is not separable from the full agent on returns: ΔSharpe +0.26, range [−0.40, +0.66] (Table 7, p.29).
  - Table 8 (p.30): the full agent has Sharpe 2.26, MaxDD −4.1%, Beat/8 5.7 and turnover 0.15. SPY has Sharpe 2.29 and MaxDD −8.9%.
  - Coherence gap: agreement vs judge is +0.30 in backtest (0.58 vs 0.28) and +0.23 live (0.63 vs 0.40) (p.34). Under a full judge swap the backtest gap is +0.27 (p.36).
  - Live Table 10 (p.34): claude-sonnet-4.6·auto leads with APM-CS 83.6, and every model scores higher in autonomous than structured mode (p.33).
  - Live holds (p.36-37): gemini-3.5-flash·auto held 9/18 rounds and minimax-m3·auto 8/18. All 17 hold rounds precede the round-14 scaffold remediation.
  - Regime (p.31): the window is "benign": SPY +27%, max DD −8.9%, VIX median 17.
- Limitations (as stated by authors, with page) and limitations we observe (label these "our observation"):
  - Authors, statistical power and scope (p.41):
    - The returns axis is underpowered for module attribution.
    - The HIGH tier is mis-calibrated.
    - Acuity and Discipline are mode-confounded.
    - The ablation uses one anchor model.
  - Authors, construct validity (p.41-42): the axes are sensitive to construct definitions. The informative composite's alpha is "Stated, not proven", and the ±0.42 band is assumed to transfer across variants.
  - Authors, regime, costs and judges (p.42):
    - There is a single, predominantly rising regime, and backtest costs are modeled, not executed.
    - Live returns are indicative only (daily flatten).
    - The cohort is small.
    - A single author-rater anchors the judge.
    - Canary and future-shift probes were not exercised (p.15, p.42-43).
  - Authors (p.21): bootstrapped CIs and Kendall-τ rank stability are deferred to a later pass.
  - Our observation: nearly every quantitative claim is a point estimate or run range over at most five seeds in one bull window of 26 rounds. The paper's own noise band makes most between-configuration return differences non-significant.
  - Our observation: backtest model cutoffs (p.22) are the authors' statements and are not independently verified in the paper. The live cohort uses different model versions than the backtest, so cross-track "consistency" is not same-model transfer. The authors acknowledge this (p.35).
  - Our observation: the broker is not named ("a commercial broker’s paper-trading API", p.16), and its integration is "available on request" (p.16). This limits reproducibility of the live track.
  - Our observation: this is a 53-page arXiv preprint by two authors (one listed as "Independent Researcher", p.1) with no peer review confirmed.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "the capability leader is not the Sharpe leader; that agents’ allocations systematically fail to follow their own stated analysis — a stating-versus-doing gap stable across both tracks (+0.30 backtest, +0.23 live)" (p.1)
- "Net of realistic costs, agents clear defensive baselines but do not cleanly beat the index." (p.1)
- "Module importance is not rankable by Sharpe — even across repeated runs (RQ5)." (p.27)
- "Everything above is measured in one benign, rising market: SPY +27% over the campaign, a −8.9% drawdown, calm VIX (median 17; above 25 on just 7% of days)." (p.31)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- Evaluation design: this is the most directly reusable evaluation template in this batch.
  - Its eight-member passive baseline suite (p.17) is a ready-made baseline list for our paper: SPY, IEF, 60/40, EW-universe, buy-and-hold, equal-weight, risk-parity and 12-1 momentum.
  - The LOW cost tier (spread 2.0 / commission 0.5 / slippage 1.0 bps, square-root impact; p.47) is a concrete, published reference point for our cost model. We should cite it, not copy it silently.
  - The ±0.42-Sharpe repeated-run noise band (p.27) argues that we must report run-to-run variation, not single-run ΔSharpe, when ablating agents (fundamentals/news/macro/politics) or memory layers.
- Point-in-time and paper trading: TimeGate (p.15) and per-model cutoff eligibility (p.22) mirror our CLAUDE.md rule 3. The broker paper-trading live track (p.16, p.33) parallels our Alpaca paper forward test.
  - We differ on long-only / cash-only: CLQT's default mandate is long-only with leverage ≤ 1 (p.12). We enforce cash-only and long-only as a pre-trade check.
- Explainability and CoT: the "stating-versus-doing" coherence gap (p.34) is a warning for any claim that our explanations are faithful.
  - We should measure agreement between the typed per-domain judgments and the final portfolio action, analogous to their signal–action agreement (p.18).
  - Per docs/jev/docs_introduction.txt, Jev returns "typed values and probability distributions" with "No text generation, no parsing". Our decision rationale is therefore a vector of typed judgments combined by deterministic code, not generated prose. This is a structural difference we can test rather than assume.
- Reliability: CLQT's D5 axis and parse-failure findings (p.24-26) document schema failures of generative LLMs that produce structured decisions. We should report analogous telemetry (API errors, timeouts, abstentions) for Jev so that any reliability advantage is measured, not asserted.
- How we differ:
  - CLQT is a benchmark for generative-LLM agents (committee vs autonomous scaffolds, 19 MCP tools). Our contribution is a trading system whose core decision engine is a typed decision model.
  - Their bi-weekly S&P-100-derived 72-name setting (p.16, p.22) is close to our mid/low-frequency US-equity scope. If comparable, it could serve as an external reference setting.

## BibTeX
```bibtex
@article{qu2026clqt,
  author  = {Qu, Bo and Chen, Mingguang},
  title   = {{CLQT}: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of {LLM} Portfolio-Management Agents},
  journal = {arXiv preprint arXiv:2606.29771},
  year    = {2026},
  eprint  = {2606.29771},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2606.29771}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check pass).
- What was checked:
  - Metadata (title, authors, dates, comment, ACM classes, missing journal-ref/DOI) against literature/meta/qu2026clqt.arxiv_abs.html. A fresh arXiv export API record was saved as literature/meta/qu2026clqt.arxiv_api.xml; it also has no journal_ref or DOI. Crossref (re-queried) has no record. DBLP (dblp.org timed out; the dblp.uni-trier.de mirror served a bot-challenge page) and Semantic Scholar/OpenAlex (HTTP 429) could not be used. The venue stays "arXiv preprint".
  - The PDF (arXiv v3, 53 pages; the pdfinfo title matches) is the paper named in the metadata.
  - Every number, date, model name, baseline, cost parameter and table value was checked against the PDF text. This covers Tables 1, 2, 5, 6, 7, 8, 10 and A1, pp. 1-47.
  - All four verbatim quotes and the short in-line quotes match the text, apart from line-wrapping. The "survivorship artifact" quote runs across pp. 23-24.
  - The Jev statement matches docs/jev/docs_introduction.txt.
  - The BibTeX matches the arXiv record.
- What was corrected:
  - The seven-feature claim is now stated as what Table 1 shows: twelve surveyed benchmarks, the best at 5/7, and the caveat that CLQT's multi-asset mark covers substrate support only.
  - The Table 2 data sources were fixed: backtest fundamentals come from Finnhub, and SEC EDGAR is a live-track source only.
  - The earlier text said "every module-off ΔSharpe falls inside the ±0.42 band". This was wrong: the no-memory (+0.52) and no-world-macro (+0.66) means exceed it. The paper's actual claim is that every run range straddles or sits inside the band.
  - The metadata-files list now includes the arXiv API record.
