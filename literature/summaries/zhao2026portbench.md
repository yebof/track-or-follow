# zhao2026portbench: PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management

## Verified metadata
- Title: PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management [source: arxiv, datacite]
- Authors: Yuxuan Zhao, Sijia Chen, Ningxin Su [source: arxiv, datacite]. The PDF lists the Yantai Research Institute of Harbin Engineering University and HKUST (Guangzhou) (p.1).
- Venue: arXiv preprint, not peer reviewed [source: arxiv]. The PDF stamp gives cs.AI (p.1). The DataCite record lists no related published DOI [source: datacite], and a Crossref bibliographic title query on 2026-09-25 returned no matching record. DBLP could not be queried (bot-check page / HTTP 429), so the absence of a DBLP record is unverified.
- Year: 2026 (v1 submitted 2026-05-27; v5, the version we read, submitted 2026-09-12) [source: datacite, arxiv]
- DOI / arXiv: 10.48550/arXiv.2605.27887 (arXiv DataCite DOI) / arXiv:2605.27887 (v5)
- Metadata files: literature/meta/zhao2026portbench.arxiv.html (arXiv abstract page), literature/meta/zhao2026portbench.datacite.json
- Note: on 2026-09-25 the arXiv export API returned HTTP 406/429 for this ID, so the arXiv record is the abstract page. DBLP returned a bot-check page, and the OpenAlex budget was used up.

## PDF
- Path: literature/pdf/zhao2026portbench.pdf (arXiv 2605.27887v5). The title on p.1 and in the PDF info matches.
- Text: literature/text/zhao2026portbench.txt ; pages: 36

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Existing financial benchmarks evaluate portfolio management (PM) poorly for two reasons (p.1):
    - They are often single-asset-class, or treat assets in isolation, and ignore cross-asset correlation.
    - They do not evaluate the full sequential PM pipeline, from market interpretation to risk monitoring, including error cascades, stress regimes and investor risk profiles.
  - LLM multi-agent PM systems are evaluated with "proprietary backtests that differ in data period, stock pools, and metrics" (p.1).
- Method / architecture:
  - **Market base dataset (p.2).** 183 instruments across six asset classes from January 2015 to December 2025: equities (126 tickers), commodities (16), bonds (15), crypto (12), real estate (10) and cash equivalents (4). Each has daily prices, macro indicators and time-aligned news.
  - **Static QA layer (p.2; p.4).** 6,269 QA pairs over seven templates T1–T7 at complexity levels L1–L4.
  - **Dynamic five-stage pipeline (p.2-3).** It is replayed point-in-time.
    - The LLM produces S1 Market Interpretation, S2 Signal Generation and S3 Weight Optimization.
    - A deterministic sandbox computes S4 Execution Simulation (turnover vs oracle) and S5 Risk Monitoring (VaR, drawdown, weight drift vs oracle).
  - **Metrics (p.3).**
    - CEPS = clip(mean stage score − λ·Σ drops, 0, 1), with λ = 0.1.
    - The S3 score mixes weight accuracy against a signal-constrained max-Sharpe oracle with a correlation score (intra-class concentration penalty, inter-class hedging credit), α = 0.5.
  - **Other features (p.3).**
    - Paired stage interventions: replace one stage with ground truth and rerun the downstream stages.
    - A stress drawdown gate over three windows.
    - Three investor profiles scored with PAS and AdaptScore.
  - **Real-time evaluation mode (p.6).** Uses point-in-time inputs to mitigate pretraining contamination.
- Data, universe, evaluation period, baselines (for trading papers: tickers/universe size, dates, frequency, costs modeled?, LLMs used):
  - **Universe (p.2).** 183 multi-asset instruments, 126 of them equity tickers.
  - **Periods.**
    - Normal period: a 2024 bull-market window (p.4).
    - Stress windows: 2015 China Shock (Aug. 2015 – Feb. 2016), 2020 COVID Crash (Feb. 2020 – May 2020), 2022 Crypto Collapse (May 2022 – Dec. 2022) (Table 6, p.20).
    - Real-time: 12 trading days, 2026-07-16 to 07-31, with daily rebalancing (p.6; p.21).
  - **Frequency and costs (p.4; p.21).** Monthly rebalancing. "Target weights incur 10 bps slippage and 5 bps commission per traded value" (p.21).
  - **LLMs (p.4).** Ten: Kimi-K2.6, Qwen3.7-Max, Qwen3.6-Plus, Qwen3.6-35B-A3B, GLM-5.1, Doubao-Seed-2.0-Lite/Pro, Hunyuan3-Preview and DeepSeek-V4-Flash/Pro. Temperature 0, max 8192 output tokens, thinking enabled.
  - **Baselines (p.4).** Equal-Weight (1/n), 60/40, Risk Parity, Covariance Risk Parity, Minimum Variance and Black–Litterman.
- Key results (copy numbers EXACTLY, with page):
  - **Headline (p.1, p.6).** "only 32.5% of 120 evaluations beat equal weighting on Sharpe across four market periods" (p.1). LLMs beat EqW "in only 39 cases (32.5%)" (p.6).
  - **Beat rates by period (p.6).** "77% in the 2020 crash-and-rebound window, 43% in 2015–16, 10% in the 2024 bull market, and 0% in the prolonged 2022 bear market".
  - **Static QA (Table 1, p.4).** The highest mean is 0.819 (DS-V4-Flash, Qwen3.7-Max). "no model exceeds 0.520 on return prediction (T1)" (p.4).
  - **Pipeline (p.4).** "Mean S3 is .301, combining a weight-accuracy component of only .101 with a correlation component of .501". Models realize "only 12.8% of oracle turnover".
  - **Normal period vs EqW (p.5).** "Only 3 of the 30 model–profile cells outperform EqW on Sharpe". Pooled CEPS correlates with Sharpe at "ρ = −.17".
  - **QA vs pipeline ranks (p.6).** They correlate at "ρ = −.49".
  - **Normal-period baselines (Table 2, p.5).**

    | Baseline | Sharpe | Return % | MaxDD % |
    |---|---|---|---|
    | EqW | 0.863 | 11.54 | −4.48 |
    | 60/40 | 0.841 | 11.95 | −4.27 |
    | MinVar | −0.043 | 4.11 | −1.28 |

    - Best LLM Sharpe (conservative): Qwen3.6-35B-A3B at 1.068.
  - **Stress gate (p.5).** "seven of ten models fail at least one conservative-profile scenario, mostly in the 2022 window".
  - **Active tilts (p.6-7).** Distance from EqW correlates with volatility (ρ = 0.68) and drawdown (ρ = 0.69) "but not with better ∆Sharpe (ρ = −0.08, 95% CI [−0.47, 0.31])". Model maximum weights are "below 0.05". Mean HHI ".0207" means "about 48 effective positions" (p.7).
  - **Paired intervention (p.6).** Replacing S1 with ground truth raises S2 by "+.626", but its mean effect on S3 is "only +.017".
- Limitations:
  - **Stated by the authors (p.9):**
    - (1) Deterministic transaction costs ignore microstructure, liquidity and market impact.
    - (2) Primary analysis is monthly only, apart from the short real-time window.
    - (3) S1–S3 come from isolated LLM prompts "without persistent memory, external tool access, or multi-agent coordination".
  - **Ethics statement (p.9):** passing the stress gate is not certification for real-world use.
  - **Our observation:**
    - The normal-period test is a single 2024 bull window.
    - The universe is multi-asset, not a US-equity-only long-only book.
    - The oracle is a concentrated max-Sharpe allocation, so CEPS rewards matching it even when realized risk is higher (p.5).
    - No decision model (Jev) and no GPT, Claude or Gemini model is among the ten LLMs evaluated (p.4).
    - Historical windows (2015–2022) likely lie inside the LLMs' pretraining periods; the authors raise this contamination risk themselves (p.6).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "only 32.5% of 120 evaluations beat equal weighting on Sharpe across four market periods." (p.1)
- "LLM portfolios therefore show no stable advantage over EqW across market regimes." (p.6)
- "Their tilts are large enough to raise risk but not accurate enough to improve Sharpe." (p.7)
- "The model produces S1–S3 through isolated LLM prompts without persistent memory, external tool access, or multi-agent coordination" (p.9)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Baselines.** It is strong recent evidence that equal-weight (1/n) is a hard baseline for LLM-driven allocation: only 3/30 normal-period cells beat it (p.5). Our evaluation must include EqW over our universe, plus simple risk-based baselines such as MinVar and risk parity (p.4). We must report whether Jev-driven portfolios beat EqW across multiple regimes, not a single window (p.6).
- **Evaluation choices.** Their protocol supports several of our choices:
  - point-in-time replay;
  - explicit costs (10 bps slippage + 5 bps commission, p.21);
  - stress windows;
  - a real-time, post-cutoff window to counter pretraining contamination (p.6).
  Our forward paper-trading period plays the role of their real-time evaluation.
- **Design lesson.** Stage-wise accuracy (good interpretation and signals) does not translate into good allocation. Diffuse tilts raise risk without improving Sharpe (p.6-7). This motivates an explicit, principled aggregation layer in our system, mapping calibrated agent probabilities to position sizes with risk control, rather than asking an LLM for weights directly.
- **Optional benchmark.** PortBench could be an additional evaluation setting for our portfolio-level decisions. But it is multi-asset, and its pipeline lacks multi-agent coordination and memory by design (p.9). We would need to adapt it to a US-equity long-only setting.
- **How we differ.** They benchmark single-prompt generative LLMs producing sentiment, signals and weights. We use Jev's typed probabilistic decisions from specialized agents with shared memory and deterministic aggregation and risk control.

## BibTeX
```bibtex
@misc{zhao2026portbench,
  author        = {Zhao, Yuxuan and Chen, Sijia and Su, Ningxin},
  title         = {{PortBench}: A Correlation-Aware, Full-Pipeline Benchmark for {LLM}-Driven Portfolio Management},
  year          = {2026},
  eprint        = {2605.27887},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2605.27887},
  howpublished  = {arXiv preprint arXiv:2605.27887},
  url           = {https://arxiv.org/abs/2605.27887}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- Metadata checked:
  - Title, authors, arXiv ID, version history (v1 2026-05-27 to v5 2026-09-12), categories (cs.AI, q-fin.PM) and the comment against the saved arXiv abstract page and DataCite record.
  - DataCite was re-queried live and confirms them (version 5, no related published DOI).
  - The PDF stamp reads arXiv:2605.27887v5 [cs.AI] 12 Sep 2026, the PDF info title matches, and the affiliations match p.1.
  - A Crossref title query found no record. DBLP and Semantic Scholar/OpenAlex could not be queried (bot-check / rate limits).
- Content checked against the PDF text (36 pages):
  - Dataset and QA: universe counts (183; 126/16/15/12/10/4); 6,269 QA pairs.
  - Pipeline: CEPS formula with λ = 0.1 and α = 0.5; the LLM list; baselines; temperature/tokens settings.
  - Periods and costs: stress-window dates (Table 6, p.20); real-time window and daily rebalancing (p.6, p.21); the cost quote (p.21).
  - Results: 39/120 = 32.5%; beat rates 77/43/10/0%; Table 1 means and T1 cap; S3 .301/.101/.501; 12.8% turnover; 3/30 cells beating EqW, cross-checked against Table 2; ρ = −.17 and −.49; Table 2 baseline rows and the 1.068 maximum; the stress gate; tilt correlations and CI; HHI .0207 ≈ 48 positions; intervention +.626/+.017.
  - Limitations (p.9) and all four verbatim quotes (p.1, p.6, p.7, p.9). Each matches exactly.
- Corrections: the venue line asserted "No DBLP … record exists", but DBLP was never successfully queried. This is now stated as unverified. No content errors found.
