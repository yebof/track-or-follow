# li2025time: Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking

## Verified metadata
- Title: Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking [source: crossref; arxiv]
- Authors: Changlun Li, Yao Shi, Chen Wang, Qiqi Duan, Runke Ruan, Weijie Huang, Haonan Long, Lijun Huang, Nan Tang, Yuyu Luo [source: crossref; arxiv]. Crossref capitalizes "Yao SHI" and "Runke RUAN"; the arXiv capitalization is used here.
- Venue: Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 184676-184704. Publisher: Neural Information Processing Systems Foundation, Inc. Crossref event: "39th Conference on Neural Information Processing Systems (NeurIPS 2025)", San Diego, California, USA and Mexico City, Mexico [source: crossref]. The Datasets and Benchmarks Track is stated in the arXiv comment and on the PDF footer (p.1) [source: arxiv].
- Year: 2025 [source: crossref]
- DOI / arXiv: 10.52202/085713-5554 / arXiv:2505.11065 (v2, 2025-10-14)
- Metadata files: literature/meta/li2025time.crossref.json, literature/meta/li2025time.arxiv_abs.html. The arXiv export API returned HTTP 429, so the abs page with citation_* meta tags was saved.

## PDF
- Path: literature/pdf/li2025time.pdf. This is arXiv v2, whose footer reads "39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmarks" (p.1).
- Text: literature/text/li2025time.txt ; pages: 22

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Historical back-testing of LLM trading strategies lets models "time travel". They can exploit future information embedded in their pretraining corpora, which leaks information and inflates performance (p.1).
  - The authors illustrate this with differing knowledge cutoffs: GPT-4o up to October 2023 and DeepSeek-V3 up to July 2024. Evaluating a model before its cutoff yields "overly optimistic metrics" (p.2).
  - Our observation: Table 2 (p.4) lists DeepSeek-V3 with release date "Mar 2025" and knowledge cutoff "Dec 2024", which differs from the "July 2024" given for DeepSeek-V3 in the p.2 text. The paper does not reconcile the two (they may refer to different checkpoints, but the paper does not say so). Cite the per-model cutoffs as the paper's reported values, not as verified facts.
- Method / architecture:
  - DeepFund is a live forward-testing benchmark (p.2-3).
  - A single selected LLM powers all agents in a multi-agent workflow (p.2). The Financial Planner selects analysts. An Analyst Team (Fundamental, Technical, Insider, Company News, Macro Economic, Policy) emits Bullish/Bearish/Neutral signals with justifications. A Portfolio Manager makes Buy/Sell/Hold decisions and handles risk control over holdings and cash (p.3; Table 1 p.4).
  - The workflow follows an orchestrator-worker paradigm (p.3).
  - A live environment ingests real-time market data, fund assets and trading history through a modular API gateway (Yahoo Finance, Alpha Vantage) (p.3).
  - Typed schemas carry the information: Signal, Decision and Portfolio are wrapped in a FundState object (p.17).
  - Memory is dual (p.17). The short-term memory is FundState. The long-term memory is the trading history.
  - Parameters (p.18): temperature 0.5, technical window 100 days, 10 news items, decision memory size 5.
- Data, universe, evaluation period, baselines:
  - Universe (p.4): Berkshire Hathaway's top five holdings as of Q1 2025, namely AAPL, AXP, BAC, KO and CVX.
  - Capital (p.4): $100,000 per LLM.
  - Analysts used (p.4): only four (Technical, Company News, Policy, Insider). The Fundamental and Macro analysts were not used because they are long-horizon.
  - Period (p.5): live trading from March 17 to April 17, 2025, 24 trading days at daily frequency. The window includes the March 18-19 FOMC meeting and the April 2-9 tariff shock.
  - LLMs (Table 2, p.4), with release dates and knowledge cutoffs: GPT-4.1, Llama 4 Scout, Gemini 2.5 Flash, Claude 3.7 Sonnet, Grok 3 mini Beta, DeepSeek-V3, Qwen2.5-Max, Doubao-1.5-pro, GLM-4-Air.
  - Metrics (p.5): CR, CR under buy-and-hold (CRbnh), SR, MDD, win rate, beta, alpha.
  - Baseline: the S&P 500 appears as a reference row (p.6).
  - Costs: transaction fees are not modeled (p.9). The evaluation cost about $100 in total (p.5).
  - An appendix extends the live window to all of Q2 2025 (p.16).
- Key results (copy numbers EXACTLY, with page):
  - Validity (p.5): 4144 of 4320 signals (96% validity) and 1059 of 1080 trading decisions (98% validity) were valid.
  - Table 3 (p.6):

    | Model / reference | CR(%) | CRbnh(%) | SR | MDD(%) | WR(%) | β | α |
    |---|---|---|---|---|---|---|---|
    | Grok 3 mini Beta | +1.1 | -3.09 | 0.51 | 5.5 | 61 | 0.42 | 0.2 |
    | Gemini 2.5 Flash | -1.9 | -1.58 | -1.37 | 6.4 | 61 | 0.35 | 0.0 |
    | Claude 3.7 Sonnet | -3.7 | -2.94 | -1.45 | 10.1 | 70 | 0.64 | 0.0 |
    | Llama 4 Scout | -4.3 | -3.62 | -2.42 | 8.9 | 61 | 0.36 | -0.1 |
    | DeepSeek-V3 | -5.7 | -5.6 | -1.39 | 14.5 | 57 | 0.94 | 0.0 |
    | GPT-4.1 | -5.9 | -4.41 | -1.87 | 12.8 | 52 | 0.77 | 0.0 |
    | Qwen2.5-Max | -6.7 | -4.86 | -3.12 | 10.7 | 65 | 0.48 | -0.2 |
    | GLM-4-Air | -7.5 | -3.90 | -2.31 | 13.2 | 57 | 0.78 | -0.1 |
    | Doubao-1.5-pro | -8.1 | -5.37 | -2.35 | 13.6 | 65 | 0.84 | -0.1 |
    | S&P 500 | -6.91 | NA | 0.3 | 13.7 | NA | 1.00 | 0.0 |

  - Only Grok 3 had positive CR, and the authors note that passive Buy & Hold "would have more resilience" (p.6).
  - Decision effectiveness (p.7), from the Grok-vs-DeepSeek case study that Figure 4(b) illustrates on AAPL (the text does not say whether the counts cover AAPL only): Grok made 11 Buy decisions (7 effective) and 10 Sell decisions (5 effective). DeepSeek made 3 Buy (1 effective) and 8 Sell (5 effective). An effective decision is a Buy followed by a price increase or a Sell followed by a decrease, excluding the final day (p.7).
  - Q2 2025 extension (p.16): Grok 3 has the highest averaged weekly asset total value, 106262.44 in the week ending 2025-06-29 (Table 4). GPT-4.1, Claude 3.7 and DeepSeek V3 ended with net profits, while Qwen Max, GLM 4 and Gemini 2.5 still showed losses.
  - Cost-efficiency (p.15): among OpenAI models, gpt-4.1-mini is judged the best cost-efficiency choice.
- Limitations (as stated by authors, with page) and limitations we observe (label these "our observation"):
  - Authors (p.9):
    - US stocks only.
    - No transaction fees, market trading restrictions or hybrid strategies.
    - The evaluation period "was short and occurred during a volatile market".
    - Longer tests across market regimes are needed.
  - Our observation: the evidence base is thin. It covers 5 large-cap tickers over 24 trading days, apparently a single run per LLM (no seeds or repeats are described), and no significance tests. Differences of a few percent CR between models are therefore not statistically interpretable.
  - Our observation: the text says Grok's "maximum drawdown was a mere 3%" (p.8) while Table 3 reports MDD 5.5 for Grok 3 mini Beta (p.6).
  - Our observation: every agent role is the same LLM (p.2). The multi-agent structure is therefore held fixed while the backbone varies, which is the 'evaluating models not agents' critique later raised by AMA.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "inadvertently enabling LLMs to “time travel” – leveraging future information embedded in their training corpora, thus resulting in possible information leakage and overly optimistic performance estimates." (p.1)
- "specifically data published after each model’s pretraining cutoff – to ensure fair and leakage-free evaluations." (p.1)
- "Notably, even cutting-edge models such as DeepSeek-V3 and Claude-3.7-Sonnet incur net trading losses within DeepFund real-time evaluation environment" (p.1)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- It is the closest architectural precedent: a planner, domain analysts (fundamental, technical, insider, news, macro, policy) and a portfolio manager with typed Signal/Decision schemas (p.3, p.17). Our design (fundamental / technical / news / political-policy / macro agents feeding a portfolio layer) must be positioned against it.
  - Its analysts emit categorical Bullish/Bearish/Neutral labels from a text-generating LLM (p.3).
  - Our Jev agents return typed Score/Choice answers with confidences (per docs/jev/). That allows principled, calibration-weighted aggregation rather than an LLM "reading" other LLMs' signals.
- Its "time travel" framing and Table 2 of knowledge cutoffs (p.4) directly motivate our look-ahead diagnostics. For CLAUDE.md rule 3 we must report Jev's knowledge cutoff (from docs/jev/) and evaluate on post-cutoff windows and/or forward (paper-trading) windows.
- Its limitations list (no fees, short volatile window, tiny universe; p.9) is a checklist of what our evaluation should improve on: costs, longer horizons, broader universe, multiple seeds and significance testing.
- Its dual memory, FundState plus trading history (p.17), is simpler than our planned multi-layer shared memory. This is a point of contrast.

## BibTeX
```bibtex
@inproceedings{li2025time,
  title     = {Time Travel is Cheating: Going Live with {DeepFund} for Real-Time Fund Investment Benchmarking},
  author    = {Li, Changlun and Shi, Yao and Wang, Chen and Duan, Qiqi and Ruan, Runke and Huang, Weijie and Long, Haonan and Huang, Lijun and Tang, Nan and Luo, Yuyu},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS 2025)},
  pages     = {184676--184704},
  year      = {2025},
  publisher = {Neural Information Processing Systems Foundation, Inc. (NeurIPS)},
  doi       = {10.52202/085713-5554}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- Metadata was checked against the saved Crossref record and the arXiv abs page:
  - title and the 10 authors with their order. Crossref's "SHI"/"RUAN" capitalization is noted.
  - container "Advances in Neural Information Processing Systems 38", pages 184676-184704, DOI 10.52202/085713-5554, and the event (NeurIPS 2025, San Diego and Mexico City).
  - the arXiv v1 (2025-05-16) and v2 (2025-10-14) dates.
  - the arXiv comment "NeurIPS 2025 Datasets and Benchmarks Track", which matches the PDF p.1 footer.
- Every number was checked against the PDF text by page:
  - p.2: cutoffs.
  - p.4: Table 1/2, universe, capital, analysts used.
  - p.5: period, events, validity counts, $100 cost.
  - p.6: Table 3, all rows.
  - p.7: decision-effectiveness counts.
  - p.8: the "mere 3%" MDD statement.
  - p.9: limitations.
  - p.15: gpt-4.1-mini.
  - p.16: Table 4 Q2 extension.
  - p.17: schemas and dual memory.
  - p.18: parameters.
- All three quotes match verbatim, apart from line-break hyphenation. The BibTeX matches Crossref.
- Corrected: no factual errors in the existing text. One observation was added: the paper is internally inconsistent about DeepSeek-V3's knowledge cutoff, July 2024 on p.2 versus Dec 2024 in Table 2 on p.4.
- Second adversarial pass (2026-09-25, independent re-check of the current text):
  - Re-checked the metadata against the saved Crossref record (title, 10 authors, container, pages 184676-184704, DOI, event) and the arXiv abs page (v1 2025-05-16, v2 2025-10-14, Datasets and Benchmarks Track comment). All match.
  - Re-checked every number: cutoffs (p.2), Tables 1-2 and the portfolio configuration (p.4), period, validity counts and $100 cost (p.5), all Table 3 rows (p.6), decision counts (p.7), "mere 3%" (p.8), limitations (p.9), gpt-4.1-mini (p.15), Table 4 and the Q2 statement (p.16), schemas and memory (p.17), parameters (p.18). All correct.
  - Re-checked all three quotes (p.1) and the quoted phrases (p.2, p.6, p.8, p.9). They match.
  - Corrected:
    - The cutoff observation said Table 2 "contradicts" p.2. Softened to "differs from", noting that Table 2 lists a Mar 2025 release for DeepSeek-V3 and that the paper does not reconcile the two values.
    - The decision-effectiveness counts now state their context (the Grok-vs-DeepSeek case study illustrated on AAPL in Figure 4(b)) and the paper's definition of an effective decision (p.7).
