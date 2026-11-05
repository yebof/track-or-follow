# lopezlira2026chatgpt: Can ChatGPT forecast stock price movements? Return predictability and large language models

## Verified metadata
- Title: Can ChatGPT forecast stock price movements? Return predictability and large language models [source: crossref]. The arXiv title is "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models" [source: arxiv].
- Authors: Alejandro Lopez-Lira, Yuehua Tang [source: crossref; arxiv]
- Venue: Journal of Financial Economics, vol. 184, article 104335 (Elsevier BV; ISSN 0304-405X), issue dated October 2026 [source: crossref]
- Year: 2026 [source: crossref]
- DOI / arXiv: 10.1016/j.jfineco.2026.104335 / arXiv:2304.07619 (v6, 2025-10-28). The arXiv comment notes the paper was "Previously posted in SSRN" (abstract_id 4412788) [source: arxiv].
- Metadata files: literature/meta/lopezlira2026chatgpt.crossref.json, literature/meta/lopezlira2026chatgpt.arxiv.xml

## PDF
- Path: literature/pdf/lopezlira2026chatgpt.pdf. This is arXiv v6, labeled "This Version: October 30, 2025" (p.1). It is the authors' working-paper version with appendices, not the typeset JFE article; the Elsevier version was not freely downloadable. The acknowledgments thank "Dimitris Papanikolaou (the editor), two anonymous referees" (p.1). Numbers in the published article may differ.
- Text: literature/text/lopezlira2026chatgpt.txt ; pages: 142 (main text p.1-36; tables and figures after)

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Can a general-purpose LLM, without financial training, correctly assess the economic implications of firm news headlines (p.2)?
  - Do its assessments predict the initial market reaction and the subsequent drift?
  - Can LLMs serve as an instrument for studying market information processing (p.2)?
- Method / architecture:
  - Zero-shot prompting of GPT-4 ("gpt-4-0314", training data up to September 2021) per headline, with temperature 0 (p.15-16).
  - The prompt asks the model to answer "YES" (good news), "NO" (bad news) or "UNKNOWN" and then give a one-sentence rationale (p.15).
  - Answers are mapped to a score: YES = 1, UNKNOWN = 0, NO = −1 (p.16).
  - Scores drive daily-rebalanced, equal-weighted long-short portfolios (p.17-19). For news released before 9 a.m. on a trading day, the position is entered at that day's open and exited at the close. For news released after the close, it is entered at the next open and exited at the next close (p.19). Intraday-news drift is measured from the news day's close to the next day's close (Table 1 notes, p.51).
  - Comparisons (p.4-5, p.16):
    - smaller or older models: GPT-1, GPT-2, BERT, Llama2, GPT-3.5, DistilBART-MNLI (12 LLMs in total);
    - embedding-based supervised models trained on rolling 6-month windows.
  - Topic modeling provides interpretability (p.4). A theoretical model with LLM technology, information-processing capacity constraints, underreaction and limits to arbitrage guides the analysis (abstract p.1; p.3 lists LLM technology, information processing constraints and limits to arbitrage).
- Data, universe, evaluation period, baselines:
  - Sources (p.13): CRSP daily returns, TAQ intraday data, web-scraped headlines matched to RavenPack.
  - Sample (p.13): 159,137 firm-headline-date observations for 4,123 US common stocks (about 85% of 4,875 CRSP firms) from October 2021 to May 2024. The window was chosen to follow the model's September 2021 training cutoff.
  - Filters (p.14): RavenPack relevance score of 100; event-similarity-days > 90 to keep only new information. About 82% of headlines are overnight news.
  - Frequency: daily rebalancing (p.19).
  - Costs (p.20): headline results are before costs; sensitivity at 5, 10 and 20 bps round-trip.
  - Reference: a value-weight market portfolio is used for comparison (p.19).
- Key results (copy numbers EXACTLY, with page):
  - Long-short hit rates (Table 1, p.51): initial reaction 93.28% (overnight) and 88.78% (intraday); drift 58.06% and 54.67%.
  - Drift mean daily returns: 0.34% (overnight) and 0.50% (intraday) (p.51).
  - Annualized drift Sharpe ratios: 2.97 (overnight) and 2.63 (intraday), pre-transaction costs (p.18, p.51).
  - Overnight legs (p.18-19): the long leg averages 8 bps per day (Sharpe 0.78); the short leg averages 26 bps per day (Sharpe 2.01).
  - By model (p.4-5): GPT-4's drift Sharpe of 2.97 compares with 1.66 for GPT-3.5 and 1.26 for DistilBART-MNLI.
  - Embedding baselines (p.5): OpenAI embeddings reach Sharpe 1.71 (intraday) and 2.53 (overnight), vs GPT-4's 2.63 intraday.
  - Cumulative return and costs (p.19-20): about 700% from 2021m10 to 2024m5 without costs. It stays above 300% at 5 bps round-trip and above 100% at 10 bps, and the strategy is unprofitable at 20 bps.
  - Turnover (p.20): about 190% per day. The Sharpe ratio at 10 bps is 1.29 with full rebalancing vs 1.34 with 25% partial rebalancing.
  - Decay over time (p.6, p.35): Sharpe falls from 6.54 (2021Q4) to 3.68 (2022), 2.33 (2023) and 1.22 (January-May 2024).
  - By size (p.35): after negative news, large-cap stocks show no drift while small-cap stocks keep drifting.
- Limitations (as stated by authors, with page) and limitations we observe (label these "our observation"):
  - Authors:
    - Exploiting the drift "is only feasible for market participants whose transaction costs are sufficiently low, such as market makers" (p.3), and the strategy is unprofitable at 20 bps (p.20).
    - Llama-2's later knowledge cutoff (September 2022) may overstate its results (p.16).
    - Other factors besides LLM adoption could explain the Sharpe decline (p.35).
  - Our observation: this is a headline-level, next-day (high-turnover) signal study, not a trading agent. It fuses nothing across fundamentals, technicals or macro, and the LLM output is a three-way label rather than a probability.
  - Our observation: the documented alpha decays toward the end of the sample (p.35). A news agent deployed in 2026 should not assume the 2021-2022 magnitudes.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "Using post-knowledge-cutoff headlines, GPT-4 captures initial market responses, achieving approximately 90% portfolio-day hit rates for the non-tradable initial reaction." (p.1)
- "A long-short daily-rebalanced strategy based on GPT-4’s assessments of overnight news generates average returns of 34 basis points (bps) per day before transaction costs." (p.3)
- "A transaction cost of 20 bps round-trip makes the strategy unprofitable." (p.20)
- "the annualized Sharpe ratio declines from 6.54 in 2021Q4 to 3.68 in 2022, 2.33 in 2023, and 1.22 over January-May 2024." (p.35)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- It is the top-finance-journal evidence (JFE) that zero-shot LLM judgments of news headlines carry return-relevant information. This justifies a dedicated news agent. Its YES/NO/UNKNOWN-to-{1, 0, −1} scoring (p.15-16) maps directly onto a Jev Choice question, with Jev's confidence adding a graded signal (per docs/jev/).
- Its methodology is a template for our look-ahead discipline:
  - evaluate only on post-knowledge-cutoff news (p.13);
  - align news timestamps with tradable entry times, splitting overnight and intraday news with a 9 a.m. cutoff (p.14, p.19).
  - This matches CLAUDE.md rule 3 (point-in-time news by publication timestamp).
- Its cost sensitivity (unprofitable at 20 bps; p.20) and high turnover (about 190% per day; p.20) argue for our mid/low-frequency design. We must report cost-inclusive results and show turnover control.
- Its evidence of decaying LLM-news alpha (p.35) motivates combining news with other sub-domain agents (fundamental, technical, macro, policy) instead of relying on news alone. It is also a caution against over-claiming news-driven returns in our 2026 setting.

## BibTeX
```bibtex
@article{lopezlira2026chatgpt,
  title     = {Can {ChatGPT} forecast stock price movements? Return predictability and large language models},
  author    = {Lopez-Lira, Alejandro and Tang, Yuehua},
  journal   = {Journal of Financial Economics},
  volume    = {184},
  pages     = {104335},
  year      = {2026},
  month     = oct,
  publisher = {Elsevier BV},
  issn      = {0304-405X},
  doi       = {10.1016/j.jfineco.2026.104335}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- Metadata was checked against the saved records:
  - Crossref: title, both authors, Journal of Financial Economics vol. 184, article number 104335, published-print 2026-10, Elsevier BV, ISSN 0304-405X (print), DOI 10.1016/j.jfineco.2026.104335.
  - arXiv (v6, updated 2025-10-28): the SSRN comment and the title capitalization.
- The PDF is arXiv v6. p.1 carries "This Version: October 30, 2025" and thanks the editor and referees.
- Every number was checked against the text, with page:
  - p.1: abstract.
  - p.3: 34 bps; market-maker caveat.
  - p.4-5: 93.3/88.8; 12 LLMs; Sharpe 2.97/1.66/1.26; embeddings 1.71/2.53 vs 2.63.
  - p.6: decay.
  - p.13-14: data, sample counts, 85%, 82%, relevance 100, event-similarity > 90, the 9 a.m. cutoff.
  - p.15-16: prompt, temperature 0, gpt-4-0314, score mapping, Llama-2 cutoff.
  - p.17-20: portfolio construction, leg returns and Sharpes, 700%/300%/100%, 20 bps, turnover 190%, Sharpe 1.29 vs 1.34.
  - p.35: Sharpe decay, size asymmetry.
  - p.51: Table 1.
- All four quotes match the text. The only difference is that "post-knowledge-cutoff" breaks across a line in the PDF.
- The BibTeX matches Crossref.
- Corrected: made the trade-entry timing precise (p.19). It had said "overnight news enters at the open", but "overnight" covers both pre-9 a.m. and after-close news, and these have different entry times. Also added the intraday drift-measurement window from the Table 1 notes (p.51). No other errors found.
- Second adversarial pass (2026-09-25, independent re-check of the current text):
  - Re-checked the metadata against the saved Crossref record (JFE vol. 184, article 104335, published-print 2026-10, Elsevier BV, ISSN 0304-405X print, DOI) and the arXiv Atom record (v6 updated 2025-10-28, SSRN comment). All match. The PDF p.1 stamp reads "arXiv:2304.07619v6 [q-fin.ST] 28 Oct 2025" and "This Version: October 30, 2025".
  - Re-checked every number: p.2-6 (introduction figures, 12 LLMs, 2.97/1.66/1.26, 1.71/2.53/2.63, Sharpe decay), p.13-16 (data, 159,137 / 4,123 / 4,875 / 85%, 82% overnight, relevance 100, event similarity > 90, prompt, temperature 0, gpt-4-0314, score mapping, Llama-2 footnote), p.17-20 (portfolio timing, legs, 700%/300%/100%, 20 bps, 190% turnover, 1.29 vs 1.34), p.35 (decay, size asymmetry, other factors) and Table 1 (p.51). All correct.
  - Re-checked all four quotes (p.1, p.3, p.20, p.35). They match; "post-knowledge-cutoff" breaks across a line on p.1.
  - Corrected: the theoretical-model sentence cited only p.3, but "underreaction" appears in the abstract (p.1), not in the p.3 list. The citation now covers both pages.
