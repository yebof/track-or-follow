# xiao2024tradingagents: TradingAgents: Multi-Agents LLM Financial Trading Framework

## Verified metadata
- Title: TradingAgents: Multi-Agents LLM Financial Trading Framework [source: arxiv]
- Authors: Yijia Xiao, Edward Sun, Di Luo, Wei Wang [source: arxiv]
- Venue: arXiv preprint [source: arxiv]. The arXiv comment reads "Tauric Research @ https://github.com/TauricResearch; Oral @ Multi-Agent AI in the Real World", and there is no journal_ref [source: arxiv]. Crossref returned no matching record (search saved as literature/meta/xiao2024tradingagents.crossref-search-nomatch.json). The workshop presentation is NOT confirmed by DBLP, Crossref or a publisher page, so the paper is treated as a preprint.
- Year: 2024 (v1 published 2024-12-28; v7 updated 2025-06-03) [source: arxiv]
- DOI / arXiv: none / arXiv:2412.20138 (v7)
- Metadata files: literature/meta/xiao2024tradingagents.arxiv.xml (arXiv API), literature/meta/xiao2024tradingagents.crossref-search-nomatch.json (Crossref bibliographic search; no matching record). DBLP (timeouts, then a bot-challenge page), OpenAlex (HTTP 429) and Semantic Scholar (HTTP 429 on three back-off attempts) could not be used on 2026-09-25, so the workshop status remains unverified and the paper is cited as an arXiv preprint.

## PDF
- Path: literature/pdf/xiao2024tradingagents.pdf (arXiv:2412.20138v7)
- Text: literature/text/xiao2024tradingagents.txt ; pages: 38
- Page numbers are PDF page indices, which here equal the printed page numbers.

## Summary (grounded; every factual sentence cites a page)
- **Problem:** The paper names two limitations of existing financial language agents (p.2):
  - a lack of realistic organizational modeling of trading firms;
  - inefficient natural-language communication that causes a "telephone effect" as conversations lengthen.
- **Method / architecture:**
  - **Roles.** A simulated trading firm with seven roles: Fundamentals Analyst, Sentiment Analyst, News Analyst, Technical Analyst, Researcher, Trader and Risk Manager (p.5).
  - **Analyst team.** Four analysts gather data concurrently (Fig. 1, p.3; p.5-6).
    - The sentiment analyst uses web search, Reddit and X/Twitter tools (p.5).
    - The news analyst also covers government announcements and macroeconomic indicators (p.6).
  - **Researcher team.** A bullish and a bearish researcher debate for n rounds. A facilitator picks the prevailing view and records it as a structured entry (p.6, p.9).
  - **Trader.** Decides timing and size of trades (p.7).
  - **Risk management and approval.** The risk management team deliberates from risk-seeking, neutral and conservative perspectives for n rounds. A fund manager then approves and updates the decision (p.8-9).
  - **Agent framework.** All agents follow ReAct (p.7).
  - **Communication.** Agents "communicate primarily through structured documents and diagrams" that are stored in a global state. Natural-language dialogue is used only in debates (p.8-9).
  - **Backbones.** Quick-thinking models (gpt-4o-mini, gpt-4o) handle retrieval and summarization. Deep-thinking models (o1-preview) handle analysis and decision-making (p.9).
  - **Long and short.** The transaction-history figures mark long and short positions (Fig. 6, p.11; Fig. S1, p.20), so the evaluated action space is not long-only.
- **Data, universe, evaluation period, baselines:**
  - **Period and frequency.** Backtest from 2024-01-01 to 2024-03-29 with daily decisions (p.10). Decisions use only "data available up to each trading day" (p.10).
  - **Stated universe.** Apple, Nvidia, Microsoft, Meta and Google (p.10).
  - **Data.** Prices; news from Bloomberg, Yahoo, EODHD, FinnHub and Reddit; social media posts and sentiment; insider sentiment and transactions (SEDI); financial statements; company profiles; 60 technical indicators (p.10).
  - **Baselines.** Rule-based only: Buy and Hold, MACD, KDJ+RSI, ZMR and SMA (p.10, p.19).
  - **Metrics.** CR, AR, SR and MDD (p.10, p.19-20).
  - **Costs.** No transaction-cost model is described (our observation, based on a full-text search).
- **Key results (numbers copied exactly):**
  - Table 1 (p.11), TradingAgents CR% / ARR% / SR / MDD%:

    | Stock | CR% | ARR% | SR | MDD% |
    |---|---|---|---|---|
    | AAPL | 26.62 | 30.5 | 8.21 | 0.91 |
    | GOOGL | 24.36 | 27.58 | 6.39 | 1.69 |
    | AMZN | 23.21 | 24.90 | 5.60 | 2.11 |

  - B&H CR% in the same table: AAPL -5.23, GOOGL 7.78, AMZN 17.1 (p.11).
  - The authors state that TradingAgents achieves "at least a 23.21% cumulative return and a 24.90% annual return on the three sampled stocks" (p.11).
  - Cost footnote (p.12): the benchmark covered only 3 months "due to intensive LLM and tool use (11 LLM calls & 20+ tool calls" per prediction. The highest Sharpe ratio "exceeds our expected empirical range", which the authors attribute to few pullbacks.
- **Limitations stated by the authors:**
  - The backtest is short because of LLM and tool cost; future work will "optimize LLM reasoning & tool use to enable longer backtesting under limited budgets" (p.12).
  - Future work also includes live deployment and real-time data (p.13).
- **Limitations we observe (our observation):**
  - The test window is under three months, and there are no statistical tests or repeated runs that we could find.
  - Results are reported for only three stocks (AAPL, GOOGL, AMZN; p.11), whereas the setup lists Apple, Nvidia, Microsoft, Meta and Google (p.10). AMZN is not in that list, and NVDA, MSFT and META results are not shown in Table 1.
  - Only rule-based baselines are used. There are no LLM-agent baselines such as FinMem or FinAgent, although they are discussed in related work (p.4).
  - No transaction costs are modeled.
  - We found no discussion of the backbone LLMs' training-data cutoffs relative to the 2024 test window, i.e., look-ahead through model knowledge, even though the paper claims to eliminate look-ahead bias at the data level (p.10).
  - Explainability is claimed from natural-language logs (p.13) and is not measured.
  - The illustrative agent reports and the full appendix trading log are for AAPL "as of November 19, 2024, a recent date chosen for demonstration purposes" (p.22; also Fig. 2, p.5), i.e., outside the 2024-01-01 to 2024-03-29 backtest window, so they do not show the evaluated decisions.

## Verbatim quotes (copied exactly from the PDF text)
1. "This approach often results in a “telephone effect”, where details are lost, and states become corrupted as conversations lengthen." (p.2)
2. "TradingAgents agents communicate primarily through structured documents and diagrams." (p.8)
3. "Agents make decisions based solely on data available up to each trading day, ensuring no future data is used (eliminating look-ahead bias)." (p.10)
4. "We believe the exceptionally high SR resulted from the phenomenon that there were few pullbacks in TradingAgents during that period." (p.12)

## Relevance to our project
- **Main text-generating multi-agent baseline.** Its role decomposition (fundamentals, sentiment, news incl. macro/government, technical; bull/bear debate; risk team; fund manager) closely matches our specialist-agent layout (p.5-9). It is the design our Jev-driven system must be contrasted with and, ideally, beaten under identical data, universe, period and costs.
- **Motivation for typed decisions.** The paper itself identifies natural-language communication as a failure source ("telephone effect", p.2) and moves toward structured reports (p.8). Our design takes this further: per docs/jev/docs_introduction.txt, Jev returns typed values and probability distributions with no text generation. Inter-agent messages then become typed judgments with calibrated confidence instead of reports that another LLM must read.
- **Cost and latency argument.** The footnote reports 11 LLM calls and 20+ tool calls per prediction, which limited the backtest to 3 months (p.12). This supports our claim that a cheaper, faster typed-decision engine enables longer, multi-regime and larger-universe backtests. Any concrete Jev cost or latency numbers must be cited from docs/jev/ and measured by us.
- **Evaluation gaps we should close.** Longer periods, more tickers, transaction costs, LLM baselines, statistical significance, and an explicit LLM knowledge-cutoff diagnostic.
- **Reproducibility.** Code is public (p.1), so a faithful re-run on our protocol is feasible. The exact implementation details must come from the repository, not from this summary.

## BibTeX
```bibtex
@article{xiao2024tradingagents,
  title   = {{TradingAgents}: Multi-Agents {LLM} Financial Trading Framework},
  author  = {Xiao, Yijia and Sun, Edward and Luo, Di and Wang, Wei},
  journal = {arXiv preprint arXiv:2412.20138},
  year    = {2024}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check pass).
- What was checked:
  - Metadata: title, 4 authors, arXiv id, v1 date (2024-12-28), v7 date (2025-06-03), comment field and absence of journal_ref against literature/meta/xiao2024tradingagents.arxiv.xml; Crossref search file (no matching record). Re-queries of DBLP (bot-challenge page), OpenAlex (HTTP 429) and Semantic Scholar (HTTP 429 after back-off) failed, so the workshop "Oral" claim remains unconfirmed and the preprint treatment is kept. The PDF header matches (arXiv:2412.20138v7, 3 Jun 2025).
  - Text: every page reference, the 7 roles, team workflow, backbone models, backtest dates, data sources, baselines, metrics, Table 1 (p.11) cell by cell, the footnote on p.12 and all four verbatim quotes against literature/text/xiao2024tradingagents.txt (38 pages; equals `pdfinfo`); appendix S1.1-S1.2 (p.19-20) for baseline and metric definitions. Full-text search found no transaction-cost, significance-test or repeated-run text. Jev statement checked against docs/jev/docs_introduction.txt.
  - BibTeX fields against the arXiv record.
- What was corrected:
  - No factual errors were found in the existing text.
  - Added two grounded facts: long and short positions in the transaction-history figures (p.11, p.20), and that the illustrative reports/logs are dated November 2024, outside the backtest window (p.5, p.22).
  - Metadata-source note updated with the failed re-queries.
- Second pass (2026-09-25, resumed after an interruption): all quotes and inline fragments re-matched on their cited pages; Table 1 (p.11) re-compared cell by cell; the "Nov 4-19, 2024" label in Fig. 2 (p.5) and the long/short captions (p.11, p.20) re-confirmed. The venue re-query failed again: DBLP bot-challenge page, OpenAlex daily budget exhausted, Semantic Scholar HTTP 429 after a 40 s back-off. The workshop "Oral" claim remains unverified, so the paper is still cited as an arXiv preprint. No new errors found.
