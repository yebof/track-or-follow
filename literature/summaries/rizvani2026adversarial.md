# rizvani2026adversarial: Adversarial News and Lost Profits: Manipulating Headlines in LLM-Driven Algorithmic Trading
## Verified metadata
- Title: Adversarial News and Lost Profits: Manipulating Headlines in LLM-Driven Algorithmic Trading [source: crossref; arxiv; openalex; PDF p.1]
- Authors: Advije Rizvani, Giovanni Apruzzese, Pavel Laskov [source: crossref; arxiv; openalex]
  - Affiliations on PDF p.1: Liechtenstein Business School, University of Liechtenstein; Apruzzese also Reykjavik University.
- Venue: 2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), Munich, Germany, 23–25 March 2026, IEEE, pp. 849–866 [source: crossref (type proceedings-article, container title, event, pages 849-866); arxiv comment: "accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)"]
  - DBLP was unreachable (bot-protection page).
- Year: 2026 (Crossref issued 2026-03) [source: crossref]
- DOI / arXiv: DOI 10.1109/SaTML68715.2026.00053; arXiv:2601.13082 (v1 19 Jan 2026, cs.CR)
- Metadata files:
  - literature/meta/rizvani2026adversarial.crossref.json
  - literature/meta/rizvani2026adversarial.arxiv_abs.html
  - literature/meta/rizvani2026adversarial.openalex.json (W7144093666)
## PDF
- Path: literature/pdf/rizvani2026adversarial.pdf. This is the arXiv v1 author version (https://arxiv.org/pdf/2601.13082), not the IEEE Xplore typeset version; the title matches.
- Text: literature/text/rizvani2026adversarial.txt ; pages: 18 (page numbers below are arXiv PDF pages, not proceedings pages 849-866)
## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - RQ: what happens to the profitability of an algorithmic trading system (ATS) that ingests "adversarial news"? Such news is manipulated to mislead an LLM while remaining visually unaltered to humans (p.1).
  - The authors argue that prior work had not quantified the system-wide monetary impact on LLM-supported ATS (p.1). Their systematic review of over 25k papers found no such evaluation (p.2-3, p.15-16).
- Method / architecture:
  - Threat model (p.4):
    - The attacker is outside the ATS. It knows the ATS ingests headlines, maps them to tickers and scores sentiment, and it knows at least one traded stock s*.
    - It can manipulate headlines mentioning s* on a single day t*: by writing the news at the source, through a man-in-the-middle, or (unlikely) at the vendor.
    - It "cannot touch prices, retrain models, manipulate the news beyond the headline, or persist across multiple days".
  - Two human-imperceptible attacks (p.1, p.4):
    - Unicode homoglyph substitution: Latin → Cyrillic letters in the company name (Tables XII-XIV, p.18), which makes stock-name association fail ("misroute").
    - Hidden-text injection: a sentiment-reversing clause wrapped in `<span style="display:none">`. The experiments use the fixed string "losses and layoffs" (p.7).
  - ATS design (p.5-6):
    - Implemented in Backtrader.
    - An LSTM on 50-day OHLCV windows forecasts the next-day price, giving ∆. It is trained Jan 2013–Dec 2023.
    - An LLM maps each headline to a ticker (prompt in Listing 1, p.17). FinBERT scores polarity, averaged daily and smoothed with a 7-day moving average.
    - A fused score Σ = w_p∆ + w_n s̄ is thresholded into long, short or hold. Hyperparameters are tuned on the training set (p.6).
    - Signals at the end of day t execute at the open of t+1 (p.5).
  - Evaluation: every combination of 10 stocks × 420 days × 2 attacks, "≈8k perturbations". Each attacked run is compared with a paired clean backtest (p.7).
- Data, universe, evaluation period, baselines:
  - Universe: 10 large caps {GOOGL, AAPL, NVDA, MSFT, AMZN, META, TSLA, LLY, JPM, XOM} (p.5).
  - Data: Yahoo Finance daily bars (2013–2025) and Refinitiv headlines (p.5).
  - Test window: Feb 2024–Apr 2025 (14 months), with no overlap with training (p.6).
  - Trading setup: initial capital $1,000,000; transaction cost $0.005 per share; slippage 0.02 (p.6). Decisions are daily, at most one per asset (p.5).
  - Baselines: an LSTM-only ATS vs the LSTM+LLM ATS (p.6).
  - Transfer (p.8): FinBERT is swapped for FinGPT, FinLLaMA, O3, O3 Pro, 4o, 4o-mini-high, 4o-mini, GPT-5 and Gemini Pro 1.5. Decoding is deterministic where supported (temperature 0), offline models use seed 42, and API models were tested in July/August 2025 (p.17).
  - User study: 27 valid responses from FinTech practitioners, out of 83 invitations sent Sept. 10–23, 2025 (32% response rate) (p.10).
- Key results (copy numbers EXACTLY, with page):
  - Clean baselines: LSTM-only CR 7.9%; LSTM+LLM CR 19.22% at the end of the test window (p.6).
  - Proof of concept, NVIDIA, 1 Aug 2024, 9 headlines:
    - Homoglyph: clean association recognizes 8 of 9; after the attack, none is recognized (Table I, p.6-7).
    - Hidden text: 7 of 9 headlines flip polarity (Table II, p.7).
  - NVDA, attack on a random day:
    - Homoglyph: avg CR ≈ 17.5%, "about 1.7 percentage points" lower, "a net loss of 17,200$" on $1M (p.7).
    - Hidden text: avg CR ≈ 16.0%, about 3.2 pp lower, "an approximate $32.2k shortfall" (p.8).
  - All days, all stocks, FinBERT, homoglyph: 5,957 of 6,012 headlines (99.1%) misroute the association (p.7). Table III (p.8):
    - days with decision change 100.0%;
    - mean ∆CR −3.67 pp;
    - mean |∆CR| 4.46 pp;
    - worst case −17.70 (TSLA, 28 Feb 2024);
    - best case +4.58 (NVDA, 29 Feb 2024).
  - All days, all stocks, FinBERT, hidden text (p.8): 65.6% of headlines flip polarity; 77.3% of trials reduce CR; mean −3.18 pp ∆CR; worst case −17.7 pp (TSLA, 28 Feb 2024).
  - Transfer, homoglyph mapping accuracy clean → attack (Table V, p.9): O3 96.72 → 88.40; GPT-5 94.50 → 50.00; FinBERT 90.04 → 0.90.
  - Transfer, hidden-text flip rates (Table VI, p.9): O3 40.0%, GPT-5 43.0%, FinBERT 65.6%, Gemini Pro 1.5 86.0%.
  - Transfer, system level: homoglyph worst case O3 −2.0 pp vs FinBERT −17.70 pp (Table VII, p.9); hidden text worst case O3 −2.4 pp (Table VIII, p.9).
  - Clean CR ranges from FinLLaMA 19.5% (highest) to Gemini Pro 1.5 15.2% (lowest) (p.8). The most robust model (O3) is not the most profitable clean model (p.11).
  - Several models become more confident under attack (O3: +0.31), "so confidence-based monitoring alone may miss failures" (p.9).
  - Scraping libraries: Scrapy, BeautifulSoup and Cheerio keep hidden HTML, while newspaper3k strips it. "none of these libraries normalize Unicode by default" (p.12; Table X, p.16).
  - A Cyrillic→Latin sanitizer "always defuses our homoglyph attack" in under 0.1 s per headline, but other homoglyphs evade it (p.12).
  - Survey Q9: 55.6% of practitioners rate manipulated-news-misleads-LLM as "somewhat realistic" and 33% as "very realistic" (p.10).
- Limitations:
  - As stated by the authors (p.11):
    - they "do not claim generality" across other ATS designs, frequencies, windows or models;
    - real-world ATS may behave differently;
    - the attack space is unbounded;
    - they assume manipulations reach the LLM unsanitized;
    - the survey sample is too small for generalizable claims (p.10).
  - Our observations:
    - **Abstract vs body metric.** The abstract says returns fall "by up to 17.7 percentage points" of "annual returns" (p.1). The body measures the change in end-of-window cumulative return over the 14-month test (∆CR, p.7-8), not an annualized return.
    - **Numbers disagree across sections.**
      - The contributions list gives FinBERT failing on hidden text "67% of the time" (p.2), vs 65.6% in §V-C and Table VI (p.8-9).
      - The average profitability drop is "≈3.5%" in the introduction and "≈3%" in the contributions (both p.2), vs −3.67 pp (homoglyph) and −3.18 pp (hidden text) in §V (p.8).
      - §VI-A announces "six general-purpose models" but lists seven (O3, O3 Pro, 4o, 4o-mini-high, 4o-mini, GPT-5, Gemini Pro 1.5), for nine transfer models in total (p.8).
      - Table VI gives 4o-mini a hidden-text flip rate of 46.0%, close to GPT-5's 43.0%. Yet the prose and Table VIII rank 4o-mini among the most vulnerable models at system level, with 76.1% of days impacted (p.9). The two measures differ: the flip rate "counts only polarity reversals (+1 ↔ −1)" (p.9), while system level counts impacted days. So model-level flip rates should not be used to rank system-level risk.
    - **Mapping accuracy conflict.**
      - The authors report that manual checks of ≈7,000 stock-name associations found them "always correct" (p.5). Yet Table V lists 90.04% clean mapping accuracy for the FinBERT configuration (p.9), and Table I shows one clean headline as "Unrec." (p.7).
      - FinBERT itself is a sentiment classifier, and which model performs the association in that configuration is not made explicit (p.5). Table XI lists a separate "ChatGPT (4o-mini)" entry without stating its role (p.17).
    - **"420 days".** The count for Feb 2024–Apr 2025 (p.7) exceeds the roughly 300 trading days in 14 months (our estimate), so "days" apparently counts calendar dates; the paper does not say.
    - **Path dependence.** The worst case falls on the same stock and day (TSLA, 28 Feb 2024) for both attacks, with the same −17.7 pp (p.8). This suggests the tail loss comes from path-dependent reallocation after one wrong decision rather than from the attack type.
    - **Single run.** Results come from one deterministic backtest path per perturbation. The authors use fixed seeds (p.16-17) but report no repeated runs and no CIs on the aggregate ∆CR (p.8-9). The only significance claim is a t-test for O3's robustness (p.9). The ATS may short (p.6).
## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "the news headline may include “malicious” content that remains invisible to human readers but which is still ingested by the LLM." (p.1)
- "so confidence-based monitoring alone may miss failures." (p.9)
- "none of these libraries normalize Unicode by default." (p.12)
- "What is lost is the counterfactual margin—the extra return that would have been realized under clean inputs." (p.12)
## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Concrete attack families for the static red-team.**
  - Our A4 ("obfuscated instruction: zero-width characters, homoglyphs, mid-sentence", docs/design/design_v1.md §13.1) uses homoglyphs to hide an instruction.
  - This paper adds two distinct, peer-reviewed families we should cover:
    - entity misrouting: homoglyphs in the company name, which make a real item vanish from a stock's evidence (p.4, p.7);
    - hidden-markup clauses (display:none, font-size 0, near-background color) that change polarity without any visible instruction (p.4, p.11).
  - Both are cheap and need no model access.
  - Actions:
    - add a misrouting variant targeting our focal-entity matching (regex aliases) and a hidden-clause variant to the A-family templates;
    - add a preprocessing defense to N0: Unicode NFKC plus confusable folding and HTML-to-visible-text rendering. It must run before any S1 call and be logged as part of D0–D6.
- **News sentinel and the demotion cap.**
  - Hidden negative clauses ("losses and layoffs", p.7) are exactly the kind of single-source negative manipulation our demotion cap bounds: a single cluster can at most halve a position, and exclusion needs ≥ 2 corroborating items (design_v1 §2, sentinel table).
  - Their worst case of −17.7 pp from one day's manipulation (p.8) motivates reporting a paired "counterfactual margin" (p.12) for attacked stock-weeks alongside the binary C1 success. Our replay design makes this nearly free.
- **Calibrated probabilities are not an attack detector.** Confidence rises under attack for several models (p.9). Jev's calibrated probabilities (docs/jev/docs_introduction.txt) are calibrated on clean data, so we must not claim that low S1 confidence flags manipulation unless C1 measures it. C1 should report the confidence/probability shift on attacked items.
- **Detection is hard, so bound by construction.** The authors note that the loss is counterfactual and the ATS stays profitable, so operators do not notice (p.7, p.12). This supports a by-construction bound plus a write gate over post-hoc detection.
- **Monetary-impact framing for the paper.** A peer-reviewed SaTML result quantifies the dollar impact of one-day headline manipulation on an LLM-driven ATS (p.7-8). It is a good citation for the Web-security motivation of WWW '27. We must quote it as a 14-month cumulative-return delta, not an "annual return" drop.
- **How we differ:**
  - Their ATS trades daily on 10 names and fuses an LSTM with headline sentiment, with shorting (p.5-6). Ours is weekly, long-only and cash-only on a PIT top-100 universe, with typed S1 judgments over news and policy items and a code-level influence bound.
  - Their attacker acts on one day. Our A7 repeats a plant over three weeks.
## BibTeX
```bibtex
@inproceedings{rizvani2026adversarial,
  author    = {Rizvani, Advije and Apruzzese, Giovanni and Laskov, Pavel},
  title     = {Adversarial News and Lost Profits: Manipulating Headlines in {LLM}-Driven Algorithmic Trading},
  booktitle = {2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  pages     = {849--866},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/SaTML68715.2026.00053}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check pass; files touched 2026-09-26 local time).
- Metadata checked against saved raw records: `meta/rizvani2026adversarial.crossref.json` (DOI 10.1109/SaTML68715.2026.00053, type proceedings-article, container "2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)", event Munich, Germany, 2026-03-23 to 2026-03-25, pages 849-866, issued 2026-03, three authors in order, publisher IEEE), `meta/rizvani2026adversarial.arxiv_abs.html` (arXiv 2601.13082 v1, 19 Jan 2026; the SaTML acceptance comment; citation_doi matches Crossref), `meta/rizvani2026adversarial.openalex.json` (W7144093666, DOI match, 2026). The arXiv abstract page was re-fetched live on 2026-09-26 and still shows only v1. PDF p.1 matches the title, authors, affiliations and "arXiv:2601.13082v1 [cs.CR] 19 Jan 2026". The acknowledgments thank "the anonymous SaTML’26 reviewers" (p.12). The venue claim and the BibTeX are supported; DBLP was blocked by a bot check.
- Content checked against `text/rizvani2026adversarial.txt` page by page: the RQ, abstract claims and contributions (pp.1-2), the SLR (pp.2-3, 15-16), the threat model and quote (p.4), the ATS design, universe, data and the ≈7,000 manual checks (p.5), the 7-day MA, fusion eq. (3), shorting, capital, costs, slippage and baseline CRs 7.9% / 19.22% (p.6), Tables I-II, 420 days, ≈8k perturbations, 17.5% / 1.7 pp / $17,200, and 5,957/6,012 (p.7), Table III, the hidden-text aggregates, 16.0% / 3.2 pp / $32.2k and Table IV (p.8), Tables V-VIII and the confidence remark (p.9), the survey (p.10), lessons, limitations and sanitization caveats (p.11), countermeasures, library findings and the sanitizer (p.12), Tables IX-X (p.16), Table XI and Listings 1-2 (p.17), Tables XII-XV (p.18). All four verbatim quotes match exactly (pp.1, 9, 12, 12). The docs/jev and design_v1 cross-references (A4, A7, sentinel table, calibrated decisions) exist.
- Corrections made: (1) "no seeds" was wrong, since the paper states fixed seeds (p.16) and seed = 42 for offline models (p.17); the observation now says there are no repeated runs or CIs; (2) the transfer bullet now records the "deterministic where supported" and seed details (p.17); (3) added two inconsistencies the summary missed: "six general-purpose models" listing seven (p.8), and the tension between 4o-mini's model-level flip rate (46.0%, Table VI) and its system-level ranking among the most vulnerable (76.1% of days impacted, Table VIII; p.9); (4) noted Table XI's unexplained "ChatGPT (4o-mini)" entry for the mapping-model question (p.17).

