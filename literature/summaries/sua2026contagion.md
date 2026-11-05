# sua2026contagion: Contagion on the Trading Floor: How Adversarial Signals Spread in Multi-Agent Trading Systems
## Verified metadata
- Title: Contagion on the Trading Floor: How Adversarial Signals Spread in Multi-Agent Trading Systems [source: crossref (the title contains non-breaking spaces); arxiv; openalex; PDF p.1]
- Authors: Qi Rong Sua, Junhao Dong, Nguyen Duc Thai, Yuqing Wen, Cheston Tan, Yew-Soon Ong [source: crossref; arxiv; openalex]
  - Affiliations on PDF p.1: Nanyang Technological University, National University of Singapore, and CFAR/IHPC A*STAR.
- Venue: Machine Learning and Knowledge Discovery in Databases. Research Track (ECML PKDD 2026), Lecture Notes in Computer Science, Springer Nature Switzerland, pp. 206–223 [source: crossref (type book-chapter, container titles "Lecture Notes in Computer Science" and "Machine Learning and Knowledge Discovery in Databases. Research Track", pages 206-223, ISBN print 9783032376725 / electronic 9783032376732); arxiv (comment "Published in ECML PKDD 2026"; journal-ref "Machine Learning and Knowledge Discovery in Databases. Research Track, ECML PKDD 2026, LNCS 16946, Springer (2027) 206-223")]
  - The LNCS volume number 16946 appears only in the author-supplied arXiv journal-ref. The Crossref record has no volume field, and the Springer chapter page returned a bot-check page, so the volume is not independently confirmed and is left out of the BibTeX.
  - DBLP was unreachable (bot-protection page).
  - The Crossref record of the whole volume (DOI 10.1007/978-3-032-37673-2, fetched 2026-09-26, saved as `meta/sua2026contagion.crossref_book.json`) confirms the book title, the LNCS series, the ISBNs and the editors (Michelangelo Ceci, Dragi Kocev, Anna Monreale, Celine Vens, Peter Flach, Giuseppe Manco, Myra Spiliopoulou, Elio Masciari). It too has no volume field and gives the book's publication year as 2027.
- Year: 2026. Crossref gives published-online 2026-09-05 and published-print 2027. [source: crossref]
- DOI / arXiv: DOI 10.1007/978-3-032-37673-2_12; arXiv:2609.19789 (v1 17 Sep 2026)
- Metadata files:
  - literature/meta/sua2026contagion.crossref.json
  - literature/meta/sua2026contagion.arxiv_abs.html
  - literature/meta/sua2026contagion.openalex.json (W7208758797)
  - literature/meta/sua2026contagion.crossref_book.json (volume-level Crossref record, added during verification)
## PDF
- Path: literature/pdf/sua2026contagion.pdf. This is the arXiv v1 author version (https://arxiv.org/pdf/2609.19789), not the Springer typeset chapter; the title matches.
- Text: literature/text/sua2026contagion.txt ; pages: 25 (main paper p.1-18, supplementary material p.19-25)
  - Page numbers below are arXiv PDF pages, not LNCS pages 206-223.
## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - How vulnerable are LLM multi-agent trading stacks to "black-box, input-only attacks that enter solely via admissible social-media feeds" (p.1-2)?
  - The authors choose social media because it is high-volume, weakly curated and easily seeded with misinformation, yet often too valuable to remove (p.2).
- Method / architecture:
  - GMATS (Generic Multi-Agent Trading System), a pipeline of analysts → coordinators → controllers → executors (p.3-4). Its messages are typed, on a shared bus, point-in-time, and replayable (p.3-5).
    - Theorem 1: any pipeline with typed message passing, PIT discipline and stage separation can be compiled into GMATS with an identical executed-order sequence (p.5; proof p.19-21).
    - TradingAgents, TradingGroup, ContestTrade and QuantAgent are shown to compile (p.5, p.23-24).
  - Attackers (p.7): budget-constrained LLM post generators on a single target asset with daily budget B.
    - The random attacker posts random-sentiment content.
    - The always-negative attacker posts only strongly bearish content, like a "short-and-distort" campaign.
  - Admission assumption: attacker posts are assumed admissible. Strength is set through p_poison, the share of the social analyst's consumed top-k posts that come from the attacker (eq. 1, p.6; p.7).
  - Metrics:
    - Belief Signal Shift, BSS = mean over (t, i) of f_attack − f_clean for the forecast f, at the analyst and coordinator layers (eq. 2, p.6).
    - Attack−clean deltas ∆CR, ∆Vol and ∆Sharpe, following the FinSABER protocol with annualization A = 252 (p.6-7).
  - Topologies:
    - baseline: 3 analysts → 1 coordinator;
    - wide coordinator: two coordinators with cross-talk;
    - single agent;
    - two defense-oriented coordinators: wide bull–bear, and pairwise triangulation (a meta-coordinator that down-weights outliers) (p.9-11).
- Data, universe, evaluation period, baselines:
  - Instance: three analysts (social, fundamentals, news), one coordinator, one risk controller and one executor (p.7).
  - Main model: gpt-4o-mini; replication with Qwen2.5-7B-Instruct (p.7, p.13).
  - Trading window: a single ticker (AAPL), 01 Nov 2021–01 Dec 2021, "for controlled mechanistic analysis" (p.7-8).
  - Social corpus: tweets and Reddit posts for 7 large-cap tech firms, 30 Sept 2021–29 Sept 2022 (p.8). Reddit comes from Pushshift dumps and Twitter from a public Kaggle dataset (p.16).
  - Agents operate strictly point-in-time (p.8).
  - Evaluation: every table averages 5 runs (p.8, p.10, p.12, p.14). Returns are defined as "net (after-cost)" (p.6), but no cost model is specified in the text.
  - An additional TradingAgents baseline appears in the supplement (p.24).
- Key results (copy numbers EXACTLY, with page):
  - Experiment A, fixed GMATS, always-negative attacker at p_poison 15/30/45/60/75% (Table 1, p.8):

    | Metric | 15% | 30% | 45% | 60% | 75% |
    |---|---|---|---|---|---|
    | BSS_analyst | −0.777 | −1.084 | −1.204 | −1.291 | −1.413 |
    | BSS_coord | −0.409 | −0.544 | −0.544 | −0.642 | −0.594 |
    | ∆CR | −0.006 | −0.032 | −0.036 | −0.043 | −0.044 |
    | ∆Sharpe | +0.864 | −5.501 | −5.970 | −6.167 | −6.060 |

    - Under the random attacker, ∆CR stays between +0.009 and +0.023 and ∆Sharpe stays positive (p.8).
    - The authors report "a pronounced regime change around 30%–45% ingestion" (p.9).
  - Experiment B, topology (Table 2, p.10):
    - Single agent: ∆Sharpe −7.153, −7.150, −7.274, −6.773, −8.670; ∆CR −0.052 to −0.062.
    - Wide coordinator: BSS_fuse −0.281 to −0.499; ∆CR −0.003 to −0.024.
  - Experiment C, defense coordinators (Table 3, p.12):
    - Wide bull–bear: BSS_fuse −0.403, −0.568, −0.559, −0.552, −0.590; ∆Sharpe +1.871, +2.524, +5.513, +3.234, +0.252.
    - Pairwise triangulation: BSS_fuse −0.314, −0.389, −0.405, −0.403, −0.397; ∆Sharpe +2.347, +1.586, +1.069, +0.359, +1.713.
  - Qwen2.5-7B replication (Table 4, p.14):
    - Baseline, always-negative: BSS_analyst −1.159 to −1.694; ∆CR −0.015 to −0.061.
    - Single agent: ∆Sharpe −6.242 to −7.546 (the prose says "approximately −6.2 and −7.5", p.14).
    - Triangulation: BSS_fuse −0.247 to −0.341, with positive ∆Sharpe throughout.
  - TradingAgents under the always-negative attacker (supplement Table 2, p.24): BSS_analyst −0.788 → −1.369, BSS_fuse −0.247 → −0.152, ∆Sharpe +7.764 … +1.942. The authors read this as "risk-damping under attack rather than as evidence that poisoning improves profitability" (p.24-25).
- Limitations:
  - As stated by the authors:
    - The attackers are scripted, input-only, non-adaptive and single-channel (social), evaluated under offline replay (p.15).
    - The study covers a narrow set of assets and horizons (p.15).
    - Sharpe is annualized and sensitive to short windows, so absolute Sharpe levels are not standalone claims. Conclusions rest on deltas cross-checked with ∆CR and ∆Vol (p.9).
    - Magnitudes are model-dependent (p.14).
    - Admission of attacker posts is assumed, not modeled (p.7).
  - Our observations:
    - **Tiny sample.** One ticker and one month (p.8), roughly 21 daily decisions (our estimate). There are no confidence intervals or significance tests; the text reports only 5-run averages.
    - **Unstable Sharpe deltas.** ∆Sharpe changes sign while ∆CR moves only a few percent. Examples: always-negative +0.864 at 15% vs −5.501 at 30% (Table 1, p.8); wide coordinator +3.235 at 15% vs −3.113 at 60% (Table 2, p.10). The positive ∆Sharpe of the "defense" coordinators under attack (Table 3, p.12) is therefore weak evidence of robustness. The authors themselves read the similar TradingAgents pattern as risk damping (p.24-25).
    - **BSS has no stated unit.** The forecast f_{i,t} has no stated unit or scale (p.6). The Fig. 4 traces show coordinator µ values such as −0.25 (p.13), which suggests f is the coordinator's µ, but this is not stated.
    - **Typing is the envelope only.** GMATS "typed messages" wrap free-form natural-language payloads "verbatim" (supp. p.22). The schema therefore types the envelope, not the semantics.
    - **No code link.** The abstract says code and data "are available in this repository", but the arXiv text has no URL (p.1).
## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "We study the vulnerability of LLM trading stacks to black-box, input-only attacks that enter solely via admissible social-media feeds." (p.1)
- "the negative bias propagates through the stack rather than being fully absorbed." (p.8)
- "absolute Sharpe levels in our one-month study are not intended as standalone performance claims." (p.9)
- "We deliberately restricted our study to scripted, input-only attackers that are non-adaptive, single-channel (social), and evaluated under offline replay." (p.15)
## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Same threat model as C1, peer-reviewed.** Their adversary is black-box and only injects admissible Web content (p.1, p.7). That matches our C1 threat model: attacks A1–A8 enter only via news/policy items (docs/design/design_v1.md §13.1). This is the natural peer-reviewed citation for "Web-text-only poisoning of multi-agent trading stacks".
- **Damping vs bounding.**
  - Their coordinator designs partially damp contagion. In Experiment A, BSS_coord is about 0.42–0.53 of BSS_analyst (our arithmetic from Table 1, p.8).
  - Their damage is not bounded: ∆Sharpe reaches about −6 once p_poison ≥ 30% (p.8-9).
  - Our design adds a hard guarantee: the band guard, the text cap κ_t, and the ×0.5 single-cluster demotion cap with ≥ 2 corroborating items for exclusion (design_v1 §2).
  - Their always-negative campaign is the case our demotion families (A2-neg, A6-neg, A8, "demotion ASR in D-sized") measure. We should present our cap as turning partial damping into a closed-form limit.
- **A layer-wise contagion metric we can adopt.** BSS at the analyst and coordinator layers (p.6) maps onto our C1 secondaries:
  - item-level ΔE[effect];
  - stock-level Δ combined rank;
  - then book entry.

  Reporting a layer-by-layer shift (S1 item answer → agent feature → allocator logit → selection) would make our results directly comparable. It would also tie into our explanation cards, which localize the planted item (S4). Their Fig. 4 traces do this only qualitatively (p.13).
- **Evaluation design lesson.** Their one-month Sharpe deltas swing sign (p.8, p.10), and the authors warn about short-window Sharpe (p.9). This supports our choices:
  - a binary portfolio-level success over 400 stock-weeks × 7 families, with a week-clustered bootstrap, rather than Sharpe deltas;
  - Sharpe/IR only as estimation over 115 weekly decisions (design_v1 §1, §13.1).
- **Typed bottleneck vs GMATS typing.**
  - GMATS's schema safety concerns routing. Payloads stay free-form text (supp. p.22), so contaminated prose flows to coordinators unchanged.
  - In our system, only S1 reads untrusted text, and what flows onward are typed Choice/Score/Noul answers (docs/jev/docs_introduction.txt) and code features (design_v1 §2).
  - We should cite GMATS as the prior formalism and state that our typing is a semantic bottleneck, not just a message envelope. Whether this lowers ASR is what C1 tests.
- **Adaptive attacker is their open problem.** Their future work proposes an RL attacker that steers the post generator with a contagion-aware reward (eq. 7, p.15). Our adaptive arm (black-box search on E[effect], ≤ 64 queries) partially addresses this gap, which we can state in related work.
- **How we differ:** they test a single ticker and month on gpt-4o-mini/Qwen2.5-7B with daily decisions and social media only (p.7-8, p.13). We use a PIT top-100 US-equity universe with weekly decisions over about two years, and news plus policy text as the untrusted channel.
## BibTeX
```bibtex
@inproceedings{sua2026contagion,
  author    = {Sua, Qi Rong and Dong, Junhao and Thai, Nguyen Duc and Wen, Yuqing and Tan, Cheston and Ong, Yew-Soon},
  title     = {Contagion on the Trading Floor: How Adversarial Signals Spread in Multi-Agent Trading Systems},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  editor    = {Ceci, Michelangelo and Kocev, Dragi and Monreale, Anna and Vens, Celine and Flach, Peter and Manco, Giuseppe and Spiliopoulou, Myra and Masciari, Elio},
  series    = {Lecture Notes in Computer Science},
  pages     = {206--223},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-37673-2_12}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check pass; files touched 2026-09-26 local time).
- Metadata checked against saved raw records: `meta/sua2026contagion.crossref.json` (chapter DOI 10.1007/978-3-032-37673-2_12, type book-chapter, the two container titles, pages 206-223, ISBNs, published-online 2026-09-05, published-print 2027, six authors in order, no volume), `meta/sua2026contagion.arxiv_abs.html` (arXiv 2609.19789 v1, 17 Sep 2026; the comment and journal-ref quoted above; the DOI link), `meta/sua2026contagion.openalex.json` (W7208758797, DOI match, 2026). New during verification: the volume-level Crossref record (`meta/sua2026contagion.crossref_book.json`; no volume number, editors listed, year 2027), and a live re-fetch of the arXiv abstract page on 2026-09-26, byte-identical to the saved `meta/sua2026contagion.arxiv_abs.html` (still v1, same journal-ref), so it was not kept. The LNCS volume 16946 is still attested only by the author-supplied journal-ref, so it stays out of the BibTeX. PDF p.1 matches the title, authors, affiliations and the arXiv stamp "arXiv:2609.19789v1 [cs.AI] 17 Sep 2026".
- Content checked against `text/sua2026contagion.txt` page by page: the abstract and research question (pp.1-2), GMATS definition and invariants (pp.3-5), Theorem 1 and Corollary 1 (p.5), eqs. (1)-(7) (pp.6-7, 15), the attacker taxonomy and admission assumption (p.7), setup, window, corpus and PIT (pp.7-8), every always-negative cell of Table 1 plus the random-attacker ranges (p.8), the regime-change and Sharpe caveats (p.9), Table 2 (p.10), Table 3 (p.12), the Fig. 4 mu values (p.13), Table 4 and the Qwen prose (pp.13-14), limitations (p.15), data provenance (p.16), the proof and compilation recipe ("payloads verbatim", pp.19-22), and supplementary Table 2 with the risk-damping sentence (pp.24-25). "net (after-cost)" is the only mention of costs in the text. PyMuPDF finds no hyperlink on p.1, which confirms the missing code URL. The BSS_coord / BSS_analyst ratio of 0.42-0.53 was recomputed and holds. All four verbatim quotes match exactly (pp.1, 8, 9, 15).
- Corrections made: (1) Qwen single-agent ∆Sharpe range was wrong: "−6.369 to −7.546" -> "−6.242 to −7.546" (Table 4, p.14; −6.242 at 45% is the least negative value); (2) added the editors to the BibTeX from the verified volume-level Crossref record, and documented that record and the 2027 book year; (3) added the new metadata files to the file list.

