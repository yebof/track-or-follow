# kirtac2026financial: Financial Language Models as Applied Artificial Intelligence Systems for News-Based Trading under Market Frictions

## Verified metadata
- Title: Financial Language Models as Applied Artificial Intelligence Systems for News-Based Trading under Market Frictions [source: arxiv (OAI-PMH, abs page); crossref (SSRN posted-content record); PDF p.1]
- Authors: Kemal Kirtac [source: arxiv]. Affiliations on p.1: Department of Computer Science, University College London; Department of Computer Science, University of Warwick.
- Venue: arXiv preprint (cs.CL, cs.LG, q-fin.TR) [source: arxiv]. The arXiv comment and the PDF (p.1) state "Revise and resubmit at Engineering Applications of Artificial Intelligence"; not accepted. Crossref lists the same title as SSRN posted content, DOI 10.2139/ssrn.7213792 (literature/meta/kirtac2026financial.crossref_search.json).
- Year: 2026 (arXiv v1 submitted 2026-09-20) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.23703 / arXiv:2609.23703 (v1); SSRN DOI 10.2139/ssrn.7213792
- Metadata files: literature/meta/kirtac2026financial.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json

## PDF
- Path: literature/pdf/kirtac2026financial.pdf (arXiv 2609.23703v1; stamp "arXiv:2609.23703v1 [cs.CL] 20 Sep 2026", p.1)
- Text: literature/text/kirtac2026financial.txt ; pages: 47

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem: research lacks a protocol that "jointly tests financial language-model outputs under event-time observability, probability calibration, execution timing, transaction costs, liquidity constraints, capacity limits, operational diagnostics, and statistical inference" (p.1).
- Method (MFAST): a pipeline from timestamped news to trades: firm matching, temporal eligibility, a five-day cosine-similarity novelty screen, calibration, signal ranking, cost-aware value-weighted portfolios with a 10% daily dollar-volume participation cap, factor adjustment, multiple-testing corrections, compute diagnostics and a public-data replication arm (p.4-5, p.10, p.30). Six stated requirements: temporal observability, semantic reliability, implementability, robustness, computational feasibility, reproducibility (p.4).
- Data: Refinitiv News Analytics linked to CRSP; 3,129,924 US news items from January 2010 to January 2026, reduced to 973,481 tradable items for 3,452 firms (p.4-5). Models: LLaMA-3, OPT, RoBERTa, BERT, FinBERT and the Loughran–McDonald dictionary (p.5). The primary out-of-sample test (June 2024–January 2026) uses news after the latest disclosed LLaMA-3 family data-freshness cutoff and checkpoint release (p.5, p.29).
- Labels and metrics: classification of the sign of the three-day excess return, with accuracy, AUC, Brier score and ECE; next-day return regressions with firm and date fixed effects (p.26).
- Training (added 2026-09-29 from literature/text/kirtac2026financial.txt, for the round-1 review of paper v2): "All transformer models are fine-tuned as binary classifiers on the training split" (p.24); the label is "a binary return-relevant label based on the sign of the associated stock’s cumulative three-day excess return after the article becomes observable" (p.18); articles from January 1, 2010 to December 31, 2023 form the supervised training set (p.18); the model input "concatenates the headline, lead paragraph, and main body until the model-specific token limit is reached" (p.20). So the LLaMA-3 probabilities are supervised on return-sign labels and read the article body, not only headlines (our observation from these pages).
- Key results (numbers copied exactly): LLaMA-3 accuracy 0.787, AUC 0.846, Brier 0.151, ECE 0.032; the dictionary 0.503, 0.512, 0.249, 0.143 (Table 7, p.29). Predictive coefficient 0.312 (t = 6.44) for LLaMA-3 vs 0.049 (t = 1.31) for the dictionary, N = 190,236 (Table 8, p.30). Net long-short Sharpe under 5 bp costs 2.85 for LLaMA-3 vs 0.68 for the dictionary (Table 9, p.30).
- Limitations: stated by the author, US English equities only; not a full order-book simulator; pretraining exposure "cannot be eliminated entirely"; the public replication validates the pipeline, not the proprietary results (p.36).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "The Brier and calibration results matter because portfolio construction uses continuous probabilities rather than only binary labels." (p.29)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Pre-empts much of C2 and part of K5.** Item-level language-model probabilities for news are validated against short-window excess returns with Brier and ECE, on post-cutoff news, with a novelty screen, costs, capacity caps and multiple-testing corrections (p.4-5, p.26-30). Our C2 (N1 E[effect] vs AR[0,+1], with AUC/Brier/ECE and a placebo, design §13.1) must be framed as the same kind of test for a typed-judgment interface, not as a new kind of validity evidence; and "delayed" is the wrong word for a one- to three-day label.
- For K5, "event-time observability + calibration + execution timing + costs + capacity + statistical inference" as an integrated protocol is claimed here as the contribution (p.1, p.4-5). Our protocol differences are narrower: pre-registered confirmatory endpoints with Holm, a look-ahead battery for a black-box typed model with no documented cutoff, and a trust-partitioned decision layer.
- Useful baselines: decoder LLM sentiment probabilities beat FinBERT and the LM dictionary on the same labels (Table 7), which supports keeping LM dictionary and FinBERT as the weak text baselines and an LLM-probability headline baseline (our HS).

## BibTeX
```bibtex
@article{kirtac2026financial,
  title   = {Financial Language Models as Applied Artificial Intelligence Systems for News-Based Trading under Market Frictions},
  author  = {Kirtac, Kemal},
  journal = {arXiv preprint arXiv:2609.23703},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against OAI-PMH, abs page, DataCite and the Crossref SSRN record; sample sizes, test window, labels, Tables 7-9 and limitations against the text (p.1-5, p.10, p.26, p.29-30, p.36).
