# boustanifar2026same: Same Text, Different Numbers: The Divergence of LLM-Based Measures

## Verified metadata
- Title: Same Text, Different Numbers: The Divergence of LLM-Based Measures [source: arxiv; datacite; PDF p.1]
- Authors: Hamid Boustanifar, Sasan Mansouri [source: arxiv]. PDF p.1 footnotes: EDHEC Business School; University of Groningen.
- Venue: arXiv preprint (cs.AI); comment "86 pages, including an online appendix" [source: arxiv]. No peer-reviewed venue.
- Year: 2026 (v1 submitted 2026-09-25) [source: arxiv]
- DOI / arXiv: 10.48550/arXiv.2609.31013 / arXiv:2609.31013 (v1)
- Metadata files: literature/meta/boustanifar2026same.arxiv.xml, literature/meta/boustanifar2026same.datacite.json

## PDF
- Path: literature/pdf/boustanifar2026same.pdf (arXiv v1)
- Text: literature/text/boustanifar2026same.txt ; pages: 86 (read: p.1-6)

## Summary (grounded)
- Problem: whether LLM-based textual measures used in accounting and finance are invariant to the choice of model; measurement reliability is treated as logically prior to validity (p.1-3).
- Data and method: the Q&A sections of 1,946 earnings calls of S&P 500 firms in 2024; seven models from seven providers (OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, Alibaba) with identical construct definitions, scales and output instructions; thirteen measures of twelve constructs (sentiment, uncertainty, management clarity, climate and political risk, answer specificity, corporate culture and five components, plus a rule-based specificity ratio) (p.2-3).
- Key results (numbers copied exactly):
  - Mean pairwise rank correlation between providers 0.52; sentiment 0.85, innovation 0.76, integrity 0.32, rule-based specificity 0.23 (p.3).
  - Variance components: transcript 34.4%, provider 33.4%, provider-by-transcript 32.2%; transcript share 74% for sentiment (p.4).
  - Self-reported confidence does not identify more dependable scores (p.4).
  - Cross-model disagreement is not associated with later analyst dispersion, forecast errors, spreads, volume or return variance (p.5).
  - Correlations with established dictionary measures are positive but generally low, at most about 0.55 (p.5).
  - Model choice changes the sign, magnitude or significance of downstream coefficients on announcement returns (p.1, p.5-6).
- Limitations: our observation — one year of calls, one text type, generative models with written outputs (not typed readouts); we read p.1-6 only.

## Verbatim quotes
- "LLM-generated variables should therefore be treated as model-contingent measurements and validated across providers." (p.1)
- "Cross-model disagreement is therefore informative regardless of which model, if any, is closest to the ground truth" (p.3)

## Relevance to our project
- Supports reporting cross-implementation agreement and per-implementation validity as a measurement issue (our emu/VERB/NG agreement and per-implementation H2 rows). Our emu and VERB share one LLM, so only NG is a second model in their sense; their average cross-provider rank correlation (0.52; 0.85 for sentiment) is the context for our reported agreement values.

## BibTeX
```bibtex
@article{boustanifar2026same,
  title   = {Same Text, Different Numbers: The Divergence of {LLM}-Based Measures},
  author  = {Boustanifar, Hamid and Mansouri, Sasan},
  journal = {arXiv preprint arXiv:2609.31013},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-29 (novelty review r3). Metadata against the arXiv API record and DataCite; numbers and quotes against pdftotext output of p.1-6 (the remaining 80 pages were not read and nothing from them is claimed).
