# pathade2026attack: Attack Success Rate Is Not a Number: On Measurement Validity in Agentic AI Security Evaluation

## Verified metadata
- Title: Attack Success Rate Is Not a Number: On Measurement Validity in Agentic AI Security Evaluation [source: arxiv (OAI-PMH, abs page); PDF p.1]
- Authors: Chetan Pathade, Prathamesh Pawar, Shubham Patil (independent researchers) [source: arxiv; PDF p.1]
- Venue: arXiv preprint, primary category cs.CR [source: arxiv]; comment "7 pages, 2 tables, 2 Figures". Crossref search returned unrelated items (literature/meta/pathade2026attack.crossref_search.json).
- Year: 2026 (v1 submitted 2026-09-21) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.25173 / arXiv:2609.25173 (v1)
- Metadata files: literature/meta/pathade2026attack.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json

## PDF
- Path: literature/pdf/pathade2026attack.pdf (arXiv 2609.25173v1; stamp p.1)
- Text: literature/text/pathade2026attack.txt ; pages: 7

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Claim: ASR is "a family of metrics parameterized by six design choices": unit of analysis, success oracle, trial count, attacker adaptivity, attacker knowledge of the defense, and binarization of partial success (p.1-3).
- Meta-analysis: 259 agentic-security arXiv papers (2025-02-27 to 2026-09-17) that report a quantitative ASR (p.3). 65.3% report neither variance nor repeated runs (58% [44–71] in a hand-coded sample of 50); 30.9% give any decoding information; 29.7% of judge-using papers validate the judge against humans; 30.5% evaluate an adaptive attacker; 10.8% state an attack budget; 16.2% state attacker knowledge; 2.3% address partial success; 37.8% report benign utility (Table II, p.4; p.5-6).
- Analysis: on a 100-instance benchmark the minimum detectable ASR difference at 80% power is 18.2 percentage points; two defenses 5 points apart are ranked in the wrong order by a single run about 21% of the time (p.1, p.5).
- A ten-item reporting checklist (p.5-6).
- Limitations: arXiv-only corpus, automatic coding with single-coder adjudication, analytical rather than empirical inversion results (p.6).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "on a 100-instance benchmark, the minimum difference in ASR detectable at conventional power is 18.2 percentage points" (p.1)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- C1's reporting must state all six axes: unit (stock-week × family pairs), oracle (portfolio membership change by replay), trials, adaptivity (≤ 64 queries), attacker knowledge (Kerckhoffs, design §3.1), and partial success (demotion ×0.5 vs removal). Adopting the checklist costs nothing and pre-empts a reviewer objection.
- Our planned C1 design (paired 2,800 units, MDE 0.035–0.05) is well above the 100-instance regime criticized here, which positioning can say with this citation.

## BibTeX
```bibtex
@article{pathade2026attack,
  title   = {Attack Success Rate Is Not a Number: On Measurement Validity in Agentic {AI} Security Evaluation},
  author  = {Pathade, Chetan and Pawar, Prathamesh and Patil, Shubham},
  journal = {arXiv preprint arXiv:2609.25173},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against OAI-PMH, abs page, DataCite; all rates, the MDD and inversion numbers, and the limitations against the full 7-page text.
- Quotes were checked against a reading-order pdftotext extraction because the saved -layout text interleaves the two columns.
