# fonseca2026lookahead: Look-Ahead-Freedom as Temporal Non-Interference: A Verifiable Correctness Property for Backtesting and Agentic Trading Pipelines

## Verified metadata
- Title: Look-Ahead-Freedom as Temporal Non-Interference: A Verifiable Correctness Property for Backtesting and Agentic Trading Pipelines [source: arxiv (OAI-PMH, abs page); PDF p.1]
- Authors: Xavier Fonseca [source: arxiv]. Affiliation in the PDF: Academy for AI, Games and Media, Breda University of Applied Sciences, The Netherlands (p.1).
- Venue: arXiv preprint; primary category cs.CR, cross-listed cs.LO, cs.PL, cs.SE, q-fin.PM [source: arxiv OAI]. The arXiv comment reads "Submitted to ACM Transactions on Software Engineering and Methodology" [source: arxiv abs page]. The PDF carries a TOSEM template with placeholder DOI "10.1145/nnnnnnn.nnnnnnn" (p.1), i.e. not an accepted article. No Crossref record could be retrieved (the bibliographic search returned an unrelated top hit; literature/meta/fonseca2026lookahead.crossref_search.json). Treat as a submitted preprint.
- Year: 2026 (v1 submitted 2026-07-06) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2607.04958 / arXiv:2607.04958 (v1)
- Metadata files: literature/meta/fonseca2026lookahead.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json

## PDF
- Path: literature/pdf/fonseca2026lookahead.pdf (arXiv 2607.04958v1; stamp "arXiv:2607.04958v1 [cs.CR] 6 Jul 2026", p.1)
- Text: literature/text/fonseca2026lookahead.txt ; pages: 30

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Look-ahead bias is handled with "construct-specific recipes and empirical detectors, which are sound only channel by channel and certify nothing by their silence" (p.1). There is no property whose satisfaction certifies a pipeline free of look-ahead of every kind (p.2).
- Method / contributions:
  - Identification: fixing a decision epoch, look-ahead-freedom is temporal non-interference over a time-indexed information lattice (p.1-2).
  - A time-indexed pipeline calculus that separates a datum's availability (when it can be read) from its reference time (which instant it describes) (p.2). Constructs include windowing, resampling, joins, `asof` (point-in-time / vintage reads) and `retrieve` (agentic context fetch as of an availability time) (p.7, p.9).
  - Undecidability: where availability may depend on data values, look-ahead-freedom is Π⁰₁-hard; "leakage is recursively enumerable but freedom is not" (p.1-3).
  - A type-and-effect system for the value-independent fragment, proved sound and decidable in linear time (p.3). Re-stamping (relabelling a future value to an admissible time) is identified as the delicate channel (p.3).
- Evaluation:
  - Scaling: log–log slope 1.023, 95% CI [1.021, 1.026] over four orders of magnitude (p.3).
  - Real-data oracle validation on five archetypes from proprietary market data: no checker-accepted pipeline has an oracle-witnessed leak (Table 1, p.24).
  - Adversarial corpus of 53 pipelines (33 leaking, 20 clean): the checker misses 0/33 leaks (rule-of-three upper bound 9.1%), a two-run differential detector misses 18/33 (54.5%), a window-tiling detector misses 33/33; the checker false-positives on 15/20 clean cases, all on opaque value operations (Table 2, p.25).
  - The two-run detector fails through coverage misses (leak at unsampled epochs), perturbation-insensitivity misses, and re-stamp misses (p.24).
- Limitations:
  - Stated by the author: the adversarial corpus measures existence, not frequency, of leaks; post-hoc power 76.7% (p.25, p.27); guarantees apply only to pipelines expressed in or faithfully modelled by the fragment, not to arbitrary Python backtests (p.27); the agentic treatment bounds availability at the retrieval interface but treats the model as a black box, so pretraining look-ahead is outside the certified property (p.27); headline figures use proprietary data; the artifact reproduces qualitative claims on synthetic data (p.1, p.28).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "Look-ahead bias is temporal non-interference." (p.2)
- "a detector reports the leaks it happens to trigger and certifies nothing by its silence." (p.2)
- "the sound checker misses none of 33 planted leaks while a differential two-run detector of the kind used in practice misses 18 (54.5%)" (p.3)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Directly bears on positioning K3 and K5.** Our memory rule "every read enforces available_at < c_t" and our separation of `created_at`/`updated_at`/`filed`/vintage from the week a datum describes (design §2.4, §7) is exactly the availability-vs-reference-time distinction this paper formalizes, including point-in-time reads and agentic retrieval (p.2, p.7, p.9). We cannot present a PIT memory predicate or an availability-time discipline as a contribution.
- **Weakens our planned PIT evidence.** `tests/test_memory_pit.py` (write randomized future records, require bit-identical decisions; positioning K3 item 1) is a two-run differential detector, the class that missed 54.5% of planted leaks here (p.3, p.25). Positioning must describe it as a test, not a certificate; a cheap improvement is to declare each data source's availability term (all ours are value-independent: timestamps, filing dates, vintages) and check them statically in the spirit of this paper.
- Useful framing: the paper itself names the model's pretraining cutoff as the remaining uncertified channel (p.27), which is where our D1–D4 look-ahead diagnostics sit.
- It also connects our trust lattice (taint W > S > T) and time lattice under one information-flow view; if we use "non-interference" language for J = 0 (positioning K2 table, FIDES row), cite this paper for the temporal counterpart.

## BibTeX
```bibtex
@article{fonseca2026lookahead,
  title   = {Look-Ahead-Freedom as Temporal Non-Interference: A Verifiable Correctness Property for Backtesting and Agentic Trading Pipelines},
  author  = {Fonseca, Xavier},
  journal = {arXiv preprint arXiv:2607.04958},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against saved OAI-PMH, abs page and DataCite records (comment "Submitted to ACM Transactions on Software Engineering and Methodology"); PDF stamp; all numbers against the text (p.1-3, Table 1 p.24, Table 2 p.25, limitations p.27, data availability p.28).
- Not verified: acceptance at TOSEM (none found; placeholder DOI in the template).
