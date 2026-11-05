# tang2026typed: Typed Decision Models: An Early Evidence Audit and Evaluation Checklist

## Verified metadata
- Title: Typed Decision Models: An Early Evidence Audit and Evaluation Checklist [source: arxiv; PDF p.1]
- Authors: Lijuan Tang, Yuemeng Zheng [source: arxiv]. PDF p.1 gives Northeastern University.
- Venue: arXiv preprint (cs.CL); comment "25 pages, 8 tables. Corpus: 28 arXiv papers (19-24 September 2026)" [source: arxiv]. No peer-reviewed venue.
- Year: 2026 (v1 submitted 2026-09-26) [source: arxiv]
- DOI / arXiv: arXiv:2609.32160 (v1). The DataCite DOI 10.48550/arXiv.2609.32160 returned HTTP 404 on 2026-09-29 (not yet registered).
- Metadata files: literature/meta/tang2026typed.arxiv.xml

## PDF
- Path: literature/pdf/tang2026typed.pdf (arXiv v1; stamp "arXiv:2609.32160v1 [cs.CL] 26 Sep 2026", p.1)
- Text: literature/text/tang2026typed.txt ; pages: 25

## Summary (grounded; every factual sentence cites a page)
- Problem: what is technically new about the typed-decision-model (TDM) interface, and what does the first week of evidence after Jev's release support (p.1-2)?
- Method: a review of 28 arXiv papers posted 19–24 September 2026 that evaluate hosted Jev or build/evaluate a model with the typed interface; searches on 24 and 25 September (API `all:jev`, abstract searches for "typed decision model", "decision model", "System One model", "TypeSafe"); each paper coded against a fixed extraction schema, with a blind second pass re-checking quantitative claims (p.7-8). Defines a TDM by three properties: support closed at request time, several isolated questions per state, the distribution as the product (p.2). Traces the components to label-probability classification, constrained decoding, discriminative judging and confidence-based deferral (Table 1, p.3).
- Corpus roles (Table 3, p.7): step replacement, controller, judge or screener, measurement instrument (social-science annotation, crash narratives, population simulation), diagnostics (option-name polarity, context flipping, prompt injection), ecosystem. No corpus paper evaluates a TDM against financial-market outcomes (our reading of Tables 3-5, p.7, p.9-10).
- Key results (numbers copied exactly):
  - Finding 1: "the typed readout has not yet shown an independent accuracy advantage"; "Twenty-one of the 27 papers that evaluate a model include no label-probability baseline" (p.8). Where tested: label-token readout AUROC .950 vs .974 for a three-seed typed ensemble, not significant (p.5, p.8); typed head and LM head both .761 macro accuracy (p.8); restricted option-logit readout matches greedy generation whenever the generated answer is valid (p.8).
  - "task training and backbone quality matter more for accuracy than the choice between a typed head and a conventional label-probability readout" (p.6).
  - Finding 2: parity claims often come from saturated tests; Jev trails stronger hosted models on discriminating evaluations (e.g., median 11.6 macro-F1 points on social-science annotation) (p.8).
  - Finding 4: calibration must be evaluated per model and workload (p.11).
  - Jev is not deterministic (4 of 510 labels changed on an identical repeat; at most 1.33% of decisions between two runs) and returns probabilities on a two-decimal grid (p.4).
  - For measurement instruments, "Aggregate accuracy alone is insufficient; systematic error, prevalence-specific calibration, run-to-run stability, and downstream robustness also matter" (p.6).
  - Checklist (Table 6, p.13): B1 label log-probability baseline (compliance 5/26), T3 option-name invariance test (1/27), R2 intervals on headline results (15/27), C1 held-out calibration where probabilities are called calibrated (13/17), among 14 items.
- Limitations (stated): a nine-day, one-hosted-model snapshot; arXiv-only search, so industrial experience and systems without Jev terminology may be missed; several checklist items require judgment; mechanism claims apply only to open TDMs (p.14-15).

## Verbatim quotes
- "In this early literature, the typed readout itself has not shown an independent accuracy advantage over comparable label-probability readouts." (p.1)
- "These results do not establish that all readouts are equivalent, but they show why accuracy gains cannot be attributed to the typed interface without a matched label-probability baseline." (p.8)
- "the review should be read as an early evidence map rather than a settled assessment of the model class." (p.1)

## Relevance to our project
- Our confirmatory implementation (emu) *is* a label-probability readout of a generative LLM. By this audit's Finding 1, nothing in our H2 result can be attributed to "typed" reading as such; the headline must say what the reader is, and the typed framing is an interface claim, not a capability claim.
- Its 28-paper corpus (19–24 Sept.) contains no market-outcome evaluation, which supports a scoped "no arXiv study through 2026-09-24" sentence, but see kathiravelu2026jev (OSF, 2026-09-27).
- Checklist items we can meet cheaply and should report: B1 (emu is itself the label-probability readout; NG is the typed one), T3 option-name invariance, R1 exact versions and dates, R2 intervals, run-to-run stability.

## BibTeX
```bibtex
@article{tang2026typed,
  title   = {Typed Decision Models: An Early Evidence Audit and Evaluation Checklist},
  author  = {Tang, Lijuan and Zheng, Yuemeng},
  journal = {arXiv preprint arXiv:2609.32160},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-29 (novelty review r3). Metadata checked against the arXiv API record; PDF stamp and authors checked on p.1. Every number and quote checked against pdftotext output of the PDF at the cited page.
