# guo2026just: Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures

## Verified metadata
- Title: Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures [source: arxiv (OAI-PMH, abs page); PDF p.1]
- Authors: Ruoqi Guo, Yi Liu, Gelei Deng, Yuekang Li, Lida Zhao, Yutao Wu, Simin Chen, Ying Zhang, Leo Yu Zhang [source: arxiv OAI (first eight) and PDF p.1 (all nine)]. Affiliations on p.1 include Griffith University, NTU, UNSW, Deakin, George Mason and Wake Forest.
- Venue: arXiv preprint (cs.AI, cs.CL, cs.CR) [source: arxiv]. The PDF header reads "Under review as a conference paper at ICLR 2027" (p.1). Not accepted anywhere as of 2026-09-26.
- Year: 2026 (v1 submitted 2026-09-24) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.29429 / arXiv:2609.29429 (v1)
- Metadata files: literature/meta/guo2026just.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json (no match)

## PDF
- Path: literature/pdf/guo2026just.pdf (arXiv 2609.29429v1; stamp p.1)
- Text: literature/text/guo2026just.txt ; pages: 50

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem: whether Jev detects alignment failures (sycophancy, jailbreaks, deception, prompt injection, hallucination and others) has not been measured (p.1-2).
- Method: RLCDAlignBench, 44 benchmarks, ten failure types, 7,193 detection instances from five open 2–7B target models, labelled by each benchmark's scorer and on two benchmarks by humans (p.1, p.3). The question asked is varied separately from the fields of the state; targeted questions are selected on one half and scored on the other (p.3).
- Key results (numbers copied exactly):
  - A single generic Noul question reaches median AUROC 0.886 zero-shot over 31 benchmarks; targeted wording adds +0.006 [−0.004, +0.015] out of sample (p.3).
  - Prompt injection: per-type median AUROC 0.962 (Fig. 1, p.2; Table 9, p.26); InjecAgent 0.988 (p.25); Open-Prompt-Injection 0.231 against its official label, which tracks injected-task accuracy, and 0.975 against the corrected label "the response follows the injection" (p.38-39).
  - Probabilities rank well but do not transfer as thresholds: median ECE 0.168 vs a null of 0.074 (p.3).
  - One pass over 19 judge-scored benchmarks costs $0.30, 63× less than LLM judges (p.1, p.3).
- Limitations: stated by the authors, relational failures need the reference in the state; context fields that encode the label drive large gains (p.3). Our observation: detection targets whether a target model's response exhibits a failure; it is not evaluated as a filter on untrusted input text.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "The probabilities rank well but do not transfer as thresholds" (p.3)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- Prior evidence that a single typed Jev question can detect prompt-injection behaviour (p.2, p.38-39); our D2 guard Noul `instructs_reader` (design §3.3) is a related but different use (screening news items before they are read), so cite this paper as the closest measurement and do not claim S1 injection detection as new.
- Its finding that thresholds do not transfer (median ECE 0.168, p.3) warns against our fixed .5 guard thresholds (design §3.3); report clean false-quarantine rates, as positioning K2 already plans.

## BibTeX
```bibtex
@article{guo2026just,
  title   = {Just Ask {Jev}: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of {AI} Alignment Failures},
  author  = {Guo, Ruoqi and Liu, Yi and Deng, Gelei and Li, Yuekang and Zhao, Lida and Wu, Yutao and Chen, Simin and Zhang, Ying and Zhang, Leo Yu},
  journal = {arXiv preprint arXiv:2609.29429},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against OAI-PMH, abs page, DataCite; the author list against PDF p.1; the AUROC, ECE and cost numbers against the text (p.1-3, p.26, p.38-39).
