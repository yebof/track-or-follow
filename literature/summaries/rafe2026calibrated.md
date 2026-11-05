# rafe2026calibrated: Calibrated Decisions at Scale: Converting Police Crash Narratives into Probabilistic Crash Variables with a System One Model (Jev)

## Verified metadata
- Title: Calibrated Decisions at Scale: Converting Police Crash Narratives into Probabilistic Crash Variables with a System One Model (Jev) [source: arxiv (OAI-PMH, abs page); PDF p.1-2]
- Authors: Amir Rafe, Subasish Das [source: arxiv]. The PDF gives the Ingram School of Engineering, Texas State University (p.2).
- Venue: arXiv preprint, primary category cs.CL [source: arxiv]. The PDF footer reads "Preprint submitted to Elsevier" (e.g., p.43). Crossref search returned unrelated items (literature/meta/rafe2026calibrated.crossref_search.json). No peer-reviewed venue found.
- Year: 2026 (v1 submitted 2026-09-21) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.24052 / arXiv:2609.24052 (v1)
- Metadata files: literature/meta/rafe2026calibrated.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json

## PDF
- Path: literature/pdf/rafe2026calibrated.pdf (arXiv 2609.24052v1; stamp p.1)
- Text: literature/text/rafe2026calibrated.txt ; pages: 48

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem: coding crash narratives at scale is blocked by cost, unverifiable generated text and the lack of a rule for how much output a human must check (p.2-3).
- Method: narrative coding is formulated as gated, typed decisions answered by Jev: presence questions open detail questions, entities are selected from closed option sets (p.2-4). Calibration is audited against coded fields and a human reference set drawn under a stated sampling design; a review budget is defined over flagged records (p.2-4).
- Data: 499,500 Texas narratives screened; 195,857 coded with a 27-question schema; model version jev-1.13.0 (p.2, p.15). 2,416 blinded human judgments (p.2).
- Key results (numbers copied exactly):
  - Against human labels: Jev F1 0.908, κ 0.906; claude-fable-5-1 F1 0.967; gpt-5.6-sol 0.885 (Table 9, p.25).
  - Jev is not calibrated as delivered: Spiegelhalter z −5.3, slope 1.63, ECE 0.0231; out-of-fold Platt or isotonic recalibration lowers pooled ECE from 0.0231 to 0.0069, a factor of 3.3 (p.18, p.25, p.31).
  - Returned probabilities lie on a two-decimal grid and the model "never returned zero in 2.4 million answers", so a resolution-floor bound applies to rare variables (Proposition 1, p.13).
  - "Calibration varies by model rather than by paradigm, so each model must be audited" (p.2).
- Limitations: stated by the authors, the claim concerns output validity, not the absence of option-order or wording effects, which were not tested (p.5); the narrative is the officer's account, not ground truth (p.2).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "Calibration varies by model rather than by paradigm, so each model must be audited." (p.2)
- "The model audited here never returned zero in 2.4 million answers" (p.13)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- Another Jev study graded against labels that exist at labelling time (coded fields and human judgments, p.2), consistent with our reading that no published Jev study grades against market outcomes. But it shows gated typed-question programs and a calibration audit at population scale already exist for Jev (p.2-4), so "typed judgment programs" and "auditing calibration" are not new as such.
- Independent evidence for our wording rules: Jev is not calibrated as delivered (slope 1.63) and its probabilities sit on a two-decimal grid (p.13, p.25); outcome-based recalibration, as in our allocator (design §6), is what fixes level errors.
- How we differ: our judgments are graded by market returns and fused into positions; no human labels (decisions.md, O5 declined).

## BibTeX
```bibtex
@article{rafe2026calibrated,
  title   = {Calibrated Decisions at Scale: Converting Police Crash Narratives into Probabilistic Crash Variables with a System One Model ({Jev})},
  author  = {Rafe, Amir and Das, Subasish},
  journal = {arXiv preprint arXiv:2609.24052},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against OAI-PMH, abs page, DataCite; corpus sizes, model version, Table 9, calibration statistics, Proposition 1 and the untested option-order caveat against the text (p.2, p.5, p.13, p.15, p.18, p.25, p.31).
