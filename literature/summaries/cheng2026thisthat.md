# cheng2026thisthat: this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent

## Verified metadata
- Title: this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent [source: arxiv (OAI-PMH, abs page); PDF p.1]
- Authors: Zehua Cheng, Wei Dai, Jiahao Sun [source: arxiv]. The PDF lists the University of Oxford and FLock.io (p.1).
- Venue: arXiv preprint, primary category cs.CL [source: arxiv]. No comments, no journal reference. Crossref search found no matching record (literature/meta/cheng2026thisthat.crossref_search.json).
- Year: 2026 (v1 submitted 2026-09-20) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.23886 / arXiv:2609.23886 (v1)
- Metadata files: literature/meta/cheng2026thisthat.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json

## PDF
- Path: literature/pdf/cheng2026thisthat.pdf (arXiv 2609.23886v1; stamp p.1)
- Text: literature/text/cheng2026thisthat.txt ; pages: 16

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem: programs that delegate branches to models need "one of n declared options and a number it can threshold", not prose; a frontier-model call costs latency, tokens and a parser (p.1-2).
- Method:
  - A 2B-parameter typed decision model: the answer is read from the hidden state at a designated slot and normalised only over the declared option labels, so no text is generated and malformed output is "unrepresentable"; N questions about one state are answered in one forward pass and are conditionally independent given the input (Eqs. 1-3, p.4-5).
  - A schema-first prompt layout whose prefix is cached across states (p.5); a hybrid linear-attention backbone (p.5).
  - Training on a convex combination of cross-entropy and Brier score, including an unbiased estimator for outcome (not label) supervision (p.6); abstention options inserted during training (p.6).
  - It is adapted from decider-2b (Apache-2.0); NanoJev is an open 0.6B reproduction of the interface whose recorded hosted-Jev cohort is reused (p.4, p.14).
- Key results (numbers copied exactly):
  - On a third party's recorded 68 questions over 17 states: this-that-model-1.0 accuracy 0.941, Brier 0.042, 30.9 ms per question; Jev 0.765, Brier 0.133 (Table 1, p.7).
  - Released 2,250-question benchmark subset: this-that-model-1.0 0.844 vs Jev 0.803, gpt-5.6 0.897 (Table 3, p.12); the authors note their model was trained on these fifteen question shapes and Jev was not (p.12-13).
  - Stochastic-actuator family: accuracy 0.745 against a computed ceiling of 0.746; "the systems that generated the most tokens are the least accurate" (Fig. 2 and Table 2, p.8-9).
  - Context ladder: every capable system is flat and near-perfect when given the part of the state the answer depends on and decays on the whole state (p.9-10).
  - Weak spots: multi-step arithmetic 0.560 vs 0.98–1.00 for hosted models (p.1); search-type questions average 0.574 (p.12-14).
- Availability: weights on Hugging Face under the MIT licence; "an existing caller changes a model name and nothing else"; runs on CUDA, Apple Silicon or CPU (p.14).
- Limitations: stated by the authors, the 68-question comparison is small and its 18-point accuracy gap "rests on 12 questions"; the benchmark comparison is in-distribution for their model (p.12-13).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "The honest summary of the trade is that a frontier model is a better judge and a worse component" (p.3)
- "68 questions is a small cohort, its wording is theirs rather than ours, and the accuracy gap of 18 percentage points rests on 12 questions." (p.13)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Backend choice.** An open, MIT-licensed, non-generative typed decision model with the same interface as Jev exists and runs locally (p.4, p.14), plus the open decider-2b and NanoJev checkpoints it names (p.4). Our only evaluated backend is a generative LLM's letter log-probabilities (gpt-4.1-mini emulator, design §11). A reviewer will ask why no genuinely non-generative typed backend was used, and positioning K1's rule "only a non-generative typed model reads text" is not met by the emulator. A this-that-model or NanoJev row on a subset would make the architecture claim honest at near-zero cost.
- **Independent Jev numbers.** Jev Brier 0.133 on 68 questions and 0.803 on the 2,250-question subset (Tables 1 and 3) are further evidence that Jev is not uniformly calibrated or accurate; consistent with our "never call Jev calibrated" rule.
- **Decomposition evidence.** The context-ladder result (p.9-10) supports decomposing judgments into questions over the part of the state they depend on (our programs, design §5.0); cite it as support, not as ours.

## BibTeX
```bibtex
@article{cheng2026thisthat,
  title   = {this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent},
  author  = {Cheng, Zehua and Dai, Wei and Sun, Jiahao},
  journal = {arXiv preprint arXiv:2609.23886},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against OAI-PMH, abs page, DataCite; Tables 1-3, Eqs. 1-3, availability section and the stated caveats against the text (p.1-14).
- Not verified: the hosted-Jev rows were run by third parties (p.7, p.12); we did not re-run anything.
