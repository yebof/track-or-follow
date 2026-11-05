# xu2026jevout: JevOut: Natural Context Can Flip Decision Models

## Verified metadata
- Title: JevOut: Natural Context Can Flip Decision Models [source: arxiv, datacite]
- Author: Zixiang Xu (single author) [source: arxiv, datacite]. The PDF lists University of Southern California (p.1).
- Venue: arXiv preprint, not peer reviewed [source: arxiv]. The primary category is cs.CL. The DataCite record lists no related published DOI [source: datacite], and a Crossref bibliographic title query on 2026-09-25 returned no matching record. DBLP could not be queried (bot-check page / HTTP 429), so the absence of a DBLP record is unverified. The arXiv comment reads "32 pages, 5 figures, 23 tables", with a homepage and code link [source: arxiv].
- Year: 2026 (v1 submitted 2026-09-24) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.30243 (arXiv DataCite DOI) / arXiv:2609.30243 (v1)
- Metadata files: literature/meta/xu2026jevout.arxiv.xml, literature/meta/xu2026jevout.datacite.json

## PDF
- Path: literature/pdf/xu2026jevout.pdf (arXiv 2609.30243v1). The title on p.1 and in the PDF info matches.
- Text: literature/text/xu2026jevout.txt ; pages: 32

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Dedicated decision models such as Jev return a probability distribution over typed options that downstream software consumes directly (p.1).
  - Real inputs arrive with surrounding context: conversation history, background, retrieved evidence (p.1).
  - The paper asks whether short, fluent, answer-preserving context additions can redirect a correct decision to a specific wrong option fixed in advance. The source, question, choices and gold answer all stay unchanged (p.2).
- Method / architecture:
  - **Targeted-flip setting (p.4).** For each initially correct item, the target is the most probable wrong option (Eq. 1). A context z is a list of (insertion boundary, added sentence) pairs, and every original character is preserved.
  - **Targeted Flip Rate (TFR) (Eq. 2, p.4).** The fraction of initially correct decisions for which at least one accepted context within budget B makes the model select the target. Switching to another wrong option does not count.
  - **Probability-guided context optimization (p.4-5).**
    - Progress is measured by the log-probability margin between the target and its strongest competitor (Eq. 3).
    - A language-model proposer adds one sentence and an insertion boundary at a time (p.4); the proposer is the base Gemma4-12B model (p.6).
    - Mechanical checks are applied, plus a separate target-blind call of the base model that checks coherence and answer preservation (p.4, p.6).
    - Parents are chosen from an archive with a Gibbs/softmax allocation over margins, mixed with uniform allocation (Eq. 4-5, p.4-5).
    - Budget: 16 proposal slots × 4 iterations, at most 64 accepted target evaluations, early-stop threshold 0.7 (p.6).
  - **Learned proposers (p.5, p.9-10).** V1 (transition-trained) and V2 (complete-context-trained) are fine-tuned on development trajectories. They then generate contexts for held-out items without target feedback.
- Data, universe, evaluation period, baselines (for trading papers: tickers/universe size, dates, frequency, costs modeled?, LLMs used):
  - Not a trading paper.
  - **Datasets (p.6).** Seven: MMLU-Pro, SuperGPQA, MuSR, ToMBench, LAR-ECHR, SATA-Bench (multi-answer, reduced to binary Noul units) and BFCL V4 (tool routing). Each has 50 development and 100 held-out items (p.6, p.15).
  - **Targets (p.6; Table 4, p.16).**
    - Jev 1.13.0 (hosted; primary).
    - OpenSourceJev (Qwen3-1.7B Q8).
    - Von 1.0 (395M, non-autoregressive).
    - A plain Qwen scorer that shares OpenSourceJev's weights.
  - **Initially correct populations (p.6).** Jev 508, OpenSourceJev 328, Von 328, plain Qwen 285.
  - **Controls (p.6).** A neutral one-shot addition and a target-aware one-shot addition.
  - **Scale (p.6).** Primary optimization produced 68,912 candidate contexts and 60,685 accepted target evaluations.
  - **Jev cost footprint (p.32).** The formal Jev components used 37,277 logical evaluations and 53.7 million logical input tokens.
- Key results (copy numbers EXACTLY, with page):
  - **Jev headline (p.6).** Redirected on "312 of 508 initially correct decisions", TFR 61.4% (95% Wilson "57.1%–65.5%"). One-shot controls reach "16.9% for one target-aware addition and 2.2% for one neutral addition".
  - **Table 1 (p.7).** TFR (flips):

    | System | TFR | Flips with pt ≥ 0.7 |
    |---|---|---|
    | Jev | 61.4 (312) | 45.1 (229) |
    | OpenSourceJev | 72.6 (238) | 64.3 (211) |
    | Von | 73.2 (240) | 54.0 (177) |
    | Plain Qwen | 64.9 (185) | 60.4 (172) |

  - **High-confidence wrong answers (p.7).** For Jev, "229/508 decisions (45.1%) reach target probability at least 0.7, and 91/508 (17.9%) reach 0.9".
  - **By task (p.7).** Jev TFR ranges "from 40.0% on legal reasoning to 85.7% on tool routing" (Figure 2 shows 45.8 for MMLU-Pro).
  - **Short additions suffice (p.8).** Of the 312 successful Jev contexts, "51.9% contain one addition and 81.1% contain at most two". The median added length is "31 words (IQR: 21–54)".
  - **Budget curve (p.8).** Jev TFR goes "from 40.4% at 16 accepted calls to 51.4% at 32, 57.3% at 48, and 61.4% at 64".
  - **Probability vs label feedback (p.8).** On 140 matched Jev decisions: "probability-only redirects 89/140 decisions (63.6%), compared with 79/140 (56.4%) for label-only", +7.1 points, paired 95% interval "2.1–12.9".
  - **Transfer (p.9).** Frozen contexts transfer in every directed pair, "with rates ranging from 21.1% to 47.8%". Jev→Von is "25.4%".
  - **Learned proposers (p.10).** V2 raises one-generation TFR "from 16.1% to 21.9%" and best-of-four TFR "from 29.1% to 34.3%".
  - **Clean accuracy of Jev on held-out items (Table 3, p.16).** MMLU-Pro 83.0, SuperGPQA 46.0, MuSR 61.0, ToMBench 72.0, LAR-ECHR 80.0, SATA 82.3 [42.0 exact match], BFCL V4 70.0.
- Limitations:
  - **Stated by the author (p.15, p.32):**
    - Thresholds (0.7, 0.9) apply to reported probabilities and are "not assumptions of common calibration across models" (p.15).
    - The study conditions on initially correct decisions and measures a bounded optimizer's discoveries.
    - Broader scales, multilingual contexts, continuous actions and naturally occurring deployment streams are left for future work (p.32).
    - The LAR-ECHR splits are item-disjoint but not case-disjoint (p.15).
  - **Our observation:**
    - The additions are adversarially optimized with access to target probabilities (up to 64 evaluations per item). Natural financial news is not optimized against our agents, so TFR is a worst-case, white-box-probability measure, not a natural-noise rate.
    - Answer preservation is checked by an automatic LLM checker, "an operational approximation to the semantic constraints" (p.14), not by humans.
    - No finance datasets are included.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "short, ordinary-looking context can shift a correct choice to a high-confidence wrong one." (p.1)
- "returning a structured probability distribution does not make a choice dependable." (p.3)
- "Models used to drive downstream actions must distinguish context that changes what should be decided from context that merely changes which option they favor." (p.11)
- "tracking context provenance, evaluating invariance under answer-preserving additions, abstaining when branch probabilities are unstable, and separating untrusted context from decision criteria." (p.32)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Core robustness threat for long, noisy financial contexts.** Our fundamentals, news, policy and macro agents will pass long contexts (filings excerpts, multiple headlines, policy text) to Jev. JevOut shows that one or two fluent, on-topic sentences (median 31 words, p.8) can move jev-1.13.0 from a correct choice to a preselected wrong one in 61.4% of cases under optimization (p.6). In 45.1% of cases the wrong option reaches probability ≥ 0.7 (p.7). High reported confidence therefore cannot by itself justify a trade.
- **Informs design decisions:**
  - (i) Context provenance and minimal, structured states: feed Jev extracted, source-tagged facts rather than raw concatenated text.
  - (ii) Invariance tests: re-ask with irrelevant or neutral context added, removed or reordered, and abstain when the decision is unstable. This is also recommended on p.32.
  - (iii) Multi-agent aggregation and cross-source agreement as a partial defense, which is our hypothesis to test, not a paper claim.
  - (iv) Keeping decision criteria (rubrics) separate from untrusted context.
- **Evaluation choices.** We can report a "context-perturbation flip rate" for our agents: neutral and topical-but-irrelevant additions from real news, not optimized attacks. We should cite TFR as the adversarial upper bound. The neutral one-shot rate (2.2% on Jev, p.6) is a useful reference point for benign noise.
- **Cost context.** The Jev footprint of 53.7M logical input tokens (p.32) shows large-scale Jev evaluation is feasible. Cost claims about Jev pricing must still come from docs/jev.
- **How we differ.** JevOut studies single-decision benchmark items with fixed gold answers. Our trading decisions have no gold label at decision time and are evaluated by realized returns. The robustness of aggregated, multi-agent portfolio decisions to such context manipulation is untested there.

## BibTeX
```bibtex
@misc{xu2026jevout,
  author        = {Xu, Zixiang},
  title         = {{JevOut}: Natural Context Can Flip Decision Models},
  year          = {2026},
  eprint        = {2609.30243},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2609.30243},
  howpublished  = {arXiv preprint arXiv:2609.30243},
  url           = {https://arxiv.org/abs/2609.30243}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- Metadata checked:
  - Title, single author, arXiv ID, v1 date (2026-09-24), primary category cs.CL and the comment ("32 pages, 5 figures, 23 tables") against the saved arXiv API and DataCite records.
  - DataCite was re-queried live and confirms them, with no related published DOI.
  - The PDF stamp reads arXiv:2609.30243v1 [cs.CL] 24 Sep 2026. The PDF info title and author match, and the page count is 32.
  - A Crossref title query found no record. DBLP and Semantic Scholar/OpenAlex could not be queried (bot-check / rate limits).
- Content checked against the PDF text (32 pages):
  - All headline and table numbers: 312/508 = 61.4% with Wilson 57.1–65.5; one-shot controls 16.9%/2.2%; all four rows of Table 1; 229/508 and 91/508; the task range 40.0–85.7% and MMLU-Pro 45.8.
  - 51.9%/81.1% and the 31-word median (IQR 21–54); the budget curve 40.4/51.4/57.3/61.4; 89/140 vs 79/140 (+7.1, CI 2.1–12.9); transfer 21.1–47.8% and Jev→Von 25.4%; V2 16.1→21.9 and 29.1→34.3.
  - Table 3 clean accuracies, Table 4 targets, the populations 508/328/328/285, the 68,912/60,685 scale, and the 37,277 evaluations / 53.7M tokens.
  - Budget settings, the limitations (p.15, p.32) and all four verbatim quotes (p.1, p.3, p.11, p.32). Each matches exactly.
- Corrections:
  - The venue line asserted "No DBLP … record exists", but DBLP was never successfully queried. This is now stated as unverified.
  - The proposer model (Gemma4-12B) was cited under p.4-5 but is named on p.6. The page is now cited.
