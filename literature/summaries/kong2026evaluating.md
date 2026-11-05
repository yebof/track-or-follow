# kong2026evaluating: Evaluating LLMs in Finance Requires Explicit Bias Consideration

## Verified metadata
- Title: Evaluating LLMs in Finance Requires Explicit Bias Consideration [source: arxiv API; arxiv abs page; PDF p.1]
- Authors: Yaxuan Kong, Hoyoung Lee, Yoontae Hwang, Alejandro Lopez-Lira, Bradford Levy, Dhagash Mehta, Qingsong Wen, Chanyeol Choi, Yongjae Lee, Stefan Zohren [source: arxiv]. The PDF marks the first three as equal contribution. Affiliations include University of Oxford, UNIST, Pusan National University, University of Florida, Chicago Booth, BlackRock, Squirrel Ai Learning and LinqAlpha (p.1).
- Venue: arXiv preprint. It is a position paper, and the PDF footer says "Preprint." (p.1). There is no journal-ref, and a Crossref title search (2026-09-25, saved as literature/meta/kong2026evaluating.crossref_search.json) returned no matching record. The Semantic Scholar record lists venue "arXiv.org" with DBLP key journals/corr/abs-2602-14233 (CoRR only). [source: arxiv; s2; crossref search]
- Year: 2026 (v1 submitted 2026-02-15; only one version) [source: arxiv]
- DOI / arXiv: none / arXiv:2602.14233 (v1; primary category cs.LG, also cs.AI and q-fin.CP)
- Metadata files: literature/meta/kong2026evaluating.arxiv.xml, literature/meta/kong2026evaluating.arxiv_abs.html, literature/meta/kong2026evaluating.s2.json, literature/meta/kong2026evaluating.crossref_search.json

## PDF
- Path: literature/pdf/kong2026evaluating.pdf (arXiv v1, downloaded from arxiv.org/pdf/2602.14233)
- Text: literature/text/kong2026evaluating.txt ; pages: 21 (main text pp.1-8, references pp.9-12, appendices A-C pp.13-21). The checklist template in Figures 10-11 (pp.20-21) is embedded as an image and is not in the extracted text. Its structure is described in text on p.19, and Figure 4 (p.6) lists the check items.

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Financial LLM evaluation practice "has not kept up". Finance-specific biases "can inflate performance, contaminate backtests, and make reported results useless for any deployment claim" (p.1).
  - LLM-for-finance papers in major ML/NLP venues rose "from 36 to 250, a 594% increase (6.9×)" between 2023 and 2025 (p.1, Fig. 1).
- Method / architecture (position paper; literature review + user study + framework):
  - Five "sins" (p.2, Table 1 p.4):
    1. Look-ahead bias: information not available at decision time, via model weights or external context.
    2. Survivorship bias: the universe silently drops firms that merge, delist or fail.
    3. Narrative bias: a clean rationale not supported by the evidence available at the time.
    4. Objective bias: training rewards confident completion over calibrated uncertainty and refusal.
    5. Cost bias: gross performance assuming zero execution friction and operating cost.
  - Look-ahead issues (pp.2-3):
    - (1) Parametric knowledge leakage. "the mere existence of a stated knowledge cutoff does not guarantee the exclusion of post-cutoff information" (p.2).
    - (2) External knowledge leakage via RAG, including retrievers and rankers using signals "not observable at time t" and documents edited after publication (p.3).
  - Survivorship channels (pp.3-4): ex-post question generation, intrinsic model bias toward surviving or large firms, skewed tail distributions, and distorted benchmarks with E[B(M)|D_survive] > E[B(M)|D_true].
  - Narrative bias channels (pp.4-5): information processing, e.g., framing shifts in summaries; and inference. CoT "should not be treated as transparent traces of the model’s internal decision process" (p.5).
  - Objective bias (p.5): incentive-induced miscalibration and alignment mismatch.
  - Cost bias (p.5): monetary cost, with R_net = R_gross − C_trans − C_M; and latency, where orders execute at P_{t+Δgen} instead of P_t.
  - The Structural Validity Framework (pp.6-7, Fig. 4) has five pass/fail checks. "if any requirement fails, the reported result can only support a stress test interpretation or a proof of concept, and it cannot support a claim of deployable alpha" (p.6).
    1. Temporal Sanitation (p.6):
       - Disclose the latest pre-training and fine-tuning data dates and make the evaluation window respect them.
       - Use point-in-time retrieval over archived snapshots, and point-in-time structured data.
       - Keep full trace logs of prompts, retrieved documents, tool outputs and timestamps.
    2. Dynamic Universe Construction (p.6):
       - Use a time-indexed universe U_t that includes later-delisted entities.
       - Report the fraction of observations from delisting entities, and report outcomes separately for survivors and non-survivors.
    3. Rationale Robustness (pp.6-7):
       - Claims must trace to retrieved passages. Report violation rates.
       - Run entity-substitution tests and negative controls with scrambled inputs.
    4. Epistemic Calibration (p.7):
       - Output "a predictive distribution, a calibrated confidence score, or a structured uncertainty estimate that downstream risk controllers can consume".
       - Include an explicit "No Trade" / "Do Not Know" abstention that is scored as valid.
    5. Realistic Implementation Constraints (p.7):
       - Report net metrics as primary outcomes, and the full latency distribution.
       - Specify spread, fees and impact. Report token usage and tool calls.
       - Use budget-matched baselines.
  - The paper also has a user study (Appendix A, pp.13-17) and a literature dashboard (Appendix B, p.17).
- Data, universe, evaluation period, baselines:
  - Literature review:
    - 164 main-conference papers from 2023-2025 at ICML, ICLR, NeurIPS, KDD, AAAI, IJCAI, CIKM, ACL, EMNLP, NAACL and ICAIF (p.2, fn.1; p.17).
    - The full corpus includes 226 workshop papers (58.0%) and 164 main-conference papers (42.0%) (p.17).
    - Papers were sorted into research-topic categories by rule-based keyword matching with human verification (p.17). The paper does not describe how it decided whether a paper mentions a given bias.
  - User study: 112 respondents, of whom 50 (44.6%) completed all required questions (Q1-Q11) (p.13). 56% were academic researchers and 44% industry practitioners (p.13).
  - No trading experiments.
- Key results (copy numbers EXACTLY, with page):
  - Literature coverage (p.2): "only 26.8% acknowledge look-ahead bias, and survivorship bias appears in just 1.2% of studies".
  - Figure 2 (p.2), counts read from the bar labels ordered by position (Look-ahead, Survivorship, Narrative, Objective, Cost): 44 (26.8%), 2 (1.2%), 19 (11.6%), 20 (12.2%) and 46 (28.0%). The bias-to-number mapping for Narrative, Objective and Cost comes from label positions in the figure, not from running text.
  - The text states "no single bias mentioned in more than 28% of the 164 papers surveyed" (p.5).
  - User study (p.2, p.7): "74% reported that ready-to-use evaluation tools are scarce or non-existent", and "50% identified the lack of tools and frameworks as the biggest bottleneck to mitigation".
    - Appendix (p.16): 48% "Scarce" and 28% "Non-existent", "combined 76%".
    - "Difficulty in accessing high-quality, point-in-time data" was the second bottleneck, at 28% (14 participants) (p.16).
    - Look-ahead bias was perceived as most critical, with 54% rating it "Extremely critical" (p.14).
    - "62% of participants reporting low confidence that benchmarks capture real-world scenarios, including latency, costs, and realistic dataset availability" (p.17).
  - Topic breakdown (p.17): Dataset & Benchmark 81 papers (20.8%), Agent 54 (13.8%), Portfolio 13 (3.3%), Risk Management 11 (2.8%).
- Limitations:
  - As stated by the authors: a "Alternative Views" section (pp.7-8) addresses four objections:
    - The checklist slows early research.
    - Point-in-time and delisting data is costly and proprietary, which raises equity concerns.
    - Strict financial constraints test pipeline engineering rather than reasoning.
    - Hallucination is more pressing than bias.
    - Their rebuttal to the data-access objection: "When researchers lack the data required to support a deployable claim, the appropriate response is to narrow the claim itself" (p.8).
  - Our observation:
    - The coding procedure behind the bias-prevalence counts (Fig. 2) is not described. The keyword rules on p.17 are for topic categories, and the paper only says that all included papers were human-verified (p.17).
    - The user study has a 44.6% completion rate (50/112) and a small sample.
    - The main text and appendix are internally inconsistent:
      - Non-existent tools: 26% on p.7 vs. 28% on p.16. The combined figure: 74% vs. 76%.
      - "Lack of awareness" as the biggest bottleneck: 6 (12%) in Fig. 5 (p.7) vs. 7 (14%) in the text and Fig. 9 (p.16). Fig. 5 also has a "Not a Priority" bar (1, 2%) that Fig. 9 lacks.
    - The paper provides no empirical demonstration of the biases' magnitude on a trading system. It is a checklist and position paper.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "if any requirement fails, the reported result can only support a stress test interpretation or a proof of concept, and it cannot support a claim of deployable alpha." (p.6)
- "However, the mere existence of a stated knowledge cutoff does not guarantee the exclusion of post-cutoff information." (p.2)
- "The action space should include explicit abstention, such as a “No Trade” or “Do Not Know” option, and the evaluator should score abstention as a valid result rather than an automatic failure." (p.7)
- "Net metrics should serve as primary outcomes, and comparisons should use budget-matched baselines with similar latency and cost constraints." (p.7)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Evaluation checklist we can satisfy item by item.** Their five checks map onto our design and evaluation. We could include a filled checklist in the appendix, as they propose (p.19):
  1. **Temporal sanitation.** CLAUDE.md rule 3 already requires point-in-time alignment: SEC filings by filed/accepted date, news by timestamp, and macro data by ALFRED vintage. Consistent with p.6:
     - Disclose Jev's training-data dates and respect them in the evaluation window. The archived docs (docs/jev/) state no knowledge cutoff, only the 2026-09-15 release date, so we must report that the cutoff is undisclosed.
     - Run a post-cutoff window, since a stated cutoff alone "does not guarantee" exclusion (p.2).
     - Log all prompts, inputs, Jev outputs and decision timestamps, which fits our explainability and memory layers.
  2. **Dynamic universe.** Our universe must be point-in-time: historical index membership, including later-delisted names. We should report the delisted fraction (p.6). If we cannot, we must narrow our claims accordingly (p.8).
  3. **Rationale robustness.** The CoT/explanation layer should cite stored sources and report violation rates. We can add entity-substitution and scrambled-input negative controls (pp.6-7). This directly addresses the risk that "explainability" is narrative bias.
  4. **Epistemic calibration.** Their requirement for calibrated probability outputs consumable by risk controllers, plus an explicit "No Trade" abstention (p.7), is almost exactly Jev's typed calibrated-probability interface (Choice/Score with probabilities, per docs/jev) and our risk layer. It is a strong external justification for a typed, calibrated decision engine. We must still measure calibration empirically (e.g., reliability/ECE) rather than assume it.
  5. **Realistic implementation.**
     - Report net-of-cost returns as the primary metric, including Jev API cost (priced per million input tokens per docs/jev) and any LLM escalation cost in C_M.
     - Report the full latency distribution, not a best case (p.7).
     - Compare against budget-matched baselines.
     - At mid/low frequency, latency matters less, but we should say so explicitly.
- **Framing citation.** The 26.8% look-ahead and 1.2% survivorship coverage statistics (p.2) motivate why our evaluation section is unusually strict.
- **How we differ:** this paper provides no system or experiments. Ours is a concrete system whose evaluation is designed to pass their checks. Jev's non-generative typed output also sidesteps part of the "objective bias" concern about verbal overconfidence (p.5), subject to our own calibration measurements.

## BibTeX
```bibtex
@article{kong2026evaluating,
  author  = {Kong, Yaxuan and Lee, Hoyoung and Hwang, Yoontae and Lopez-Lira, Alejandro and Levy, Bradford and Mehta, Dhagash and Wen, Qingsong and Choi, Chanyeol and Lee, Yongjae and Zohren, Stefan},
  title   = {Evaluating {LLMs} in Finance Requires Explicit Bias Consideration},
  journal = {arXiv preprint arXiv:2602.14233},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.14233}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- What was checked:
  - Metadata against the saved records. The arXiv API XML gives the title, ten authors in order, v1 only (2026-02-15), categories cs.LG/cs.AI/q-fin.CP, and no journal-ref. The abs page and the S2 record (venue arXiv.org, CoRR DBLP key) agree. PDF p.1 has the equal-contribution marks, the eight affiliations and the "Preprint." footer. A new Crossref title search found no published version. DBLP (bot challenge) and OpenAlex (budget exhausted) were unavailable.
  - Every number, definition and quote against literature/text/kong2026evaluating.txt (21 pages), re-checked in reading order because of the two-column layout:
    - Main text: 36->250 papers and 594% (6.9x) (p.1); the five sins, the 164-paper review, 26.8%/1.2%, the Fig. 2 bar labels 44/2/19/20/46, and the 112/50 survey headline numbers (p.2); the look-ahead and survivorship channels and Eq. (3) (pp.2-4); narrative, objective and cost bias with the R_net formula and P_{t+Δgen} (pp.4-5); "no single bias ... 28%" (p.5); the framework checks and their requirements (pp.6-7); the Q8/Q9/Q11 numbers and Fig. 5 (p.7); alternative views (pp.7-8).
    - Appendices: 44.6% completion and the 56/44 split (p.13); 54% "Extremely critical" (p.14); Q9 48/28/76%, Q11 50/28/14/8% (p.16); 62% low confidence, topic counts, and 226/164 workshop/main papers (p.17); checklist text (p.19). Figs. 10-11 are images, as stated.
    - All verbatim quotes match the text.
  - The BibTeX against the arXiv record: the ten authors, title, year 2026 and id match.
- What was corrected:
  - The summary attributed the bias-prevalence method to keyword matching. The keyword rules are for topic categories only, and the bias-coding method is not described. Fixed in both the data section and our observations.
  - Added a second grounded inconsistency: "Lack of awareness" is 6 (12%) in Fig. 5 (p.7) vs. 7 (14%) on p.16.
  - Relevance: "Disclose Jev's stated knowledge cutoff (per docs/jev)" was ungrounded, because docs/jev/ states no knowledge cutoff. It now says the cutoff is undisclosed and only the release date is documented.
  - Recorded the Crossref negative search in the metadata section.

