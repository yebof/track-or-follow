# xia2026agentic: Agentic Trading: When LLM Agents Meet Financial Markets

## Verified metadata
- Title: Agentic Trading: When LLM Agents Meet Financial Markets [source: arxiv, datacite]. The PDF title (p.1) and the PDF info field match.
- Authors: Yihan Xia, Panpan You, Taotao Wang, Fang Liu, Han Qi, Xiaoxiao Wu, Shengli Zhang [source: arxiv, datacite]. The PDF lists Shenzhen University affiliations, with Taotao Wang as corresponding author (p.1).
- Venue: arXiv preprint (cs.AI) [source: arxiv]. The text says "For the purposes of ESWA" (p.6), which suggests a manuscript prepared for Expert Systems with Applications (the PDF's own reference list pairs that journal name with the DOI prefix 10.1016/j.eswa, p.57; this is our inference, not a stated venue). No Crossref record for it exists (Crossref bibliographic search, 2026-09-25; re-run the same day with the same result), so we do not claim a peer-reviewed venue. DBLP was unreachable (bot-check page, also on the re-query).
- Year: 2026 (v1 submitted 2026-05-19) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2605.19337 (arXiv DataCite DOI) / arXiv:2605.19337 (v1)
- Metadata files:
  - literature/meta/xia2026agentic.arxiv.html (arXiv abstract page; the export API returned HTTP 429)
  - literature/meta/xia2026agentic.datacite.json
  - literature/meta/xia2026agentic.crossref_search_nomatch.json
  - No Semantic Scholar record: six retries on 2026-09-25 all returned HTTP 429, so no s2 file was saved.

## PDF
- Path: literature/pdf/xia2026agentic.pdf (arXiv 2605.19337v1). The title on p.1 matches.
- Text: literature/text/xia2026agentic.txt ; pages: 59

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - LLM trading agents have multiplied, but the literature lacks a shared lens for agent boundaries, protocol reporting and architectural roles (p.2).
  - Reported performance claims are hard to audit or compare because protocol-critical reporting is sparse (p.38).
  - The paper reframes LLM trading agents as "expert-system decision pipelines" (p.1, p.6).
- Method / architecture (of the review):
  - **Evidence map and search (p.4).** It is an audit-oriented evidence map, not an exhaustive systematic review. The search covers 2022-01-01 to 2026-03-09 in ACM DL (including ICAIF), IEEE Xplore, arXiv, SSRN and Google Scholar, plus citation chaining.
  - **Registry (p.4-5).** 92 deduplicated candidate records yield 77 included studies (15 excluded).
  - **Primary subset (p.4).** The 77 are split into a primary empirical subset of n=19 and a background tier of n=58. A primary study must have Action Output (it emits orders, position changes or allocations) and Closed-Loop Evaluation (backtest, simulation, live trading or benchmark).
  - **Coded fields (p.5).** Architecture: perception, memory, reasoning, action/execution, adaptation. Protocol-critical fields: time-consistent split, execution semantics, transaction-cost model, universe/survivorship handling, and artifacts at reproducibility tiers R0–R3.
  - **Reproducibility tiers (p.21).** R0 = no runnable artifacts, R1 = code available but not runnable, R2 = runnable with gaps.
  - **Reliability audits (p.5).** A screening double-check on 182 raw records, plus an extraction-level audit on 15 fixed-seed primary studies (seed=20260426).
  - **Architecture–Capability–Adaptation (A-C-A) lens (p.3-4).** The authors call it "a working analytical lens" and state: "We do not present A-C-A as an externally validated taxonomy" (p.4).
  - **Minimum-reporting checklist MR-1..MR-7 (p.38).**
    - MR-1 data and universe
    - MR-2 time-consistent splitting with embargo/purge
    - MR-3 action semantics
    - MR-4 execution and costs with sensitivity analysis
    - MR-5 leakage audit
    - MR-6 artifacts and logs. This includes "inference cost per decision" for LLM systems.
    - MR-7 multi-agent evaluation: roles, message protocol, consensus mechanism, shared-state consistency, and role/communication ablations.
  - **MR applicability matrix (Table 19, p.39).** Obligations are Mandatory, Recommended or Optional depending on study type.
  - **Design positions the authors put forward:**
    - Memory layers (p.10):
      - "Layer A: Deterministic State Store (Audit-Truth)": positions, orders, balances and risk limits, read-only to the LLM and updated only by the environment.
      - "Layer B: Generative Context": the transient context window.
    - A "Hierarchy of Truth": internal Layer-A state overrides noisy external signals (p.12).
    - Risk (p.26): separate a deterministic "Gating Layer" (circuit breakers) from an asynchronous LLM "Explanation Layer". Place safety layers "outside the model's generative reasoning loop".
- Data, universe, evaluation period, baselines:
  - This is a survey, not a trading experiment.
  - The corpus snapshot is screened through 2026-03-09 (p.1, p.4).
  - 18/19 primary studies were published 2024–2026. 17/19 are coded non-peer-reviewed and 2/19 peer-reviewed (p.43).
  - The multi-agent section lists TradingAgents, FINCON and TradingGroup as role-based examples with tag "R2,-E-" (Table 17, p.34). The section-level primary-study count for multi-agent work is N=12 (p.35).
- Key results (copy numbers EXACTLY, with page):
  - **Protocol reporting in the primary subset (p.1; p.41):**
    - Split protocols: 2/19 (10.5%)
    - Explicit transaction-cost models: 1/19 (5.3%)
    - Universe/survivorship handling: 1/19 (5.3%)
    - Execution timing or semantics: 11/19 (57.9%)
  - **Reproducibility tiers (p.41):** "15/19 studies at R0, 3/19 at R2, and 0/19 at R3". "R2+: 15.8%" (p.38). "78.9% of studies are at R0" (p.41).
  - **Artifacts (p.43):** 4/19 expose artifacts beyond narrative description.
  - **Explicitly foregrounded issues (p.39-40):**
    - Look-ahead bias: "N=0 primary studies" explicitly foreground it (p.39).
    - Hallucination: only one primary study explicitly foregrounds it (p.39).
    - Explainability vs performance: N=0 (p.40).
  - **Claim-to-evidence map (Table 20, p.42):**
    - The strongest cross-paper conclusion is that cost and reproducibility gaps limit comparability.
    - "Multi-agent systems improve decomposition, not guaranteed performance": coordination benefits are "plausible but attribution remains weak without role/communication ablations". Both are table-cell text reassembled from the layout extraction.
  - **Inference cost (p.40):** the authors propose "Annual API Cost = D × N × Creq". They ask papers to disclose model version, pricing snapshot date, token assumptions and the break-even gross P&L.
  - **Latency (p.40):** System 2 reasoning "is too slow for high-frequency trading". They suggest hybrid cascades in which "fast System 1 handles routine decisions ... escalating difficult cases to slower System 2 reasoning".
- Limitations:
  - **Stated by the authors:**
    - A-C-A has not been independently validated (p.42).
    - The MR-1..MR-7 dimensions reflect author judgment (p.42).
    - With only 15.8% of primary studies at R2+, architectural patterns are provisional (p.42).
    - The absence of negative results is a reporting limitation, and survivorship bias likely exists at the survey level (p.41).
    - The screening double-check sample predates the final 92-record registry (p.5).
  - **Our observation:**
    - The primary denominator (n=19) is small, so the percentages are coarse.
    - Supplementary materials S1–S3 are described (p.42-43). The arXiv PDF reproduces only the S1 search summary (Tables A.21–A.22, p.44), not the S2 evidence-mapping master table or the S3 coding sheet, so the per-study codes cannot be checked from the PDF alone.
    - Many design recommendations (Layer A/B memory, gating vs explanation layer) are author proposals, not empirically validated findings. The authors mark several sections as background or protocol guidance (e.g., p.39-40).

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "The central empirical finding is protocol incomparability: within the primary subset, only 2/19 studies report extractable time-consistent split protocols, 1/19 reports an explicit transaction-cost model" (p.1)
- "Treating this state as a “memory”subject to forgetting or hallucination is a critical vulnerability" (p.10). The missing space before "subject" is as in the extracted text.
- "LLMs are generally unsuitable for the sub-millisecond Critical Path." (p.26)
- "A trading agent can appear architecturally sophisticated while still being empirically weak if the evaluation omits temporal splits, transaction costs, universe construction, execution semantics, or reproducible artifacts." (p.6)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Motivating claim for the paper.** The strongest recent evidence map shows that, among closed-loop LLM trading studies, almost none report explicit cost models (1/19), time-consistent splits (2/19) or universe handling (1/19), and none reach R3 (p.1, p.41). Our evaluation should satisfy MR-1..MR-7 explicitly (p.38-39), and the paper should include a reporting-compliance table:
  - point-in-time universe
  - dated walk-forward splits with embargo
  - next-bar execution semantics
  - cost and slippage model with a sensitivity sweep
  - leakage audit, including a Jev knowledge-cutoff diagnostic
  - released code, logs and per-decision Jev cost
- **Memory design.** Their Layer A "Deterministic State Store" is read-only to the model, and their "Hierarchy of Truth" rule (p.10, p.12) matches our choice. Portfolio, positions and risk limits live in a deterministic store that Jev agents read but cannot write. Only the environment and broker update it. We can cite this as convergent design guidance, not as validated evidence.
- **Risk control.** A deterministic gating layer outside the generative loop plus an asynchronous explanation layer (p.26; p.26 §9.4) supports our design: Jev outputs typed probabilities, and a deterministic risk engine enforces cash-only, long-only rules, exposure caps and drawdown stops.
- **System-1 framing.** Their latency discussion describes a fast "System 1" for routine decisions escalating to slow "System 2" (p.40). This is directly relevant to positioning Jev, marketed as a "System One" model (see docs/jev/). We must not overclaim: this survey offers the cascade idea as a future direction, not as evidence (p.40).
- **Cost accounting.** MR-6 asks LLM systems to report inference cost per decision (p.38), and p.40 gives the D × N × Creq formula. We should report Jev's per-decision cost using the archived vendor pricing, with the pricing snapshot date.
- **Multi-agent claims.** Coordination benefits need role/communication ablations (Table 20, p.42; MR-7 p.38). Our experiments therefore need leave-one-agent-out and memory-layer ablations.
- **How we differ.** This is a survey and audit. We contribute a system plus a protocol-complete evaluation, and we use this paper's checklist as our evaluation contract.

## BibTeX
```bibtex
@misc{xia2026agentic,
  author        = {Xia, Yihan and You, Panpan and Wang, Taotao and Liu, Fang and Qi, Han and Wu, Xiaoxiao and Zhang, Shengli},
  title         = {Agentic Trading: When {LLM} Agents Meet Financial Markets},
  year          = {2026},
  eprint        = {2605.19337},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2605.19337},
  howpublished  = {arXiv preprint arXiv:2605.19337},
  url           = {https://arxiv.org/abs/2605.19337}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check pass).
- What was checked:
  - Metadata: title, 7-author list, arXiv ID, v1 date (2026-05-19), subject (cs.AI), comment field (59 pages) and absence of a journal-ref against literature/meta/xia2026agentic.arxiv.html and xia2026agentic.datacite.json (version 1). The PDF header (arXiv:2605.19337v1, 19 May 2026), PDF info title and page count (59, equals `pdfinfo`) match. Crossref bibliographic search was re-run (no match). DBLP returned a bot-check page, Semantic Scholar returned HTTP 429 twice after a 40 s back-off, and OpenAlex's daily budget was exhausted, so no venue record beyond arXiv exists locally.
  - Text: every page reference, count and percentage (77/92/15, n=19/58, 182, 15, seed, 2/19, 1/19, 1/19, 11/19, 15/19, 3/19, 0/19, 4/19, 17/19, 18/19, 10.5%, 5.3%, 57.9%, 15.8%, 78.9%, N=12, N=0), the MR-1..MR-7 items, Tables 17, 19 and 20, the R0-R2 definitions (p.21), Layer A/B and "Hierarchy of Truth" (p.10, p.12), the gating/explanation layer and §9.4 (p.26), the cost formula and latency passage (p.40), and all four verbatim quotes against literature/text/xia2026agentic.txt, with a non-layout pdftotext pass for column-split sentences.
  - BibTeX fields against the arXiv and DataCite records. Jev statements against docs/jev/docs_introduction.txt.
- What was corrected:
  - The A-C-A quote "not ... an externally validated taxonomy" was a stitched fragment; replaced with the exact sentence "We do not present A-C-A as an externally validated taxonomy" (p.4).
  - "Supplementary materials ... are not in the arXiv PDF" was too strong: the PDF reproduces the S1 search summary (Tables A.21–A.22, p.44). Reworded to say that the S2 master table and S3 coding sheet are absent.
  - The ESWA expansion is now marked as our inference and grounded in the PDF's own reference list (p.57).
  - Metadata-source note updated with the re-queries. No numerical errors were found.
