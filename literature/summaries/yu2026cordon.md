# yu2026cordon: Cordon-MAS: Defending RAG against Knowledge Poisoning via Information-Flow Control

## Verified metadata
- Title: Cordon-MAS: Defending RAG against Knowledge Poisoning via Information-Flow Control [source: arxiv (OAI-PMH, abs page citation_title); PDF p.1]
- Authors: Zhe Yu, Wenpeng Xing, Gaolei Li, Shuguang Xiong, Hongzhi Wang, Xuyang Teng, Meng Han [source: arxiv]. The PDF marks the first two as equal contribution and lists Zhejiang University, Binjiang Institute of Zhejiang University, Shanghai Jiao Tong University, Zhejiang Lab, Harbin Institute of Technology, Hangzhou Dianzi University and GenTel.io (p.1).
- Venue: arXiv preprint, primary category cs.CR [source: arxiv]. No comments and no journal reference on the abs page (2026-09-26). A Crossref bibliographic search found no matching record (literature/meta/yu2026cordon.crossref_search.json). No peer-reviewed venue found.
- Year: 2026 (v1 submitted 2026-05-26) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2605.26754 / arXiv:2605.26754 (v1)
- Metadata files: literature/meta/yu2026cordon.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json (no match)

## PDF
- Path: literature/pdf/yu2026cordon.pdf (arXiv 2605.26754v1; stamp "arXiv:2605.26754v1 [cs.CR] 26 May 2026", p.1)
- Text: literature/text/yu2026cordon.txt ; pages: 24

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - RAG defenses assume that detecting poisoned evidence prevents harm; the paper argues models show a "monitoring-control gap": they can detect contradictions yet still act on poisoned claims (p.1).
  - Poisoning is therefore framed as an information-flow control problem rather than a detection problem (p.1).
- Method / architecture:
  - **Cordon Principle:** no agent capable of final natural-language synthesis may access untrusted natural-language evidence; "The principle is enforced architecturally, not through prompting" (p.1).
  - Four agents with asymmetric memory privileges (p.1-3, Fig. 1 p.3): an Extractor that alone reads raw documents and turns them into structured "Evidence Claim Cards" (entity, relation, object, source, retrieval rank); an Auditor scoring cross-source support S and marginal influence I, rejecting claims with R = I·(1−S) > 0.65 (Eq. 1, p.3); a Gate that labels evidence ANSWERABLE / INSUFFICIENT / CONFLICTING; a Synthesizer that answers only from certified claims (p.2-3).
  - Three stated invariants: dirty-read isolation (the Synthesizer never receives raw text), claim-only communication (structured claim cards, not free-form language), certified synthesis (p.2).
  - An informal "attention contamination" observation argues that prompt defenses can only attenuate, not remove, poison influence, while architectural isolation removes the channel (p.3-4).
- Data, evaluation, baselines:
  - Five BEIR datasets (SciFact, FiQA, NQ, MS MARCO, HotpotQA), 50 queries each, Contriever retrieval (K = 10), DeepSeek-Chat backend, GPT-4o and Qwen2.5-32B for cross-backend checks, seed 42 (p.4).
  - Baselines: Vanilla RAG, RobustRAG, TrustRAG, Paraphrase, Debate; prompt baselines CoT-Detect and Danger Evaluator (p.4-5).
  - Threat model from Confundo; adaptive attacks: claim mimicry, consistency collusion, judge confusion (p.4). ASR is an LLM-judged endorsement rate (p.4).
- Key results (numbers copied exactly):
  - Mean ASR 2.1% for Cordon-MAS vs 27.5% for Vanilla RAG (a 92.4% relative reduction) and 14.3% for RobustRAG (Table 1, p.5). Pooled Wilson CI [0.8%, 4.7%] (n = 241) (p.5).
  - CoT-Detect reduces ASR from 34.0% to 24.0% but "cannot reach zero"; Cordon-MAS 0.0% on the same SciFact+NQ subset (Table 2, p.5).
  - Clean answerability 60% mean with a 40% safety-refusal rate; correctness on answered queries 78–86% vs 46–67% for Vanilla RAG (p.4).
  - Consistency collusion (mutually corroborating poison documents) bypasses the audit 70.3% of the time with 63.3% pipeline penetration (p.6-7); under CorruptRAG-AS, SciFact ASR is 26.0% (13/50) (Table 6, p.7).
- Limitations:
  - Stated by the authors: collusion across multiple poisoned sources is "a fundamental constraint on any defense relying on within-corpus cross-source verification"; mitigation needs external trust anchors such as source-authority metadata (p.7, p.9); all agents share one backend within a run (p.9); 50-query samples and seed-dependent variance (Cordon-MAS ASR 2.0–26.5% on SciFact across seeds) (p.9); 2.2× latency and 2.8× cost (p.9).
  - Our observation: QA answers, not decisions graded by outcomes; the Extractor, Auditor and Synthesizer are all generative LLMs.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "This reframes RAG poisoning from a detection problem to an information-flow control problem." (p.1)
- "The Synthesizer never receives raw document text." (p.2)
- "any defense relying on within-corpus cross-source verification faces this constraint when the attacker controls enough sources." (p.7)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Pre-empts our taint barrier R3** (design §3.2: synthesis states N3/P2 contain only code-rendered typed labels; raw text cannot reach synthesis questions) and the "typed messages only" rule (design §4.3). Cordon-MAS already enforces that the final synthesizer never sees raw untrusted text and that agents communicate only through structured claim cards, and shows a large ASR drop from that architectural choice (p.1-5).
- It also supplies the direct evidence behind C1's hypothesis (decomposed, structured reading of each item beats a reader that sees all raw evidence). C1 must therefore be framed as a test of this principle in a new setting (portfolio decisions, typed probabilities, delayed outcomes, trusted-channel band), not as a new mechanism.
- Its collusion result (70.3% audit bypass with mutually corroborating poison documents, p.6-7) is the empirical counterpart of our out-of-scope case (coordinated fabrication) and of the corroboration weakness in our exclusion rule (≥ 2 items / bylines, design §3.3 and design_v2_delta SAN-5). It supports the recommendation that corroboration must come from independent trusted anchors, not from more untrusted items.
- How we differ: the Cordon synthesizer is still a generative LLM that reads claim text; our decision layer is code, and influence is bounded relative to a trusted ranking (design §3.4). We measure outcomes in money-denominated positions, not QA endorsement.

## BibTeX
```bibtex
@article{yu2026cordon,
  title   = {{Cordon-MAS}: Defending {RAG} against Knowledge Poisoning via Information-Flow Control},
  author  = {Yu, Zhe and Xing, Wenpeng and Li, Gaolei and Xiong, Shuguang and Wang, Hongzhi and Teng, Xuyang and Han, Meng},
  journal = {arXiv preprint arXiv:2605.26754},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against the saved OAI-PMH record, abs page and DataCite record; PDF stamp; the Cordon Principle, invariants, Eq. 1 threshold, Tables 1, 2, 6 and the collusion numbers against the extracted text (p.1-9).
- Not verified: peer-reviewed publication (none found).
- Quotes were checked against a reading-order pdftotext extraction because the saved -layout text interleaves the two columns.
