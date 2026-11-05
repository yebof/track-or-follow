# hu2026jevadvbench: JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models

## Verified metadata
- Title: JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models [source: arxiv; datacite; PDF p.1]
- Authors: Jianyi Hu, Hangtao Zhang, Yi Liu, Yeqi Zeng, Li Zeng, Xianlong Wang, Rui Wang, Leo Yu Zhang [source: arxiv]. PDF p.1 lists the Institute of Information Engineering (CAS), Huazhong University of Science and Technology, Griffith University, Changsha University of Science and Technology, City University of Hong Kong, and UCAS.
- Venue: arXiv preprint (cs.CR); comment "33 pages, 13 figures, 19 tables. Project website: https://JevAdvBench.github.io/JevAdvBench/" [source: arxiv]. No peer-reviewed venue.
- Year: 2026 (v1 submitted 2026-09-25) [source: arxiv]
- DOI / arXiv: 10.48550/arXiv.2609.31142 / arXiv:2609.31142 (v1)
- Metadata files: literature/meta/hu2026jevadvbench.arxiv.xml, literature/meta/hu2026jevadvbench.datacite.json

## PDF
- Path: literature/pdf/hu2026jevadvbench.pdf (arXiv v1; stamp "arXiv:2609.31142v1 [cs.CR] 25 Sep 2026", p.1)
- Text: literature/text/hu2026jevadvbench.txt ; pages: 32

## Summary (grounded)
- Problem: the robustness of typed (RLCD) decision models such as Jev has not been measured; adversarial benchmarks score generated text or tool calls, but a typed model generates nothing and returns a well-formed answer even when manipulated (p.1-2).
- Method: score each attacked decision against the model's own clean decision (a flip), read against the flip rate of an identical re-run (noise floor); 812 typed questions over 66 scenarios (314 Noul, 337 Choice, 161 Score); nine single-query attacks on three request surfaces (question, state, injection into question strings) plus three out-of-schema structure edits as delivery controls; 9,744 variants; billed input tokens verify delivery (p.1-2). Threat model: black-box, one query per attempt, no adaptation (p.3).
- Key results (numbers copied exactly):
  - Rewording raises the flip rate by at most 1.2 pp over the noise floor; out-of-schema fields never reach the model (p.1-3).
  - "An observer's opinion appended to the state flips 12.1% [8.5, 15.5] of decisions, statistically tied with the strongest injected command (10.1%), although the opinion contains no instruction" (p.3).
  - The same opinion pushes 38.0% [31.1, 44.0] of confident Choice and Score answers below a 0.8 confidence gate, against 0.5% under the re-run (p.3).
  - An identical request changes the returned Score value for 38.5% of Score questions; 82.4% of the public labels are the model's own consensus answers (p.2).
- Limitations (stated): one model version (jev-1.13.0, queried 2026-09-25); a deliberately weak, non-adaptive attacker, so rates are lower bounds; answer preservation checked by one LLM annotator on 40 variants (p.11).

## Verbatim quotes
- "Applications built on RLCD models should therefore treat the state as untrusted, argued input." (p.1)
- "A typed output removes generated text as an attack target, but the model still takes an argument in its input as evidence." (p.3)

## Relevance to our project
- Item-level evidence of our contribution (iii) point "a reader that follows what items say also follows planted items": argued content without commands moves hosted Jev as much as injected commands (p.3). Our registered per-family ASR (instruction-like families stopped, fabricated facts and planted releases not) is the portfolio-level counterpart, not a new finding about typed readers.
- Its label-free flip-vs-re-run design is a reporting template for the non-determinism our paper should state for NG and emu.

## BibTeX
```bibtex
@article{hu2026jevadvbench,
  title   = {{JevAdvBench}: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models},
  author  = {Hu, Jianyi and Zhang, Hangtao and Liu, Yi and Zeng, Yeqi and Zeng, Li and Wang, Xianlong and Wang, Rui and Zhang, Leo Yu},
  journal = {arXiv preprint arXiv:2609.31142},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-29 (novelty review r3). Metadata against the arXiv API record and DataCite; numbers and quotes against pdftotext output (p.1-3, p.11).
