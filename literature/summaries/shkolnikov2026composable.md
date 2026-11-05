# shkolnikov2026composable: Composable Trust for Language Models: A proven boundary and a measured defense

## Verified metadata
- Title: Composable Trust for Language Models: A proven boundary and a measured defense [source: arxiv (OAI-PMH record, abs page citation_title); PDF p.1]
- Authors: Yakov Pyotr Shkolnikov [source: arxiv]. The PDF gives "Yakov P. Shkolnikov, Independent Researcher" (p.1).
- Venue: arXiv preprint, primary category cs.CR [source: arxiv]. No comments field and no journal reference on the abs page (2026-09-26). A Crossref bibliographic search returned no matching record (literature/meta/shkolnikov2026composable.crossref_search.json). The PDF is marked "A PREPRINT" (p.1). No peer-reviewed venue found.
- Year: 2026 (v1 submitted 2026-07-14) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2607.13149 (arXiv DataCite DOI) / arXiv:2607.13149 (v1)
- Metadata files:
  - literature/meta/shkolnikov2026composable.arxiv_oai.xml (arXiv OAI-PMH record)
  - literature/meta/shkolnikov2026composable.arxiv_abs.html
  - literature/meta/shkolnikov2026composable.datacite.json
  - literature/meta/shkolnikov2026composable.crossref_search.json (no match)

## PDF
- Path: literature/pdf/shkolnikov2026composable.pdf (arXiv 2607.13149v1; stamp "arXiv:2607.13149v1 [cs.CR] 14 Jul 2026" on p.1)
- Text: literature/text/shkolnikov2026composable.txt ; pages: 17

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Instructions and data share one token stream, so "nothing inside the model's generation can keep untrusted text from steering it" (p.1). The paper places the authority to act outside the model, in code (p.1).
  - Trust is framed as drawing on each source in proportion to its trust: a lower-trust source may inform an answer but must not overrule a higher-trust one (p.1).
- Method / architecture:
  - **Channels and rings.** Each input arrives on a channel with a fixed integrity ring; the deployed lattice is SYSTEM > USER > CONTENT (retrieved documents) > WEB (tool and web returns) (p.2). The ring is assigned out-of-band from the channel, not from what the input claims (p.2-3).
  - **Five invariants enforced by a deterministic monitor** (p.3): a derived value takes the meet (lowest) of its inputs' rings, the Biba low-water-mark rule (I1); a ring can rise only via an out-of-band authority token (I2); an action is authorized only if its controlling inputs meet a fixed threshold (I3); every write carries its own stamp (I4); the operation binds to the instruction on the highest ring present (I5).
  - **Proof by enumeration.** Because the lattice is finite, enumeration proves that no untrusted input can change the bound operation or authorize an action; monotonicity of the meet extends this to arbitrary depth (p.3). Checking the invariants over 8729 input configurations found no counterexample, which the paper calls a non-interference property (p.9).
  - **Measured (not proved) content defense: the "conditioned cascade".** An unmodified model "passivates" each below-SYSTEM channel (strips it to plain factual statements), conditioned on the already-passivated higher-trust context, then answers once; a wrapper tags each block with its ring and a per-request nonce (a spotlighting variant) (p.3-4, Algorithm 1 p.4). "The wrapper is only an instruction: it lowers the attack rate but gives no formal guarantee" (p.3).
  - Prompts are tuned jointly by a SkillOpt-style, length-capped, multi-objective (clean quality Q, rejection R) search (p.5-6).
- Data, evaluation, baselines:
  - One model, gemma-4-26B-A4B at 8-bit, no fine-tuning (p.6). Attacks compose a delivery form with a payload over fresh content; AgentDojo templates are reused verbatim as an external anchor (p.7). Five ablation conditions (base, base+prompt, wrapper, passivation, both) (p.6).
  - Judge: Claude Opus 4.8, validated against a blind human rater (κ = 0.93 on obeyed; 0.90 on the genuine-leak regrade); GPT-5.6 reproduces the genuine-leak grade (p.8).
  - The authors state sample sizes are "modest by design"; the guarantee "needs none" (p.8).
- Key results (numbers copied exactly):
  - Held-out instruction injection, genuine-leak defended rate: aggregate 27% (base) → 94% [88,97] (cascade); token-emit 31 → 100; task-hijack 42 → 91; AgentDojo templates 3 → 89 (Table 2, p.8).
  - Content integrity: content-poison 67 → 82 [67,91]; cross-ring conflict 52 → 100 (Table 2, p.8). Attribution of web-sourced facts 0 → 92 (p.8-9).
  - Clean quality 77 → 73, Qrel = 0.96; authority-rejection "0 (proved, 0/8 structural check)" (Table 2, p.8).
  - Adaptive attacks (Gemma 4 31B attacker, n = 30 per family, budget 10): authority 0 violations (0/12 actions); text defense 87% [70,95] defended for white-box and black-box families; the multi-ring optimizer broke 0/30 (Table 4, p.11).
- Limitations:
  - Stated by the author: the proof concerns the authorized action, "not the generated text" (p.9); the text defense can only be "measured and pushed down, never guaranteed" (p.10); a defense its own authors could not break "is only a floor" (p.12); a new model or ring definition needs re-tuned prompts (p.5); samples are small (p.8).
  - Our observation: single author, single model family, no finance or decision-quantity setting; the "boundary" is a discrete property (which operation runs, whether an action fires), not a bound on how much content can move a continuous decision.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "a source’s standing, not its content, decides which operation runs and whether it acts." (p.1)
- "A lower-trust source may inform an answer but not override a higher one." (p.1)
- "That is the only hard guarantee, and it constrains the authorized action, not the generated text." (p.9)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Closest structural precedent for positioning K2** ("trust partition + band guard give a closed-form bound; inside the band, typed reading lowers attack success"). This paper already pairs a *proved* trust boundary enforced by a deterministic monitor with a *measured* in-band defense, uses a trust lattice with a low-water-mark (meet) rule, lets lower-trust sources "inform but not override", and evaluates adaptive attackers (p.1-3, p.9-11). Our taint rule "max of input taints, W > S > T" (design §3.2) is the same low-water-mark idea.
- What remains different (and is all we may claim against it):
  - Its proved property is discrete (operation binding and action authorization, p.3, p.9); content influence on the answer is only measured (p.9-10). Ours bounds a *continuous* decision quantity: membership relative to a trusted ranking band, weights never above trusted weights, |Δlogit| ≤ 2κ_t (design §3.4), for any reader.
  - Its trust order is over instruction channels (system/user/content/web); ours is over data channels (prices/XBRL vs issuer text vs news) whose "trusted" side is itself a predictive signal.
  - Our setting grades actions by delayed market outcomes and reports the robustness–utility frontier over J.
- Wording consequence: the phrases "proven boundary + measured defense" and "lower-trust input may inform but not override" must not be presented as our idea; cite this paper and state the continuous-bound difference.

## BibTeX
```bibtex
@article{shkolnikov2026composable,
  title   = {Composable Trust for Language Models: A proven boundary and a measured defense},
  author  = {Shkolnikov, Yakov Pyotr},
  journal = {arXiv preprint arXiv:2607.13149},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: title, author, category, v1 date against the saved OAI-PMH record, abs page and DataCite record; PDF stamp p.1; all numbers against the extracted text (Table 2 p.8, Table 4 p.11, 8729 configurations p.9, κ values p.8). Quotes checked against the text (whitespace normalized).
- Not verified: any peer-reviewed publication (none found).
