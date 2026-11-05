# na2026poisoning: Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and Architectures in Multi-Agent Trading Systems
## Verified metadata
- Title: Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and Architectures in Multi-Agent Trading Systems [source: arxiv; PDF p.1; openalex]
- Authors: CheolWon Na, Hao Ni, Lukasz Szpruch, Zhangyang Wang, Dhagash Mehta, Saurabh Nagrecha, Alejandro Lopez-Lira, Chanyeol Choi, Yongjae Lee, Jee-Hyong Lee [source: arxiv; PDF p.1; openalex]
  - Affiliations on PDF p.1: Sungkyunkwan University, UCL, University of Edinburgh, UT Austin, BlackRock, Google (work done outside Google), University of Florida, LinqAlpha, UNIST.
- Venue: arXiv preprint [source: arxiv; openalex]
  - The PDF uses the ACM template with unfilled placeholders: "Conference’17, Washington, DC, USA" and DOI 10.1145/nnnnnnn.nnnnnnn (p.1). No venue is claimed.
  - The arXiv record has no journal_ref or comment.
  - A Crossref bibliographic search (2026-09-26) found no published version.
  - The OpenAlex record W7204267842 lists arXiv locations only.
  - DBLP was unreachable (bot-protection page).
- Year: 2026 (v1 submitted 25 Aug 2026; the arXiv abstract page re-fetched on 2026-09-26 still lists only v1 and no journal_ref) [source: arxiv]
- DOI / arXiv: arXiv:2608.24069; arXiv DataCite DOI 10.48550/arXiv.2608.24069 (used for the OpenAlex lookup)
- Metadata files:
  - literature/meta/na2026poisoning.arxiv_abs.html (the arXiv export API returned HTTP 406/429)
  - literature/meta/na2026poisoning.openalex.json
  - literature/meta/na2026poisoning.crossref_search_nomatch.json
## PDF
- Path: literature/pdf/na2026poisoning.pdf (arXiv v1, https://arxiv.org/pdf/2608.24069; pdfinfo title matches)
- Text: literature/text/na2026poisoning.txt ; pages: 16
## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - In LLM multi-agent trading systems, "a corrupted signal can propagate to the final decision and translate into realized financial loss" (p.1).
  - The paper asks two things: through which role an adversarial signal enters, and how far it survives toward the decision (p.1-2).
  - Threat model: a "black-box targeted attack". The adversary may modify only (i) external data sources and (ii) "prompt-level content entering an agent". It has no access to weights, tool servers, orchestration state or reasoning traces (p.3).
- Method / architecture:
  - A pipeline modeled on TradingAgents with four roles (p.3):
    - Analyst Team: social media, news, fundamentals, market;
    - Research Team: bull/bear debate;
    - Trader;
    - Risk Manager.
  - One attack per role (p.4; payloads verbatim in Appendix A, p.9-11, which labels them A1-A5):
    - A1 data poisoning of the news analyst;
    - A2 indirect prompt injection: a hidden imperative block appended to real Reddit posts;
    - A3 persuasive adversary replacing a researcher;
    - A4 objective hijacking: the trader's prompt replaced by a contrarian objective;
    - A5 jailbreaking of the Risk Manager: an "executive override" block prepended to its judge prompt (p.10-11).
  - ASR (eq. 1, p.3) counts days where the attack flips the decision to the target, over days whose clean decision differs from the target.
  - Architecture axis (p.5-6): three analysts (Market, Social, News) under three analyst-layer attacks at a 10% ratio: D1 data poisoning (News), D2 indirect prompt injection (Social) and A1 objective hijacking (Market) (p.5; p.13 "D1 = News, D2 = Social, A1 = Market"). Caution: this "A1" is objective hijacking, not the role-axis A1 (data poisoning) of Appendix A; the paper reuses the label. Four topologies:
    - Decentralized: majority vote;
    - Hybrid: peer revision, then mean aggregation;
    - Linear: sequential;
    - Centralized: one decision agent averages the reports.
  - APS = (1/(1+k))·(1−p) (p.6), a post-hoc lens with fixed values Dec. 0.00, Hyb. 0.33, Lin. 0.50 and Cen. 1.00 (p.3, p.6).
  - Data-level attacks poison 10% of source items (a 1:9 ratio) by default, with a 10/40/80% sweep (p.4-5).
- Data, universe, evaluation period, baselines:
  - Assets: BTC-USD, MSFT, NVDA, TSLA and AAPL (p.4, p.11).
  - Period: 2026 Q1 (2026-01-01 to 2026-03-31), 61 NYSE trading days, n = 305 asset-days per configuration (p.4, p.11).
  - Daily BUY/SELL/HOLD decision per single asset (p.3).
  - Sources: Alpha Vantage, yfinance and Reddit. Inputs are restricted to information available on or before each trading date, and both backbones are evaluated only on post-cutoff data to avoid look-ahead bias (p.4, p.11).
  - LLMs (p.4, p.12):
    - role axis: gpt-4.1 (deep-think roles) and gpt-4.1-mini (quick-think analysts);
    - architecture axis: also Qwen3-235B-A22B / Qwen3-30B-A3B.
  - Temperature 0, one debate round and one risk-discussion round (p.12).
  - Scale: about 102 pipeline configurations and "3–4 × 10^4 individual LLM calls". Per-call cost logs were not retained (p.12).
  - Backtest for Tables 2 and 4: $100,000, long-only, single position at a time (position ∈ {0, 1}), daily close. "No transaction costs: this engine applies no commission or slippage" (p.14).
  - Baselines: the clean pipeline, each topology, and a single-agent baseline (Fig. 8, p.7).
- Key results (copy numbers EXACTLY, with page):
  - Role-axis ASR, SELL-targeted (N = 225) / BUY-targeted (N = 183) (Table 1, p.4):

    | Attack | Role | SELL | BUY |
    |---|---|---|---|
    | Data poisoning | Analyst (news) | 21.8 | 19.1 |
    | Indirect injection | Analyst (social) | 29.8 | 24.0 |
    | Persuasive adversary | Researcher | 41.9 | 53.3 |
    | Objective hijacking | Trader | 18.0 | 13.8 |
    | Jailbreaking | Risk Manager | 95.5 | 98.9 |

  - Jailbreaking the Risk Manager succeeds on 98.9% of attackable days, "1.9× the next most effective attack". The others fall within 13.8–53.3% (p.4). The authors attribute this to the Risk Manager being the terminal decision node (p.4).
  - Macro-averages over assets (Fig. 3, p.5): jailbreaking 97.0%, persuasive adversary 50.6%, indirect injection 34.9%, data poisoning 26.2%, objective hijacking 25.4%.
  - Signed financial impact, EV per attempted decision (Table 2, p.5):

    | Attack | ASR | EV ($/attempt) | Median $/success |
    |---|---|---|---|
    | Persuasive adversary | 47.6% | −60 | 0 |
    | Jailbreaking | 97.2% | −43 | 0 |
    | Indirect injection | 26.9% | −30 | 0 |
    | Data poisoning | 20.5% | +4 | 0 |
    | Objective hijacking | 15.9% | +40 | −85 |

    The authors conclude "ASR is thus neither a lower nor an upper bound on realized loss" (p.5).
  - Poison-ratio sweep, BUY-targeted data poisoning: pooled ASR 19.1% → 28.4% → 30.6% at 10/40/80%. Per-asset ASR spans 13–52% at the highest ratio (p.5, Table 12 p.13).
  - The clean system is BUY-leaning: BUY 40.0%, HOLD 33.8%, SELL 26.2% (p.5, p.11).
  - Architecture-axis ASR, Dec./Hyb./Lin./Cen. (Table 3, p.6):

    | Backbone, target | Channel | Dec. | Hyb. | Lin. | Cen. |
    |---|---|---|---|---|---|
    | GPT-4.1, SELL | Market (A1) | 0.3 | 7.4 | 24.0 | 55.7 |
    | GPT-4.1, SELL | News (D1) | 0.7 | 1.7 | 5.2 | 5.6 |
    | GPT-4.1, SELL | Social (D2) | 1.6 | 4.1 | 10.5 | 21.0 |
    | GPT-4.1, BUY | News (D1) | 44.2 | 14.2 | 41.6 | 23.6 |
    | GPT-4.1, BUY | Social (D2) | 51.2 | 28.1 | 44.8 | 33.2 |
    | GPT-4.1, BUY | Market (A1) | 9.5 | 45.1 | 57.8 | 61.6 |

  - APS vs ASR: r = 0.85 (GPT-4.1) and r = 0.88 (Qwen3-235B-A22B), architecture-level (Fig. 6, p.7). Pooling all runs gives only Spearman ρ = 0.183 [0.057, 0.305] (Table 16, p.16).
  - Quorum effect under majority voting (GPT-4.1, BUY-targeted):
    - On 44.6% of attackable days, at least one benign analyst already votes BUY, so one compromised channel completes a 2/3 majority (p.7).
    - Compromising two channels raises BUY-targeted News and Social ASR to 68.7% (p.7).
  - Multi-agent vs single agent, pooled BUY-targeted (Fig. 8, p.7): GPT-4.1 architectures 29.1–48.1% vs 70.2% for a single agent; Qwen 60.1–82.2% vs 87.9%.
  - Cumulative return, GPT-4.1, BUY-targeted, clean / D1 / D2 (ΔCR in parentheses) (Table 4, p.8):

    | Structure | Clean | D1 | D2 |
    |---|---|---|---|
    | Decentralized | −2.68% | −14.68% (−12.01) | −15.06% (−12.38) |
    | Hybrid | −8.64% | −11.85% (−3.20) | −15.10% (−6.46) |
    | Centralized | −9.58% | −13.58% (−4.00) | −11.03% (−1.45) |
    | Linear | −10.33% | −15.35% (−5.02) | −14.66% (−4.33) |

  - Conclusion: "no architecture is inherently robust", and multi-agent design alone is insufficient as a defense (p.8).
- Limitations:
  - As stated by the authors (p.8):
    - the role-based evaluation uses a single backbone;
    - five large-cap assets may not represent thinner-coverage securities;
    - the daily closing-price backtest omits execution timing, transaction costs, slippage and detailed position management.
  - Our observations:
    - **Short sample, no seeds.** The window is one quarter at temperature 0. A moving-block bootstrap gives CIs for Table 2 (p.15), but the main tables report point estimates only. The universe mixes one crypto asset with four mega-cap stocks.
    - **Losing even when clean.** Every clean configuration in Table 4 has a negative cumulative return, −2.68% to −10.33% (p.8). The pipelines lose money even without attack.
    - **Decentralized mostly holds.** GPT-4.1's Decentralized clean run is HOLD on 295/305 days (96.7%) (p.11). Its low SELL-targeted ASR partly reflects inaction, and with a long/flat single-asset book, SELL flips while flat are cheap.
    - **Inconsistent A1 description.** The main text says an LLM "rewrites the selected articles to express the opposite sentiment" (p.4). Appendix A.2 says the article text is "replaced outright with a fabricated, direction-matched narrative drawn from a fixed template pool", with the numeric sentiment score left untouched (p.10).
    - **Inconsistent A5 description.** The main text says the jailbreak uses "hypothetical framing to bypass the guardrails" (p.4). Appendix A.2 shows that the payload behind the main results is an authority-override block ("EXECUTIVE OVERRIDE - PRIORITY ALPHA"), and that the hypothetical-framing variant is "unused-in-the-main-results" (p.10-11).
    - **Small numeric mismatches:**
      - Table 1 has Persuasive-SELL 41.9, but the per-asset Table 10 pools to 41.7 (the authors acknowledge the 0.2 pp, p.12).
      - Jailbreaking-SELL has N = 224 in Table 11 (p.13) vs N = 225 in the Table 1 footnote (p.4).
    - **Stronger threat model than Web text.** A3–A5 need control of agents' prompts or system prompts (p.4, p.10), which is stronger than "Web text only". Only A1/A2 are data-level.
    - **Static payloads.** A2 and A5 are explicit, static instruction blocks ("AI SYSTEM INSTRUCTION - PRIORITY OVERRIDE", "EXECUTIVE OVERRIDE", p.10-11). They are not adaptive attacks.
## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "A central finding is that no architecture is inherently robust: across the architectures examined, adversarial signals frequently survive deliberation and still reach the final decision." (p.2)
- "The result highlights the design risk of coupling safety enforcement with final decision authority without an additional check." (p.4)
- "ASR is thus neither a lower nor an upper bound on realized loss" (p.5)
- "security must come from deliberate validation and aggregation, not from agent multiplicity alone." (p.8)
## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **The closest 2026 competitor to C1; positioning.**
  - Their finding that "no architecture is inherently robust" (p.2, p.8) is about topology: how LLM reports are routed and aggregated.
  - We do not claim a topology is robust. Our claim is a closed-form, backend-independent influence bound enforced in code, independent of what any LLM/S1 outputs (docs/design/design_v1.md §2, §3.4). Its components:
    - the band guard: untrusted text can only reorder names within (K−J, K+J] of the trusted rank;
    - the text-channel cap κ_t;
    - the single-cluster demotion cap ×0.5;
    - exclusion only with ≥ 2 corroborating items.
  - This is the "deliberate validation and aggregation" the authors call for (p.8). The C1 bound check (ASR exactly 0 outside the band) is the empirical counterpart. We should cite this paper as the motivation and state that our bound holds whatever the reader's ASR.
- **Terminal safety node must not be an LLM.** Jailbreaking the LLM Risk Manager, which is also the final decision node, succeeds 98.9% of the time (p.4). Our design already has:
  - a code-only risk officer;
  - a deliberator that can only reduce positions;
  - a cash-only, long-only pre-trade check (design_v1 §2; CLAUDE.md §4).

  We should cite Table 1 as direct evidence for that choice.
- **Mean aggregation preserves adversarial signal.** In their data, Centralized/mean aggregation preserves the signal most (APS = 1). Under GPT-4.1 SELL-targeting, the Centralized ASR is 55.7% for objective hijacking of the Market analyst, and 5.6% (News, D1) and 21.0% (Social, D2) for the data-level text attacks. Each is the highest value in its row (Table 3, p.6). Our allocator is also an averaging log-opinion pool, so dilution alone would not protect us. The protection comes from the cap κ_t on the untrusted-text logit and from the band guard. Majority voting protects only below the quorum threshold (p.7). That is why we bound influence rather than rely on voting.
- **Portfolio-level outcome, not decision flips.** Decision-flip ASR mis-ranks financial harm (Table 2, p.5). This supports defining C1 success as the name entering the D-sel-EW book, and adding a paired attacked-vs-clean book-return delta as a secondary. Their ASR denominator (days whose clean decision ≠ target, eq. 1, p.3) matches our eligibility rule (the name is outside both clean books), so the ASR definitions are comparable in spirit.
- **Look-ahead handling.** They restrict evaluation to post-cutoff data (p.4). This is the same concern as our PIT rules and Jev knowledge-cutoff diagnostics (CLAUDE.md §3).
- **How we differ:**
  - Their threat model grants prompt/system-prompt control for A3–A5 (p.4). Ours is Web text only (news and policy items; A1–A8).
  - They decide daily on 5 single assets over one quarter without costs (p.4, p.14). We run weekly decisions on a PIT top-100 US-equity universe over W_clean (2024-07-05..2026-09-11, 115 decisions), with 15 bps/side costs (design_v1).
  - Their agents exchange free-text reports. Ours exchange typed S1 answers and code-computed features.
## BibTeX
```bibtex
@article{na2026poisoning,
  author  = {Na, CheolWon and Ni, Hao and Szpruch, Lukasz and Wang, Zhangyang and Mehta, Dhagash and Nagrecha, Saurabh and Lopez-Lira, Alejandro and Choi, Chanyeol and Lee, Yongjae and Lee, Jee-Hyong},
  title   = {Poisoning Agentic Alpha: Adversarial Vulnerabilities Across Roles and Architectures in Multi-Agent Trading Systems},
  journal = {arXiv preprint arXiv:2608.24069},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.24069}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check pass; files touched 2026-09-26 local time).
- Metadata checked against saved raw records: `meta/na2026poisoning.arxiv_abs.html` (title, 10 authors in order, v1 Tue 25 Aug 2026, no comment or journal_ref), `meta/na2026poisoning.openalex.json` (W7204267842, arXiv only), `meta/na2026poisoning.crossref_search_nomatch.json` (no publisher record). PDF p.1 confirms the title, authors, affiliations, the unfilled ACM placeholders ("Conference’17, Washington, DC, USA", 10.1145/nnnnnnn.nnnnnnn) and the arXiv stamp "arXiv:2608.24069v1 [cs.AI] 25 Aug 2026". The arXiv export API returned HTTP 406 on 2026-09-26. The arXiv abstract page, fetched live, still shows v1 only, with no journal_ref. DBLP was blocked by a bot check. The arXiv-preprint venue claim and the BibTeX stand.
- Content checked against `text/na2026poisoning.txt` page by page: threat model and ASR eq. (1) (p.3), role pipeline and attack definitions (pp.3-4), Table 1 and N = 225/183 (p.4), 98.9% / 1.9x / 13.8-53.3% (p.4), Fig. 3 macro-averages and Table 2 (p.5), poison sweep 19.1/28.4/30.6 and 13-52% (p.5, Table 12 p.13), BUY 40.0 / HOLD 33.8 / SELL 26.2 (p.5, Table 7 p.11), APS definition and values (p.6), all 24 GPT-4.1 cells of Table 3 (p.6), r = 0.85/0.88, 44.6%, 68.7%, Fig. 8 ranges (p.7), Table 4 (p.8), limitations and conclusion (p.8), Appendix A payloads (pp.9-11), 2026 Q1 dates, 61 days, 305 asset-days and sources (p.11), 295/305 HOLD (p.11 and Table 8 p.12), model configuration and compute scale (p.12), Table 10/11 pooled cells 41.7 and 214/224 (pp.12-13), backtest rules and "No transaction costs" (p.14), moving-block bootstrap (p.15), Table 16 rho = 0.183 [0.057, 0.305] (p.16). All four verbatim quotes match exactly (pp.2, 4, 5, 8). The design_v1 facts in the relevance section (W_clean 2024-07-05..2026-09-11, 115 decisions, PIT top-100, 15 bps/side) were confirmed in docs/design/design_v1.md.
- Corrections made: (1) the Table 3 row "Market (A1)" is objective hijacking of the Market analyst, but the summary never said so while also using "A1" for role-axis data poisoning; the architecture-axis bullet now defines D1/D2/A1 and flags the paper's label reuse; (2) the relevance bullet on mean aggregation cited the 55.7% (agent-level objective hijacking) as if it were a text attack; it now also gives the data-level Centralized ASRs 5.6% / 21.0% (Table 3, p.6); (3) added the missed A5 inconsistency (main text "hypothetical framing" vs. appendix authority-override payload; the hypothetical variant is "unused-in-the-main-results", pp.4, 10-11); (4) the quorum-effect numbers are GPT-4.1 BUY-targeted, now stated; (5) the "only v1 exists" claim was re-checked and dated.

