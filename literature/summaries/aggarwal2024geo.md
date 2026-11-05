# aggarwal2024geo: GEO: Generative Engine Optimization

## Verified metadata
- Title: GEO: Generative Engine Optimization [source: arxiv, crossref]
- Authors: Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, Karthik Narasimhan, Ameet Deshpande [source: arxiv, crossref]. The PDF lists the Indian Institute of Technology Delhi, Princeton University and two independent authors in Seattle (p.1).
- Venue: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24), pp. 5–16, ACM [source: crossref; Crossref event name "KDD '24: The 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining"]. The PDF carries the ACM reference format "KDD ’24, August 25–29, 2024, Barcelona, Spain", ISBN 979-8-4007-0490-1 and the DOI (p.1). The arXiv comment reads "Accepted to KDD 2024" [source: arxiv].
- Year: 2024 [source: crossref; published online 2024-08-24, print 2024-08-25]. arXiv v1 is dated 2023-11-16 and v3 2024-06-28 [source: arxiv].
- DOI / arXiv: 10.1145/3637528.3671900 / arXiv:2311.09735 (v3)
- Metadata files: literature/meta/aggarwal2024geo.arxiv.xml, literature/meta/aggarwal2024geo.crossref.json
- Note: DBLP answered with a bot-check page on 2026-09-26, so no DBLP record was saved. The venue is confirmed by Crossref.

## PDF
- Path: literature/pdf/aggarwal2024geo.pdf. This is arXiv 2311.09735v3, downloaded from arxiv.org. It is typeset in the ACM KDD '24 camera-ready format with the DOI on p.1. The ACM DL PDF (dl.acm.org/doi/pdf/…) returned HTTP 403. The title on p.1 matches.
- Text: literature/text/aggarwal2024geo.txt ; pages: 12

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - Systems such as BingChat, Google's SGE and perplexity.ai combine search engines with generative models. The authors call them "generative engines" (GEs): they retrieve sources and generate a response grounded in them (p.1), embedding websites as inline citations (p.2).
  - GEs answer directly, so users need not visit websites. This can reduce organic traffic to websites. Because GEs are black boxes, content creators find it hard to control how their content is ingested and portrayed (p.1–2).
  - The paper proposes Generative Engine Optimization (GEO): a creator-centric, black-box framework for making web content more visible in GE responses (p.2).
- Method / architecture:
  - **GE formalization (p.3).** A GE is a function f_GE: (q_u, P_U) → r, where P_U is personalized user information. It uses a set of generative models (query reformulation G_qr, per-source summarization G_sum, response generation G_resp) and a search engine SE that returns sources. Appendix A extends this to multi-turn conversational GEs (p.11).
  - **Visibility ("impression") metrics.**
    - Word Count, Imp_wc: the normalized word count of the sentences that cite a source (Eq. 2, p.3).
    - Position-Adjusted Word Count, Imp_pwc: word count weighted by an exponentially decaying function of citation position (Eq. 3, p.4).
    - Subjective Impression: seven facets (relevance, influence, uniqueness, subjective position, subjective count, click likelihood, diversity), each scored with GPT-3.5 in a G-Eval-style template (p.4–6).
    - Because "G-Eval scores are poorly calibrated", subjective scores are normalized to the mean and variance of Position-Adjusted Word Count (p.6). All impressions in a response are scaled to sum to 1. Methods are compared by relative improvement in a source's impression (Eq. 4, p.6).
  - **Nine GEO methods (p.4).** Each is a rewrite f: W → W′ of a source's content, performed by an LLM prompted for a specific style or content change:
    - Authoritative; Statistics Addition; Keyword Stuffing; Cite Sources; Quotation Addition; Easy-to-Understand; Fluency Optimization; Unique Words; Technical Terms.
    - Except for Keyword Stuffing, Cite Sources and Quotation Addition, the methods change only presentation and need no extra content (p.4).
    - Per query, one source is chosen at random and optimized with each method separately (p.5–6).
    - "GPT-3.5 turbo was used for all experiments" (p.12).
- Data, universe, evaluation period, baselines (for trading papers: tickers/universe size, dates, frequency, costs modeled?, LLMs used):
  - Not a trading paper.
  - **Evaluated GE (p.5).** Two steps: the top 5 Google search results are fetched, then gpt3.5-turbo writes the answer with the prompt of prior work. There is no summarization step. Five responses are sampled at temperature 0.7.
  - **GEO-bench (p.5).**
    - 10K queries from nine sources: MS MARCO, ORCAS-I, Natural Questions, AllSouls, LIMA, Davinci-Debate, Perplexity.ai Discover, ELI-5 and GPT-4-generated queries. (The body text spells them "MS Macro", "ORCAS-1" and "Davinci-Debtate"; reference [1] is "ORCAS-I", p.5, p.9.)
    - Split 8K/1K/1K (train/validation/test); 80% informational, 10% transactional and 10% navigational queries.
    - Each query comes with the cleaned text of its top 5 Google results. Queries are tagged by GPT-4 (25 domains, 7 categorizations), with manual verification on the test split.
    - The authors caution that the tags "can be noisy" (p.11).
  - **Baseline and protocol (p.5).** Unmodified sources, evaluated on the full test split, averaged over five random seeds.
  - **Deployed GE (p.12).** Perplexity.ai, with source texts uploaded as files, on a 200-sample subset of the test set.
- Key results (copy numbers EXACTLY, with page):
  - **Table 1 (p.6), overall Position-Adjusted Word Count / Subjective Impression average:**
    - No Optimization 19.3 / 19.3;
    - Keyword Stuffing 17.7 / 20.2;
    - Fluency Optimization 24.7 / 21.9;
    - Cite Sources 24.6 / 21.9;
    - Statistics Addition 25.2 / 23.7;
    - Quotation Addition 27.2 / 24.7.
  - **Headline gains.** "The best methods improve upon baseline by 41% and 28% on Position-Adjusted Word Count and Subjective Impression respectively" (p.6). Cite Sources, Quotation Addition and Statistics Addition "achieved a relative improvement of 30-40% on the Position-Adjusted Word Count metric and 15-30% on the Subjective Impression metric" (p.6).
  - **Presentation alone matters.** Fluency Optimization and Easy-to-Understand gave "a significant visibility boost of 15-30%" (p.6).
  - **Tone and keywords do not (per the text).** The authors write that a more authoritative tone gave "no significant improvement" (p.7), yet Table 1 lists Authoritative among the "High-Performing" methods at 21.3 / 22.9 against 19.3 / 19.3 (p.6). Keyword stuffing offers "little to no improvement" (p.7); in Table 1 it is below the baseline on Position-Adjusted Word Count (17.7 vs 19.3) (p.6).
  - **All sources optimized (Table 2, p.7–8).** Lower-ranked websites gain most. Cite Sources raised visibility by 115.1% for websites ranked fifth in the search results, while the top-ranked website's visibility fell by 30.3% on average (p.8).
  - **Combinations (p.8).** Fluency Optimization plus Statistics Addition outperforms any single strategy by more than 5.5% (on a 200-example subset).
  - **Perplexity.ai (Table 5, p.8–9).**
    - No Optimization 24.1 / 24.7; Keyword Stuffing 21.9 / 28.1; Quotation Addition 29.1 / 32.1; Statistics Addition 26.2 / 33.9.
    - Quotation Addition gives "a 22% improvement over the baseline" on Position-Adjusted Word Count, and Keyword Stuffing "performs 10% worse than the baseline" (p.9).
  - **Qualitative examples (Table 4, p.8).** Three rewrites (Cite Sources, Statistics Addition, Authoritative) raise a source's visibility by 132.4%, 65.5% and 89.1%. The authors describe them as adding no "substantial new information", although the first adds an attributed survey ("The International Chocolate Consumption Research Group") and the second a "staggering 70% increase" statistic (p.8).
- Limitations:
  - **Stated by the authors (p.9):**
    - Methods may need to adapt as GEs evolve.
    - Query distributions change over time.
    - The effect of GEO rewrites on search-engine rankings was not evaluated.
    - Query tags may contain labeling errors.
  - **Our observation:**
    - The main GE is a simulated two-step gpt-3.5-turbo pipeline (p.5). Only Perplexity.ai, on 200 samples, is a deployed system (p.12).
    - Visibility is measured as a share of cited text in a generated answer. It is not the effect on a downstream decision.
    - The framing is non-adversarial. The paper contrasts itself with Kumar and Lakkaraju's adversarial text (p.9).
    - No experiment concerns financial news or investment decisions.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "Generative Engines typically satisfy queries by synthesizing information from multiple sources and summarizing them using LLMs." (p.1)
- "This suggests that Generative Engines value not only content but also information presentation." (p.6)
- "Without adding any substantial new information, GEO methods significantly increase the visibility of the source content." (p.8)
- "the Cite Sources method led to a substantial 115.1% increase in visibility for websites ranked fifth in SERP" (p.8)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Web framing (design §1.1).** A KDD paper that formalizes generative engines (LLM-based readers of Web sources, p.3), argues they disadvantage "the third stakeholder – website and content creators" (p.1), and shows that the source texts they read can be rewritten to change how much of the answer those texts get (p.6).
  - It supports our premise that Web text may increasingly be written with machine readers in mind: GEO explicitly proposes this to content creators (p.2). That makes "how machine readers of the open Web can act on it safely" a Web research question, not just the use of a Web API.
  - By extension (not tested in the paper), a news item or press release could be rewritten in GEO style to gain influence over an LLM reader, without any explicit instruction.
- **A threat family between benign and adversarial (C1, §13.1).**
  - GEO's strongest rewrites add statistics, quotations or citations, or improve fluency (p.6). The authors say their examples add no "substantial new information" (p.8), although Cite Sources and Quotation Addition "may require some form of additional content" (p.4–5) and the Table 4 examples insert an attributed survey and a statistic (p.8). These are exactly the edits that screening for injected instructions (the D2 `instructs_reader` guard) would not flag.
  - A2 (fabricated authoritative fact) and A3 (self-labeling) partly cover this. We could add a GEO-style "presentation-only" rewrite family to the red team, or to a robustness check on clean items, to measure how much Layer-A answers move under meaning-preserving rewrites.
  - The band guard (§3.4) bounds the worst case in either case.
- **Visibility vs. decision influence.** GEO measures influence as a share of cited text in generated answers (p.3–4, p.6). Our decision cards give exact per-item contributions to the logit (§6.4, §10.1), so "how much did this item move the decision" is exact rather than judged by an LLM. This contrast with generative readers is worth one sentence in related work.
- **Calibration caution.** GEO normalizes G-Eval scores because they are "poorly calibrated" (p.6). This is one more example that LLM-emitted scores need care, consistent with our rule that emulator outputs are never called "calibrated" (§11.2).
- **How we differ.** GEO optimizes content for a generative answer engine and measures citation visibility. We build a typed-decision reader that trades, and we measure portfolio-level attack success under a trusted-channel influence bound. Their "content creator" is our potential attacker (a stock promoter or issuer PR).

## BibTeX
```bibtex
@inproceedings{aggarwal2024geo,
  author    = {Aggarwal, Pranjal and Murahari, Vishvak and Rajpurohit, Tanmay and Kalyan, Ashwin and Narasimhan, Karthik and Deshpande, Ameet},
  title     = {{GEO}: Generative Engine Optimization},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  series    = {KDD '24},
  pages     = {5--16},
  year      = {2024},
  publisher = {ACM},
  address   = {New York, NY, USA},
  location  = {Barcelona, Spain},
  isbn      = {979-8-4007-0490-1},
  doi       = {10.1145/3637528.3671900}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- Metadata checked against the saved raw records: title, six authors and order, arXiv id 2311.09735 (v1 2023-11-16, v3 2024-06-28, comment "Accepted to KDD 2024") against `meta/aggarwal2024geo.arxiv.xml`; container title, pages 5–16, DOI 10.1145/3637528.3671900, event "KDD '24" (Barcelona), online 2024-08-24 / print 2024-08-25 and affiliations against `meta/aggarwal2024geo.crossref.json`. ISBN 979-8-4007-0490-1, the KDD '24 date line and the 12-page count come from PDF p.1. DBLP (dblp.org and both mirrors) still returned a bot-check page on this date, so no DBLP record was added; the venue is supported by Crossref and the PDF. The BibTeX matches these records.
- PDF identity: `pdf/aggarwal2024geo.pdf` is arXiv 2311.09735v3 (12 pages, title and DOI on p.1).
- Checked against the PDF text, with page numbers: all Table 1 values (p.6, read from the layout text), Table 2 (p.7), the Table 4 percentages (p.8), Table 5 (p.8), the footnote on the 200-example combination subset (p.8), the Perplexity setup (p.12), GEO-bench composition, splits and tags (p.5, p.11), the evaluated-GE setup (p.5), Eq. 1–4 (p.3–6), the limitations (p.9), and all four verbatim quotes plus every in-line quote (they match apart from whitespace).
- Corrected:
  - The "inline citations" part of the GE description is on p.2, not p.1.
  - Added the paper's own spellings of three dataset names ("MS Macro", "ORCAS-1", "Davinci-Debtate", p.5).
  - "Authoritative gives no significant improvement" is the authors' text claim (p.7). Table 1 nevertheless lists Authoritative among the "High-Performing" methods (21.3 / 22.9 vs 19.3 / 19.3). Both are now stated. Added that Keyword Stuffing is below the baseline on Position-Adjusted Word Count in Table 1.
  - Table 4: "no substantial new information" is now attributed to the authors. The examples in fact insert an attributed survey and a "70% increase" statistic (p.8).
  - Relevance: the paper's "third stakeholder" is website and content creators, not LLM readers. Fixed the misattribution. "Web text is now written for machine readers" was not shown by the paper; it is now framed as GEO's proposal (p.2). The news/press-release point is now marked as our extrapolation. "With no new substance" is replaced by the paper's actual statements (p.4–5, p.8).
