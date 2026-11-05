# ling2026jev: Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem

## Verified metadata
- Title: Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem [source: arxiv (OAI-PMH, abs page); PDF p.1]
- Authors: Guoming Ling, Muen Xue, Zijian Ye [source: arxiv]. The PDF lists Sun Yat-sen University and The Chinese University of Hong Kong (p.1).
- Venue: arXiv preprint, primary category cs.SE [source: arxiv]. No comments, no journal reference. Crossref search returned unrelated items (literature/meta/ling2026jev.crossref_search.json).
- Year: 2026 (v1 submitted 2026-09-24) [source: arxiv, datacite]
- DOI / arXiv: 10.48550/arXiv.2609.30216 / arXiv:2609.30216 (v1)
- Metadata files: literature/meta/ling2026jev.arxiv_oai.xml, .arxiv_abs.html, .datacite.json, .crossref_search.json

## PDF
- Path: literature/pdf/ling2026jev.pdf (arXiv 2609.30216v1; stamp p.1)
- Text: literature/text/ling2026jev.txt ; pages: 11

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem: how Jev is used across applications and how public attention relates to project counts (p.1).
- Method: 2,170 public GitHub projects that use Jev for a concrete task, collected as of 2026-09-22 by repository and code search; each repository was verified and annotated by a GPT-6 Luna Max agent, with key labels reviewed by a second agent (p.2-3).
- Key results (numbers copied exactly):
  - In the first week after the 2026-09-15 release the ecosystem grew to 1,865 new repositories and 43,750 stars (Fig. 1, p.1).
  - Eight categories, 18 subcategories; the largest are Content & Expert Tasks (17.8%) and Search & Memory (17.6%) (Table 1, p.2; p.3). "Professional Tasks" has 176 projects (Table 1, p.2).
  - Routing & Automation receives 41.4% of stars, Model & Tool Routing alone 32.6% (p.3). Action selection is 52% of purpose labels in Simulation & Control and 47% in Interface Agents (p.4).
- Limitations: stated by the authors, a snapshot as of 2026-09-22 of public repositories only; private and commercial applications are out of scope (p.5). Our observation: annotation is by an LLM agent; the paper does not break out a finance or trading category.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "These patterns suggest that Jev serves as a reusable decision component whose functionality varies with the surrounding workflow." (p.1)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Scope of any "no prior S1 trading system" statement.** Within a week of release there were 2,170 public Jev projects, including agents that use Jev for action selection (p.1-4). The paper does not report a finance category, so we cannot know whether Jev trading bots exist; positioning's corpus-scoped statement ("none of the 43 LLM trading systems in our table", positioning K1) is the only defensible form, and the paper text should add that public Jev projects exist in large numbers and were not surveyed.
- Supports the Web-infrastructure framing (typed decision models as a reusable Web-served component, p.1) with measured adoption data rather than vendor claims.

## BibTeX
```bibtex
@article{ling2026jev,
  title   = {{Jev} in the Wild: A Data-Driven Analysis of the {Jev} Model's Functionality, Applications and Ecosystem},
  author  = {Ling, Guoming and Xue, Muen and Ye, Zijian},
  journal = {arXiv preprint arXiv:2609.30216},
  year    = {2026}
}
```

## Verification
- Date: 2026-09-26 (novelty-review round r2).
- Checked: metadata against OAI-PMH, abs page, DataCite; counts and shares against the text (Fig. 1 p.1, Table 1 p.2, p.3-5); searched the text for finance, trading, stock and invest terms (no occurrences in the analysis).
- Quotes were checked against a reading-order pdftotext extraction because the saved -layout text interleaves the two columns.
