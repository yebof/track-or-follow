# nestaas2025adversarial: Adversarial Search Engine Optimization for Large Language Models

## Verified metadata
- Title: Adversarial Search Engine Optimization for Large Language Models [source: publisher (proceedings.iclr.cc abstract page and BibTeX), arxiv]
- Authors: Fredrik Nestaas, Edoardo Debenedetti, Florian Tramèr [source: arxiv, publisher]. The publisher metadata spells the last name "Tramer". The PDF lists ETH Zurich (p.1).
- Venue: International Conference on Learning Representations (ICLR 2025), pp. 4857–4888 [source: publisher, ICLR proceedings abstract page and BibTeX, editors Y. Yue, A. Garg, N. Peng, F. Sha, R. Yu]. The PDF header reads "Published as a conference paper at ICLR 2025" (p.1).
  - The candidate list called it a "poster". This could not be verified, because OpenReview returned a challenge page (HTTP 403) on 2026-09-26. The presentation type is therefore not claimed.
- Year: 2025 [source: publisher; citation_publication_date 2025-05-01]. arXiv v1 is dated 2024-06-26 and v2 2024-07-02 [source: arxiv].
- DOI / arXiv: no DOI / arXiv:2406.18382 (v2)
- Metadata files: literature/meta/nestaas2025adversarial.arxiv.xml, literature/meta/nestaas2025adversarial.iclrproc.html (ICLR proceedings abstract page), literature/meta/nestaas2025adversarial.iclrproc.bib (publisher BibTeX)
- Note: DBLP answered with a bot-check page on 2026-09-26. The venue is confirmed by the ICLR proceedings site, the publisher.

## PDF
- Path: literature/pdf/nestaas2025adversarial.pdf. This is the ICLR 2025 proceedings PDF from proceedings.iclr.cc (…/2025/file/0f12b3c36a781120c4f60e90e855868d-Paper-Conference.pdf). The OpenReview PDF returned HTTP 403. The title on p.1 matches.
- Text: literature/text/nestaas2025adversarial.txt ; pages: 32

## Summary (grounded; every factual sentence cites a page like (p.4), or (abstract) for metadata-only)
- Problem:
  - LLMs increasingly select among competing third-party content, e.g., websites in LLM search engines or plugins in AI assistants (p.1).
  - The paper introduces **Preference Manipulation Attacks**: crafted web-page text or plugin descriptions that make an LLM promote the attacker's product or discredit competitors (p.1–2).
  - It combines elements of prompt injection, black-hat SEO and LLM "persuasion" (p.1). It works with explicit embedded instructions and "also with forms of misinformation without explicit instructions" (p.1).
- Method / architecture:
  - **Threat model (p.4).**
    - The adversary owns a web page or plugin, and can therefore deliver arbitrary text into the LLM's context.
    - Their goal is to have the LLM favor their product over competitors, who may also attack.
    - Motives may be economic or ideological. Attacks may be stealthy.
    - How the attacker's page gets into the context (e.g., via traditional SEO) is out of scope.
  - **Attacks (p.5, p.16–17).**
    - Simple injection techniques taken from the literature, not optimized for strength.
    - Examples:
      - hidden "[system]" instructions to recommend only one product;
      - instructions to censor a news outlet;
      - plugin descriptions claiming that competing plugins were "developed by unethical organizations" involved in scandals.
    - Attacks can be hidden as illegible text (white text, font size 1) (p.4, p.9).
  - **Metrics (p.5, p.15).**
    - For search engines: recommendation rate and citation rate of a target product. A regular expression detects "I recommend …" statements, and citations are the fallback.
    - For plugins: the rate at which the attacker's plugin is called.
    - The authors checked "hundreds of responses" manually and consider the automatic schemes accurate enough (p.15).
- Data, universe, evaluation period, baselines (for trading papers: tickers/universe size, dates, frequency, costs modeled?, LLMs used):
  - Not a trading paper.
  - **Systems (p.4).** Production LLM search engines (Bing Copilot and Perplexity) and plugin-enhanced LLMs (Anthropic's Claude 3 and OpenAI's GPT-4). Figures also report GPT-4 Turbo and Claude 3 Opus (p.7–8).
  - **Test pages (p.4–5).** 50 dummy web pages on the authors' domain, listing fictitious cameras, books and news. Real products from established brands serve as comparators.
  - **Queries (p.5).** They explicitly ask the engine to search the authors' domain, because the dummy pages do not rank highly. The authors note that this "may introduce an experimental bias".
  - **Baselines.** The no-attack condition, and the same products without injections (p.5–7).
- Key results (copy numbers EXACTLY, with page):
  - **Boosting (Table 1, p.6).** In Bing Copilot, a fake camera's recommendation rate rose from 34.0% to 59.4% after the attack, while real cameras (Nikon or Fujifilm) stayed at 57.9%.
  - **Among fictitious products only (p.6).** The attacker's camera became "2.5× more likely to be recommended" than a comparable product.
  - **Highly ranked pages (p.6).** A footnote injection on the lab's GitHub page made GPT-4o and Perplexity output the injected sentence. Bing Copilot and Perplexity often did not cite the lab's main website at all.
  - **Plugins (p.1, p.6).** A news plugin became "2–8× more likely to be selected by GPT-4" (p.1) and "up to 7.2× more likely to be selected than other news plugins". "In some cases, our attacks boost a plugin's selection rate from 0% to over 90%" (p.6).
  - **Prisoner's dilemma (p.7–8, Figures 5–6).** Each product owner gains by attacking regardless of how many others attack, but all lose recommendation rate relative to universal cooperation. Claude 3 Opus (via Perplexity) often refuses to recommend anything when it meets multiple attacks (p.8).
  - **External attacks (p.7–8).** Text on one page can promote or censor another page. On Bing these succeed "at most 25%, compared to 95%-100%" for direct attacks; Perplexity is more susceptible. "In 80% of successful attacks, the LLM does not mention the external web page that contains the injection."
  - **Order effect (p.8).** Attacks are most successful when they sit in the last page seen by the LLM.
  - **Wording overlap (p.8).** Injections that echo the user's request succeed more often: the attacker's camera is the only product mentioned 36% of the time, "contra 0%-9% otherwise".
  - **Plugins without instructions (p.8–9).** Merely presenting a plugin favorably did not bias the LLM. Highly exaggerated claims about the plugin or its competitors succeeded without explicit instructions.
- Limitations:
  - **Stated by the authors:**
    - They do not seek the strongest attack or cover all consequences (p.10).
    - Experiments are isolated, and the authors control all adversarial pages (p.10).
    - An end-to-end attack would also need traditional SEO (p.10).
    - Results cannot be exactly replicated: the systems are black boxes and non-deterministic, indexes change, and Microsoft changed Bing after disclosure (p.11).
    - Defenses could not be evaluated directly on the proprietary systems (p.9).
  - **Our observation:**
    - All targets are generative text readers (search answers and plugin choices). No typed or probability-returning decision model is tested.
    - Success rates come from small repeated trials on live systems. For example, the Perplexity model comparisons use 10 trials per model (p.19).
    - There is no finance setting.

## Verbatim quotes (2-4 short quotes, each with page number, copied exactly from the text)
- "carefully crafted website content or plugin documentations can trick an LLM to promote the attacker products and discredit competitors" (p.1)
- "attackers individually benefit from launching attacks to boost their content, but when multiple attackers target the same LLM, all parties lose in search presence." (p.2)
- "These results suggest that a successful defense against prompt injection attacks may not be sufficient to defend against Preference Manipulation Attacks" (p.9)
- "reliable data attribution remains an unsolved problem" (p.9)

## Relevance to our project (Jev-driven multi-agent trading; which of our design decisions / claims / baselines / evaluation choices it informs; how we differ)
- **Web framing and C1 (design §1.1, §3.1, §13.1).** This is a peer-reviewed demonstration on production LLM search engines that third-party Web text is written to steer LLM selection. The motive is commercial: traffic and monetization (p.1). A stock promoter or issuer who plants news aimed at an LLM trader is the finance analog, and it fits our goals:
  - G1 (promotion) corresponds to their boosting.
  - G2 (demotion) corresponds to "discredit competitors" and censoring. Their plugin injection that accuses competitors of scandals (p.17) resembles our A8 (fabricated serious problem).
- **Attack families.** Their techniques map onto our A1–A6:
  - explicit instructions to AI readers correspond to A1 and A4;
  - exaggerated self-promotion corresponds to A3 and A5;
  - planted pages correspond to A6.
  - Their finding that non-instruction persuasion also works (p.8–9) supports keeping A2 (fabricated authoritative fact) and A3 in the red team. It also supports not relying only on the `instructs_reader` guard (D2): the band guard (§3.4) is the backstop.
- **Position effects.** Attacks work best in the last page the LLM reads (p.8). Our item-level N1 questions read one item per question, and aggregates are recomputed in code (§10.3). So a planted item's position in a list should matter less than for a holistic reader. We have not tested this; it is a check we could add when comparing H/HS readers with the typed reader in C1.
- **Attribution as a defense.** They discuss attributing decisions to data as a defense (p.2, p.9), but note that "reliable data attribution remains an unsolved problem". They also note that exposing attributions may help attackers (p.9).
  - Our decision cards give exact per-item contributions by construction (§6.4, §10.1), and the localization metric (§10.3) tests whether a planted item ranks first.
  - Their caveat argues for keeping cards internal. It also argues for keeping our Kerckhoffs assumption: the attacker knows the architecture (§3.1).
- **Prisoner's dilemma.** When several product owners attack, all lose recommendation rate relative to universal cooperation (p.7–8), and the abstract says this "collectively degrades the LLM’s outputs for everyone" (p.1). By analogy (our extrapolation), if many issuers target LLM readers, text could lose information value. This is a Web-economics motivation for the trusted-channel partition, where prices, filings and macro data cannot be edited by text (§3.2).
- **How we differ.** They attack generative search and plugin selection and report recommendation rates on non-replicable live systems (p.11). We study a trading system whose typed judgments are logged and replayable, and we measure portfolio-level attack success under a mechanically verified influence bound.

## BibTeX
```bibtex
@inproceedings{nestaas2025adversarial,
  author    = {Nestaas, Fredrik and Debenedetti, Edoardo and Tram{\`e}r, Florian},
  title     = {Adversarial Search Engine Optimization for Large Language Models},
  booktitle = {International Conference on Learning Representations (ICLR 2025)},
  editor    = {Yue, Y. and Garg, A. and Peng, N. and Sha, F. and Yu, R.},
  volume    = {2025},
  pages     = {4857--4888},
  year      = {2025},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2025/file/0f12b3c36a781120c4f60e90e855868d-Paper-Conference.pdf}
}
```

## Verification
- Date: 2026-09-25 (adversarial fact-check).
- Metadata checked against the saved raw records: title, three authors (arXiv and PDF spell "Tramèr", the publisher "Tramer"), arXiv 2406.18382 (v1 2024-06-26, v2 2024-07-02) against `meta/nestaas2025adversarial.arxiv.xml`; venue "International Conference on Learning Representations", volume 2025, pages 4857–4888, editors, and citation_publication_date 2025-05-01 against `meta/nestaas2025adversarial.iclrproc.html` and `.iclrproc.bib`. The page range (32 pages) matches the 32-page PDF, whose header reads "Published as a conference paper at ICLR 2025" (p.1). DBLP (dblp.org and both mirrors) and the OpenReview API both returned bot or challenge pages on this date, so the presentation type is still not claimed. The BibTeX matches the publisher record.
- PDF identity: `pdf/nestaas2025adversarial.pdf` is the ICLR 2025 proceedings version; title and authors on p.1 match.
- Checked against the PDF text, with page numbers: the threat model (p.4), setup and queries (p.4–5), metrics and the regular-expression scheme (p.5, p.15), injection examples (p.16–17), Table 1 (p.6), the 2.5×, 7.2×, 0%→90% and 2–8× figures (p.1, p.6), Figures 5–6 (p.7–8), the external-attack numbers 25%, 95%-100% and 80% (p.8), the 36% vs 0%-9% overlap result (p.8), plugin persuasion (p.8–9), defenses (p.9), limitations (p.10), reproducibility (p.11), and the 10 trials per Perplexity model (p.19). All four verbatim quotes and every in-line quote match (apart from typographic apostrophes and whitespace).
- Corrected:
  - The external-attack section starts on p.7; the citation is now p.7–8.
  - Attribution defenses are discussed, not proposed, by the authors (p.2, p.9). Wording fixed.
  - Relevance: "all text loses information value" was attributed to the paper, which only shows that all parties lose recommendation rate (p.7–8) and that outputs degrade for everyone (p.1). The information-value point is now marked as our extrapolation.
