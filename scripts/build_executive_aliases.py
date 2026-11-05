"""Point-in-time person aliases (executives, directors, founders) from the study firms' own 10-K filings
(docs/impl/dev2_blind.md §4; directors and founders since the freeze, docs/impl/freeze_report.md).

A firm's well-known executives identify it as surely as its name ("Tesla CEO Elon Musk"; 213 of DEV #1's TSLA
states named Musk), and so do its founders and directors ("Company-X was founded by Bill Gates and Paul Allen"; DEV #2
dev2_report §10). The bank declares one person mask, `[executive]` (`conventions.masks.executive`); this script builds
the table the blinder masks with (`data.blinding.load_executives`). Sources, in order:

1. the 10-K's own list of executive officers (Reg. S-K Item 401(b), "Information about our Executive Officers",
   "Executive Officers of the Registrant", ...; Part I or Item 10), when the filing carries it;
2. the 10-K's signature page: every signer whose capacity is an officer or a board chair ("Chief Executive Officer",
   "Senior Vice President, Chief Financial Officer", "Chair of the Board", ...) is an executive (role "executive");
2b. (freeze) the same page's directors (role "director", source `signatures_director`; `director_signers`): Form 10-K
   is signed by the principal officers and at least a majority of the board (General Instruction D(2)), so every
   other natural-person signer whose capacity names the board or is not printed, and the directors signed by an
   attorney-in-fact and printed in the page's layouts (a lone "*", a list under "Directors", "Name, Director"), is a
   director. Officers are left to rule 2; attorneys-in-fact and firms (auditors) are not directors;
3. `data/raw/aliases/manual_executives.csv`: a small hand list, each row with a source document;
4. (freeze) `data/raw/aliases/manual_founders.csv` (role "founder"): the founders who are neither an executive nor a
   director in any 10-K of the store, as a minimal hand list; each row cites the firm's own SEC filing that names the
   person as a founder, and is valid from that filing's usable time.

Every row of `executive_aliases.csv` is masked as `[executive]` with the same rules, whatever its `roles`. A director
who is named mostly elsewhere (another firm's CEO, a public official, a namesake; `director_attribution`, measured on
pre-W_clean news) is not masked and is listed in `director_exclusions.csv` instead.

Every 10-K of `data/raw/sec/business/<SYM>/<filed>_<accession>.json` (the F-A store: the study firms' 10-Ks filed
2017-06..2026-09) is re-fetched from EDGAR once (its `url`, the primary document) and re-parsed; only the two sections
are kept, in `data/raw/aliases/tenk_officers/<SYM>/<filed>_<accession>.json`. A person listed in a 10-K is an
executive (or director) of that firm from the filing's usable time (the open of the session after the filing date, as
F-A reads the 10-K; `usable_from`), never before, and stays one (a former executive or director still identifies the
firm: Bill Gates left MSFT's board in 2020).

Point in time per form (v4, DEV #2 audit). Each full-name form is masked from the usable time of the first listing
that yields it (`form_starts`, aligned with `forms`), not from the person's first listing: a form first printed in a
later 10-K ("Leeny Oberg", "Jack F. Schultz") was known only from then.

Surname alone. A full name ("Elon Musk", "Timothy D. Cook", "Tim Cook") is always masked. The surname alone ("Musk
said") is masked in a text tagged to the firm or naming the person in full (the blinder decides); everywhere else only
if the surname is *distinctive*, measured point in time on the news archive: over the items created and last edited
before the listing's usable time (and before CORPUS_END, so that no W_clean-era text is read), at least MIN_ITEMS
items carry the capitalized surname and at least MIN_SHARE of them also name the person in full or are tagged to the
firm, and at most MAX_OTHER of its occurrences follow another given name. The first listing at which this holds is
`surname_from`. A surname that is a common English word (`surname_word`: in at least COMMON_WORD_DOCS 10-K documents
of the Loughran-McDonald master dictionary, e.g. "Cook", "Smith", "Call"; "Musk" is in 356) never gets `surname_from`,
and the blinder masks it alone only in a person context (v4, DEV #2 audit: "Fed's Cook Says", "Cook County").

    .venv/bin/python scripts/build_executive_aliases.py fetch [--workers 6] [--symbols A,B]
    .venv/bin/python scripts/build_executive_aliases.py build

Outputs: `data/raw/aliases/executive_listings.csv` (one row per person and 10-K or hand-list row: the evidence, with its
`source`) and `data/raw/aliases/executive_aliases.csv` (one row per firm and person: what the blinder reads; `roles` is
"executive", "director", "founder" or a "+"-join of them).
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BUSINESS = ROOT / "data/raw/sec/business"
OUT = ROOT / "data/raw/aliases"
TENK = OUT / "tenk_officers"
UA = {"User-Agent": "jev-quant-research research@example.com"}
SECTION_CAP = 40000
CORPUS_END = "2024-07-01"  # surname statistics read news items created before this (never W_clean-era text)
MIN_ITEMS, MIN_SHARE, MAX_OTHER = 20, 0.8, 0.1
COMMON_WORD_DOCS = 10_000  # a surname in at least this many 10-Ks as a word (LM master dictionary) is a common word
# freeze (docs/impl/freeze_report.md): founders who are neither an executive nor a director in any 10-K of the store,
# a minimal hand list with an SEC source per row; roles of the listing sources (executive_aliases.csv `roles`)
FOUNDERS_FILE = "manual_founders.csv"
DIRECTOR_EXCLUSIONS_FILE = "director_exclusions.csv"  # director rows named mostly elsewhere (director_attribution)
DIRECTOR_MIN_ITEMS, DIRECTOR_MIN_SHARE = 5, 0.2
ROLE_OF_SOURCE = {"exec_section": "executive", "signatures": "executive", "manual": "executive",
                  "signatures_director": "director", "manual_founder": "founder"}

# ------------------------------------------------------------------------------------------------ fetch
_EXEC_HEAD = re.compile(
    r"(?i)^\W{0,3}(?:item\s*\d+[a-z]?\.?\s*)?(?:information\s+(?:about|regarding|concerning|with respect to)\s+)?"
    r"(?:our\s+|the\s+)?(?:current\s+)?executive\s+officers(?:\s+(?:of|and)\s+(?:the\s+)?(?:registrant|company|"
    r"corporation|directors)[\w\s,]*)?(?:\s*\([^)]*\))?\W{0,3}$")
_SIGN_HEAD = re.compile(r"(?i)^\W{0,3}signatures?\W{0,3}$")
_END_HEAD = re.compile(r"(?i)^\W{0,3}(?:item\s*\d+[a-z]?\b|part\s+(?:i{1,3}|iv)\b|forward[- ]looking statements|"
                       r"signatures?\W{0,3}$|exhibit index|power of attorney)")


_BLOCK = ("p", "div", "br", "tr", "td", "th", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "hr")


def _lines(html: bytes) -> list[str]:
    import lxml.html

    doc = lxml.html.document_fromstring(html)
    for bad in doc.xpath("//script|//style|//*[local-name()='header']"):  # ix:header holds hidden XBRL facts
        bad.drop_tree()
    # a line per block element (paragraph, table cell, line break); inline runs join without a break, so small-caps
    # names set as "/S/ W<span>ARREN</span> E. B<span>UFFETT</span>" stay one word
    for el in doc.iter(*_BLOCK):
        el.tail = "\n" + (el.tail or "")
    text = doc.text_content()
    text = re.sub(r"[ \t\xa0  ]+", " ", text)
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def sections(html: bytes) -> dict[str, str]:
    """The executive-officer section(s) and the signature section of a 10-K, as newline-joined lines."""
    lines = _lines(html)
    exec_parts: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if len(ln) <= 110 and _EXEC_HEAD.match(ln) and "corporate governance" not in ln.lower():
            body = [ln]
            j = i + 1
            while j < len(lines) and j - i < 400 and not (_END_HEAD.match(lines[j]) and len(lines[j]) <= 110):
                body.append(lines[j])
                j += 1
            if len(body) >= 4:  # a table-of-contents entry has no body
                exec_parts.append("\n".join(body))
            i = j
            continue
        i += 1
    sign = ""
    # the signature page: the first "SIGNATURES" heading followed by signers ("/s/" or an attorney-signed "NAME*"),
    # through the table of signers, which can repeat its "Signature | Title | Date" header after a page break (so a
    # later table header must not be taken as the start: Ford's CEO signs on the first page)
    heads = [k for k, ln in enumerate(lines) if _SIGN_HEAD.match(ln)]
    plural = [k for k in heads if re.match(r"(?i)^\W{0,3}signatures\W{0,3}$", lines[k])]
    for k in [*plural, *reversed(heads)]:
        body = lines[k:k + 900]
        if sum("/s/" in x.lower() for x in body[:300]) >= 1:
            stop = next((m for m in range(1, len(body)) if re.match(r"(?i)^(?:exhibit index|report of independent "
                                                                        r"registered public accounting firm)", body[m])),
                        len(body))
            sign = "\n".join(body[:stop])
            break
    return {"exec_section": "\n\n".join(exec_parts)[:SECTION_CAP], "signatures": sign[:SECTION_CAP]}


def _fetch_one(job: tuple[str, str, str, str, str]) -> tuple[str, str, int]:
    import requests

    sym, filed, acc, url, out = job
    status, content = 0, b""
    for a in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=90)
            status = r.status_code
            if status in (429, 503):
                time.sleep(10 * (a + 1))
                continue
            content = r.content if r.ok else b""
            break
        except requests.RequestException:
            time.sleep(5 * (a + 1))
    time.sleep(0.6)  # <= ~10 requests per second over 6 workers (SEC fair access)
    sec = sections(content) if content else {"exec_section": "", "signatures": ""}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"symbol": sym, "filed": filed, "accession": acc, "url": url, "status": status,
                                     **sec}))
    return sym, filed, status


def fetch(workers: int = 6, symbols: set[str] | None = None, redo: bool = False) -> None:
    jobs = []
    for p in sorted(BUSINESS.glob("*/*.json")):
        d = json.loads(p.read_text())
        sym = p.parent.name
        if symbols and sym not in symbols:
            continue
        out = TENK / sym / p.name
        if not d.get("url") or d.get("status") != 200:
            continue
        # --redo-signature-table: redo filings whose stored signature page starts at a repeated table header
        if out.exists() and (not redo or not json.loads(out.read_text()).get("signatures", "").startswith("Signature\n")):
            continue
        jobs.append((sym, d["filed"], d["accession"], d["url"], str(out)))
    print(f"{len(jobs)} 10-Ks to fetch", flush=True)
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch_one, j) for j in jobs]
        for f in as_completed(futs):
            sym, filed, status = f.result()
            done += 1
            if done % 50 == 0 or status != 200:
                print(done, sym, filed, status, flush=True)
    print("fetched", done, flush=True)


# ------------------------------------------------------------------------------------------------ parse
_PARTICLES = {"van", "von", "de", "del", "della", "der", "den", "da", "di", "du", "la", "le", "st.", "mac", "bin", "al",
              "dos", "das", "ter", "ten"}
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "phd", "ph.d.", "m.d.", "md", "cpa", "esq", "esq."}
_HONORIFICS = {"mr", "mr.", "ms", "ms.", "mrs", "mrs.", "dr", "dr.", "sir", "dame", "prof", "prof."}
_NOT_NAME = re.compile(
    r"(?i)\b(?:officer|president|chief|vice|chair|chairman|chairwoman|director|executive|age|name|position|title|"
    r"ceo|cfo|coo|cto|cio|evp|svp|services|web|worldwide|consumer|operations|consolidated|balance|sheets?|"
    r"statements?|income|cash|flows?|notes|schedule|index|equity|stockholders|shareholders|total|net|assets|"
    r"liabilities|revenue|segment|products?|systems|technologies|holdings|partners|capital|trust|bank|"
    r"proceedings|disclosures?|risk|factors|unresolved|staff|comments|safety|mine|legal|energy|properties|matters|"
    r"indiana|ohio|carolinas|florida|progress|piedmont|power|light|electric|gas|utility|utilities|market|"
    r"since|date|registrant|company|corporation|inc|corp|llc|board|member|treasurer|controller|secretary|counsel|"
    r"general|senior|head|group|global|financial|operating|principal|signature|capacity|attorney|business|"
    r"information|the|and|of|our|report|annual|form|table|contents|pursuant|securities|exchange)\b")
_TOKEN = r"(?:[A-Z][a-zA-Z'’\-]*\.?|[A-Z]\.|(?:van|von|de|del|della|der|den|da|di|du|la|le|St\.|Mac|dos|das))"
_NAME = re.compile(rf"^(?:{_TOKEN})(?:\s+(?:\"[A-Z][a-z]+\"|“[A-Z][a-z]+”|\([A-Z][a-z]+\)|{_TOKEN})){{1,5}}"
                   rf"(?:,?\s+(?:Jr\.?|Sr\.?|II|III|IV))?$")
_AGE = re.compile(r"^\(?(\d{2})\)?$")
_DIRECTORS_HEAD = re.compile(r"(?i)^\W{0,3}(?:information about (?:our )?|our |the )?(?:board of directors|directors|"
                             r"non-employee directors|directors of the registrant)\W{0,3}$")
_INLINE = re.compile(rf"^((?:{_TOKEN})(?:\s+(?:\"[A-Z][a-z]+\"|“[A-Z][a-z]+”|\([A-Z][a-z]+\)|{_TOKEN})){{1,5}}"
                     rf"(?:,?\s+(?:Jr\.?|Sr\.?|II|III|IV))?)\s*(?:,\s*(?:age\s*)?|\(|\s+)(\d{{2}})\b\)?")
_INLINE_TITLE = re.compile(rf"^((?:{_TOKEN})(?:\s+(?:{_TOKEN})){{1,4}}(?:,?\s+(?:Jr\.?|Sr\.?|II|III|IV))?)\s*[,–—-]?\s+"
                           r"(?=(?:Chairman|Chairwoman|Chair|Chief|President|Executive Vice|Senior Vice|Vice President|"
                           r"Group President|Corporate Vice|Treasurer|Controller)\b)")
OFFICER_TITLE = re.compile(r"(?i)\b(?:chief|officer|president|chair|chairman|chairwoman|chairperson|ceo|cfo|coo|founder|"
                           r"treasurer|controller|comptroller|general counsel|managing partner|principal executive|"
                           r"principal financial|principal accounting)\b")
_COMMITTEE_CHAIR = re.compile(r"(?i)\b(?:co-)?chair(?:man|woman|person)?\s+of\s+the\s+[\w,&’' \-]+?\s+committee\b")


def officer_capacity(capacity: str) -> bool:
    """An officer or a board chair; a director who chairs a board committee is not an executive."""
    return bool(OFFICER_TITLE.search(_COMMITTEE_CHAIR.sub(" ", capacity or "")))


_DATE_LINE = re.compile(r"(?i)^(?:dated?:?\s*)?(?:(?:as of|on)\s+)?(?:january|february|march|april|may|june|july|august|september|october|"
                        r"november|december)\s+\d{1,2},?\s+\d{4}$|^\d{1,2}/\d{1,2}/\d{2,4}$")


def is_person_name(s: str) -> bool:
    s = re.sub(r"\s+", " ", s).strip().strip(",;:")
    if not (4 <= len(s) <= 60) or any(ch.isdigit() for ch in s) or not _NAME.match(s):
        return False
    if _NOT_NAME.search(s):
        return False
    toks = [t for t in re.split(r"\s+", s) if t.strip(".").lower() not in _SUFFIXES]
    return 2 <= len(toks) <= 6 and sum(len(t.strip(".")) >= 2 for t in toks) >= 2


def exec_names(section: str) -> list[tuple[str, str]]:
    """(name, position) of the executive-officer table or list: a name line followed by an age line (table layout) or
    "Name, 64," / "Name (64)" / "Name 64 Position" on one line."""
    lines = []
    for ln in (x.strip() for x in section.split("\n")):
        if ln and _DIRECTORS_HEAD.match(ln):  # "Board of Directors": the officers' table has ended
            break
        if ln:
            lines.append(ln)
    out: list[tuple[str, str]] = []
    for i, ln in enumerate(lines):
        m = _INLINE.match(ln)
        if m and is_person_name(m.group(1)) and 25 <= int(m.group(2)) <= 99:
            out.append((m.group(1), ln[m.end():].strip(" ,.-—–")[:120]))
            continue
        m = _INLINE_TITLE.match(ln)  # "John P. Bayne, Jr. Senior Vice President & General Manager ..."
        if m and is_person_name(m.group(1)):
            out.append((m.group(1), ln[m.end(1):].strip(" ,.-—–")[:120]))
            continue
        if is_person_name(ln):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            nxt2 = lines[i + 2] if i + 2 < len(lines) else ""
            if _AGE.match(nxt) and 25 <= int(_AGE.match(nxt).group(1)) <= 99:  # table layout: name, age, position
                out.append((ln, nxt2[:120]))
            elif (_AGE.match(nxt2) and 25 <= int(_AGE.match(nxt2).group(1)) <= 99 and not is_person_name(nxt)
                  and OFFICER_TITLE.search(nxt)):  # name, position, age
                out.append((ln, nxt[:120]))
    return out


_STAR_SIGNER = re.compile(r"^([A-Z][A-Za-z.,'’\- ]{3,60}?)\s*\*$")


def _signer_entries(lines: list[str]) -> list[tuple[int, str, int]]:
    """(line index, name as signed, index of the first line after the name) of every "/s/ Name" signer."""
    out: list[tuple[int, str, int]] = []
    for i, ln in enumerate(lines):
        m = re.search(r"(?i)/s/\s*(.*)$", ln) or _STAR_SIGNER.match(ln)  # "JAMES D. FARLEY, JR.*": signed by attorney
        if not m or (not ln.lower().startswith("/s/") and "/s/" not in ln.lower() and not is_person_name(
                " ".join(w.title() if len(w) > 2 else w for w in m.group(1).split()))):
            continue
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,;*")
        first = i + 1
        if not name:  # "/s/" on its own line, the name on the next
            name = lines[i + 1] if i + 1 < len(lines) else ""
            first = i + 2
        out.append((i, name, first))
    return out


def signers(section: str) -> list[tuple[str, str]]:
    """(name, capacity) of every "/s/ Name" signer of the signature page."""
    lines = [ln.strip() for ln in section.split("\n") if ln.strip()]
    out: list[tuple[str, str]] = []
    for i, name, first in _signer_entries(lines):
        cap = []
        for x in lines[first:first + 8]:
            if "/s/" in x.lower() or _STAR_SIGNER.match(x):
                break
            if _DATE_LINE.match(x) or re.sub(r"[^a-z]", "", x.lower()) == re.sub(r"[^a-z]", "", name.lower()):
                continue
            if re.match(r"(?i)^(?:by:?|name:?|title:?|its:?|date:?|signature:?)$", x):
                continue
            if len(x) > 120 or re.match(r"(?i)^(?:pursuant|know all|in witness|signature\s+title)", x):
                break
            cap.append(re.sub(r"(?i)^title:\s*", "", x))
            if len(" ".join(cap)) > 200:
                break
        if name.isupper():
            name = " ".join(w if (len(w) <= 3 and w.endswith(".")) or w in ("II", "III", "IV") else w.title()
                            for w in name.split())
        out.append((name, " ".join(cap)[:200]))
    return out


def _title_name(name: str) -> str:
    """An all-capitals name as the news prints it, for the director rule ("C. RAYMOND LARKIN, JR." -> "C. Raymond
    Larkin, Jr."; "JOHN T. CASTEEN III," keeps its numeral); others as they are. (`signers` keeps its own v4 casing,
    so the executive rule's names are unchanged.)"""
    if not name.isupper():
        return name
    out = []
    for w in name.split():
        core = w.strip(",")
        if core in ("II", "III", "IV"):
            out.append(w)
        elif core in ("JR.", "SR.", "JR", "SR"):
            out.append(w.replace(core, core.title()))
        elif len(w) <= 3 and w.endswith("."):
            out.append(w)
        else:
            out.append(w.title())
    return " ".join(out)


# ------------------------------------------------------------------------------------------------ directors (freeze)
# 2026-09-28 (docs/impl/freeze_report.md, dev2_report W-4): the 10-K is signed by the principal officers and by at least a
# majority of the board (Form 10-K, General Instruction D(2)), so its signature page lists the firm's directors. A
# signer whose capacity names the board is a *director* person alias of the firm, point in time like an executive: from
# the usable time of the first 10-K that lists them, and a former director still identifies the firm ("Bill Gates",
# an MSFT director in the 2017-2019 10-Ks). Directors are masked as [executive] (the bank's only person mask).
DIRECTOR_CAPACITY = re.compile(r"(?i)\b(?:director|trustee|member of the board|board member)s?\b")
_ATTORNEY_ONLY = re.compile(r"(?i)^\W*(?:as\s+)?attorney[- ]in[- ]fact\W*$")
_NOT_A_PERSON = re.compile(r"(?i)\b(?:llp|llc|l\.l\.p\.|inc\.?|corporation|company|bank|trust company|& co)\b|&")


def _compact_name(s: str) -> str:
    """Letters of a name, casefolded, without honorifics and degree suffixes ("Robert J. Alpern, M.D." ->
    "robertjalpern")."""
    s = re.sub(r"(?i),?\s*\b(?:m\.?d|ph\.?d|j\.?d|cpa|esq)\b\.?", "", s)
    s = re.sub(r"(?i)^\s*(?:mr|ms|mrs|dr)\.?\s+", "", s)
    return re.sub(r"[^a-z]", "", s.casefold())


def signer_capacities(section: str) -> list[tuple[str, str]]:
    """(name, capacity) of every "/s/" signer. The capacity is `signers`' reading (the lines after the "/s/" line),
    except where that reading is empty or holds another signer's name: two-column layouts put two "/s/" lines first
    and then each printed name with its capacity ("/s/ A", "/s/ B", "A", "Director of X", "B", "Director of X"), where
    `signers` gives A nothing and joins both capacities into B's. There the capacity is read after the signer's own
    printed name, which may carry it on its line ("John T. Cahill, Director")."""
    lines = [ln.strip() for ln in section.split("\n") if ln.strip()]
    entries = _signer_entries(lines)
    fallback = signers(section)
    keys = [_compact_name(n) for _, n, _ in entries]
    sig_idx = {i for i, _, _ in entries}
    printed: dict[int, tuple[int, int]] = {}  # line -> (signer, characters of the line that are the name)
    for j, ln in enumerate(lines):
        if j in sig_idx or "/s/" in ln.lower():
            continue
        c = _compact_name(ln)
        best = None
        for e, k in enumerate(keys):
            if len(k) >= 6 and c.startswith(k) and (best is None or len(k) > len(keys[best])):
                best = e
        if best is not None:
            printed[j] = (best, len(keys[best]))
    out: list[tuple[str, str]] = []
    for e, (i, name, first) in enumerate(entries):
        name = _title_name(name)
        cap0 = fallback[e][1] if e < len(fallback) else ""
        c0 = _compact_name(cap0)
        mixed = any(len(k) >= 6 and k in c0 for f, k in enumerate(keys) if f != e and k != keys[e])
        own = [j for j, (b, _) in printed.items() if b == e and first <= j <= i + 12]
        if (cap0 and not mixed) or not own:
            out.append((name, cap0))
            continue
        j = min(own)
        ln = lines[j]
        n, taken = printed[j][1], 0
        cut = 0
        for cut, ch in enumerate(ln):  # the rest of the printed-name line after the name's letters
            if taken >= n:
                break
            taken += ch.isalpha()
        else:
            cut = len(ln)
        rest = re.sub(r"(?i)^\W*(?:(?:m\.?d|ph\.?d|jr|sr|ii|iii|iv)\b\.?\W*)*", "", ln[cut:]).strip(" ,;-–—")
        cap = [rest] if rest else []
        for k in range(j + 1, min(j + 6, len(lines))):
            x = lines[k]
            if k in printed or k in sig_idx or "/s/" in x.lower() or _DATE_LINE.match(x) or len(x) > 120:
                break
            if re.match(r"(?i)^(?:pursuant|know all|in witness|signature\s+title|by:?|date:?)", x):
                break
            cap.append(re.sub(r"(?i)^title:\s*", "", x))
        out.append((name, " ".join(cap)[:200]))
    return out


_STAR_ALONE = re.compile(r"^\*+$")
_DIR_HEAD = re.compile(r"(?i)\bdirectors?\b.{0,80}\b(?:listed below|named below|below executed)|"
                       r"\b(?:following|below) directors?\b")
# "Directors", "Directors:", "(iv) Directors:*", "Additional Directors:"
_DIR_WORD = re.compile(r"(?i)^\W*(?:\([ivx]+\)\s*)?(?:additional\s+|the\s+)?(?:directors\W*|director\s*:\W*)$")
_DIR_TAIL = re.compile(r"(?i)^\W*(?:constituting|being|comprising)\s+(?:at\s+least\s+)?a\s+majority\s+of\s+the\s+"
                       r"(?:board\s+of\s+)?directors\W*$")  # TGT: the names above this line
_NAME_COMMA_DIRECTOR = re.compile(r"(?i)^(.{4,60}?),\s*((?:lead\s+independent\s+|independent\s+|non-executive\s+)?"
                                  r"director(?:\s+of\s+[\w .,&'’]+)?)\W*$")  # "Warner L. Baxter, Director" (USB)
_ZW = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def _as_person(line: str) -> str | None:
    """A printed person name on its own line ("John J. Zillmer", "JOHN E. CALDWELL", "(Peter Bisson)",
    "* JOHN T. CASTEEN III,", "/s/ DOUGLAS F ESAMANN", "Richard H. Anderson*"), or None."""
    x = re.sub(r"(?i)^[\s*]*(?:/s/)?[\s*]*", "", line.strip()).strip("()").strip(" ,;*")
    t = _title_name(x)
    return t if t and not DIRECTOR_CAPACITY.search(t) and not OFFICER_TITLE.search(t) and is_person_name(t) else None


def _dir_lines(section: str) -> list[str]:
    return [y for y in (_ZW.sub("", ln).strip() for ln in section.split("\n")) if y]


_BY_LINE = re.compile(r"(?i)^\W*\*?\s*by\b")


def _attorneys(lines: list[str]) -> set[str]:
    """Compact names of the page's attorneys-in-fact: a signer named on a line that mentions an attorney ("appointing
    Mark A. L. Mason their attorney-in-fact", "Devinder Kumar, Attorney-in-Fact"), or whose "/s/" line is followed
    within three lines by one ("/s/ WILLIAM F. GIFFORD, JR.", "(WILLIAM F. GIFFORD, JR.", "ATTORNEY-IN-FACT)")."""
    out = set()
    att = [_compact_name(x) for x in lines if re.search(r"(?i)attorney", x)]
    for i, name, _ in _signer_entries(lines):
        k = _compact_name(name)
        if k and (any(k in x for x in att) or any(re.search(r"(?i)attorney", x) for x in lines[i + 1:i + 4])):
            out.add(k)
    return out


def attorney_signed_directors(section: str) -> list[tuple[str, str]]:
    """(name, capacity) of directors printed without a "/s/" capacity of their own, in the layouts of the store:
    - a lone "*" followed by the capacity, the date and the printed name in any order, up to the next "*" or "/s/"
      ("*", "Director", "January 29, 2021", "Nora M. Denzel": AMD, CSX);
    - a list of names under a heading ("The Directors of Citigroup listed below executed a power of attorney", "on
      behalf of the following Directors", "Directors", "Directors:", "Additional Directors:"; C, CEG, CVX, DUK, MDT, T),
      where a name may be signed ("/s/ DOUGLAS F ESAMANN") and then printed, up to the first line that is not a name;
    - names listed right above a line "Directors" (one per line ending in "," or "*", or ";"-separated on one line: MO,
      PNC);
    - a printed name whose next line is a board capacity ("Thomas K. Brown", "Director": MMM).
    Names that the page signs with an officer capacity are left to the executive rule."""
    lines = _dir_lines(section)
    officers = {_compact_name(n) for n, c in signers(section) if officer_capacity(c)}
    attorneys = _attorneys(lines)
    out: list[tuple[str, str]] = []

    def add(n: str | None, cap: str) -> None:
        if n and _compact_name(n) not in officers | attorneys:
            out.append((n, cap))

    for i, ln in enumerate(lines):
        if _STAR_ALONE.match(ln):
            names, cap = [], []
            for x in lines[i + 1:i + 7]:
                if _STAR_ALONE.match(x) or "/s/" in x.lower() or re.match(r"(?i)^\*?\s*by\b", x):
                    break
                if _DATE_LINE.match(x):
                    continue
                n = _as_person(x)
                (names if n else cap).append(n or x)
            if len(names) == 1:
                add(names[0], " ".join(cap)[:200])
            continue
        if (_DIR_HEAD.search(ln) and len(ln) <= 300) or _DIR_WORD.match(ln):
            got: list[str] = []
            for x in lines[i + 1:i + 60]:
                if _DATE_LINE.match(x):
                    continue
                n = _as_person(x)
                if n is None:
                    break
                if not got or _compact_name(n) != _compact_name(got[-1]):  # a signed name, then its printed form
                    got.append(n)
            for n in got:
                add(n, "Director (listed under: " + ln[:120] + ")")
        if _DIR_WORD.match(ln) or _DIR_TAIL.match(ln):
            k = i - 1  # the names right above: one per line ("MICHEL COMBES,", "*JOHN T. CASTEEN III,", "CALVIN
            while k >= 0 and i - k <= 40:  # DARDEN") or ";"-separated on one line (PNC)
                x = lines[k]
                parts = [y for y in re.split(r";|,(?!\s*(?:jr|sr|ii|iii|iv)\b\.?)", x, flags=re.IGNORECASE)
                         if y.strip(" *")]
                persons = [_as_person(y) for y in parts]
                if ";" in x and persons and all(persons):
                    for n in persons:
                        add(n, "Directors (listed above: " + ln[:60] + ")")
                elif _as_person(x):
                    add(_as_person(x), "Directors (listed above: " + ln[:60] + ")")
                else:
                    break
                k -= 1
            continue
        m = _NAME_COMMA_DIRECTOR.match(ln.lstrip("*/s ").strip())
        if m and "/s/" not in ln.lower() and _as_person(m.group(1)):
            add(_as_person(m.group(1)), m.group(2))
            continue
        if "/s/" not in ln.lower():
            n = _as_person(ln)
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if n and DIRECTOR_CAPACITY.search(nxt) and len(nxt) <= 80 and not officer_capacity(nxt) and \
                    _as_person(nxt) is None and not _DIR_WORD.match(nxt):
                add(n, nxt)
    return out


def director_signers(section: str) -> list[tuple[str, str]]:
    """(name, capacity) of the 10-K's directors who are not executives. Form 10-K is signed by the principal officers
    and by at least a majority of the board (General Instruction D(2)), so a natural-person signer is an officer, a
    director or an attorney-in-fact. Kept: a "/s/" signer whose own capacity names the board ("Director", "Lead
    Independent Director", "Director of AbbVie Inc.", "Trustee") or is empty (multi-column pages that print no
    capacity per name: XOM), without an officer title (officers are the executive rule's) and not only an
    attorney-in-fact; and the directors printed without a "/s/" capacity of their own (`attorney_signed_directors`).
    A firm (an auditor's "/s/ Ernst & Young LLP") is not a person. One entry per name."""
    out, seen = [], set()
    # a signer without a printed capacity is a director unless the page names them as an officer elsewhere or as an
    # attorney-in-fact ("... appointing Mark A. L. Mason their attorney-in-fact"; MDT's "*Bradley E. Lerman, by
    # signing his name hereto, ... pursuant to powers of attorney")
    lines = [ln.strip() for ln in section.split("\n") if ln.strip()]
    officers = {_compact_name(n) for n, c in signers(section) if officer_capacity(c)}
    attorneys = _attorneys(lines)
    signed = []
    for (i, _, _), (n, c) in zip(_signer_entries(lines), signer_capacities(section)):
        if not c:
            if _compact_name(n) in officers | attorneys or _BY_LINE.match(lines[i]) or (i and _BY_LINE.match(lines[i - 1])):
                continue  # an officer, an attorney-in-fact, or the registrant's "By:" signer
            c = "(signer; no capacity printed)"
        signed.append((n, c))
    for name, cap in [*signed, *attorney_signed_directors(section)]:
        if _ATTORNEY_ONLY.match(cap) or officer_capacity(cap) or re.search(r"(?i)attorney[- ]in[- ]fact", cap) and \
                not DIRECTOR_CAPACITY.search(re.sub(r"(?i)attorney[- ]in[- ]fact\W*directors?", "", cap)):
            continue
        if not (DIRECTOR_CAPACITY.search(cap) or cap.startswith("(signer;")):
            continue
        if _NOT_A_PERSON.search(name) or _compact_name(name) in seen:
            continue
        seen.add(_compact_name(name))
        out.append((name, cap))
    return out


def parse_name(raw: str) -> dict | None:
    """first, nick, middle, last, suffix of a person's name as a 10-K prints it."""
    s = re.sub(r"\s+", " ", raw.replace("’", "'")).strip(" ,;*")
    s = re.split(r"\.\s+(?:Mr|Ms|Mrs|Dr)\.", s)[0]  # "William D. Mosley. Mr. Mosley has served ..."
    nick = None
    m = re.search(r"[\"“(]([A-Z][a-z]+)[\"”)]", s)
    if m:
        nick = m.group(1)
        s = (s[:m.start()] + s[m.end():]).strip()
        s = re.sub(r"\s+", " ", s)
    toks = s.replace(",", " ").split()
    toks = [t for t in toks if t.lower() not in _HONORIFICS]
    suffix = [t for t in toks if t.lower().strip(".") in {x.strip(".") for x in _SUFFIXES}]
    toks = [t for t in toks if t not in suffix]
    if len(toks) < 2:
        return None
    if toks[0].isupper() and len(toks[0]) > 2:
        toks = [t.title() if t.isupper() and len(t) > 2 else t for t in toks]
    k = len(toks) - 1
    while k - 1 >= 1 and toks[k - 1].lower() in _PARTICLES:
        k -= 1
    last = " ".join(toks[k:])
    first = toks[0]
    middle = toks[1:k]
    if re.fullmatch(r"[A-Z]\.", first) and middle and not re.fullmatch(r"[A-Z]\.", middle[0]):
        first, middle = middle[0], middle[1:]  # "J. Michael Smith" goes by Michael
    if len(first.strip(".")) < 2 or len(last.strip(".")) < 2:
        return None
    return {"first": first, "nick": nick, "middle": middle, "last": last, "suffix": " ".join(suffix)}


NICKNAMES: dict[str, tuple[str, ...]] = {
    "abraham": ("abe",), "albert": ("al",), "alexander": ("alex",), "alexandra": ("alex",), "alfred": ("al",),
    "andrew": ("andy", "drew"), "anthony": ("tony",), "arthur": ("art",), "benjamin": ("ben",), "bradley": ("brad",),
    "catherine": ("cathy", "kate", "cate"), "charles": ("charlie", "chuck"), "christine": ("chris",),
    "christopher": ("chris",), "clifford": ("cliff",), "daniel": ("dan", "danny"), "david": ("dave",),
    "deborah": ("deb", "debbie"), "donald": ("don",), "douglas": ("doug",), "edward": ("ed", "ted", "eddie"),
    "elizabeth": ("liz", "beth", "betsy"), "eugene": ("gene",), "frederick": ("fred",), "gerald": ("jerry",),
    "gregory": ("greg",), "harold": ("hal",), "henry": ("hank",), "jacqueline": ("jackie",), "james": ("jim", "jamie"),
    "jeffrey": ("jeff",), "jennifer": ("jen", "jenny"), "john": ("jack",), "jonathan": ("jon",),
    "joseph": ("joe",), "joshua": ("josh",), "judith": ("judy",), "katherine": ("kathy", "kate", "katie"),
    "kathleen": ("kathy", "kate"), "kenneth": ("ken",), "kimberly": ("kim",), "lawrence": ("larry",),
    "leonard": ("len",), "margaret": ("maggie", "meg", "peggy"), "matthew": ("matt",), "michael": ("mike",),
    "mitchell": ("mitch",), "nathaniel": ("nate",), "nicholas": ("nick",), "patricia": ("pat", "patty", "trish"),
    "patrick": ("pat",), "peter": ("pete",), "philip": ("phil",), "phillip": ("phil",), "raymond": ("ray",),
    "rebecca": ("becky",), "richard": ("rick", "rich", "dick"), "robert": ("bob", "rob", "bobby"),
    "ronald": ("ron",), "samuel": ("sam",), "samantha": ("sam",), "stephen": ("steve",), "steven": ("steve",),
    "susan": ("sue",), "theodore": ("ted",), "thomas": ("tom",), "timothy": ("tim",), "victoria": ("vicki",),
    "vladimir": ("vlad",),  # freeze: HOOD's 10-Ks print "Vladimir Tenev", the news "Vlad Tenev" (56 items before 2024-07)
    "william": ("bill", "will"), "zachary": ("zach",),
}


def forms(p: dict) -> list[str]:
    """Full-name forms the news uses for a person: first (or nickname) [+ middle initials] + last [+ suffix]."""
    firsts = {p["first"].strip(".")}
    if p.get("nick"):
        firsts.add(p["nick"])
    for f in list(firsts):
        firsts |= {n.capitalize() for n in NICKNAMES.get(f.lower(), ())}
    mids = [m for m in p["middle"] if m]
    variants = {""}
    if mids:
        variants.add(" " + " ".join(mids))
        variants.add(" " + " ".join((m if m.endswith(".") or len(m) == 1 else m[0] + ".") for m in mids))
    out = set()
    for f in firsts:
        for v in variants:
            base = f"{f}{v} {p['last']}"
            out.add(re.sub(r"\s+", " ", base))
            if p.get("suffix"):
                out.add(re.sub(r"\s+", " ", f"{base} {p['suffix']}"))
    return sorted(out, key=lambda x: (-len(x), x))


# words of headings and business units that an officers' table can place where a name goes ("Asia Pacific", "Wealth
# Management", "Critical Accounting Policies"): such an entry is not a person
_UNIT_WORDS = frozenset("""
management wealth pacific asia america americas europe emea apac latin global international operations operating
north south east west central region regional services solutions products markets market group division segment
business corporate consumer commercial retail digital technology technologies strategy finance financial capital
investment investments banking insurance health care legal human resources communications marketing sales supply
chain manufacturing engineering research development affairs relations public government policy policies risk
compliance audit treasury tax accounting controller officer president vice chief executive senior general counsel
secretary head lead director company worldwide enterprise enterprises industrial industries energy power critical
estimates updates liquidity supplemental information
""".split())


def plausible_person(p: dict) -> bool:
    """Not a heading or business unit: the first name is a given name or not a unit/common/place word, and at most one
    token is such a word ("Kevin Johnson", "Robert E. Jordan" pass; "Wealth Management", "Asia Pacific" do not)."""
    from jevtrader.data.blinding import COMMON_WORDS, GIVEN_NAMES, STOP_TOKENS

    toks = [t.strip(".,").casefold() for t in [p["first"], *p["middle"], *p["last"].split()] if t.strip(".,")]
    bad = [t for t in toks if t in _UNIT_WORDS or t in COMMON_WORDS or t in STOP_TOKENS]
    first = toks[0] if toks else ""
    given = GIVEN_NAMES | set(NICKNAMES) | {n for v in NICKNAMES.values() for n in v}
    if len(bad) > 1 or (first in bad and first not in given):
        return False
    # every word an English dictionary word and none a given name: a heading ("Medical Devices", "Regulatory
    # Requirements"), not a person
    words = [t for t in toks if len(t) > 1]
    return not (words and all(t in _lm_words() for t in words) and not any(t in given for t in words))


def plausible_signer(p: dict) -> bool:
    """`plausible_person` for a name signed on the signature page, where the heading filter has less to do: single-letter
    middle initials are not read as words ("Paula A. Price" is not the words "a" and "price")."""
    return plausible_person({**p, "middle": [m for m in p["middle"] if len(m.strip(".")) > 1]})


@functools.lru_cache(maxsize=1)
def _lm_words() -> frozenset[str]:
    import pandas as pd

    p = ROOT / "data/raw/lm_dictionary/Loughran-McDonald_MasterDictionary.csv"
    if not p.exists():
        return frozenset()
    return frozenset(pd.read_csv(p, usecols=["Word"], keep_default_na=False)["Word"].str.casefold())


@functools.lru_cache(maxsize=1)
def _lm_doc_counts() -> dict[str, int]:
    import pandas as pd

    p = ROOT / "data/raw/lm_dictionary/Loughran-McDonald_MasterDictionary.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p, usecols=["Word", "Doc Count"], keep_default_na=False)
    return dict(zip(d["Word"].str.casefold(), pd.to_numeric(d["Doc Count"], errors="coerce").fillna(0).astype(int)))


def surname_word(surname: str) -> bool:
    """A surname that is a common English word (COMMON_WORD_DOCS): "Cook", "Smith", "Call"; not "Musk", "Pinto"."""
    return _lm_doc_counts().get(surname.strip().casefold(), 0) >= COMMON_WORD_DOCS


def person_key(p: dict) -> str:
    return (p["first"].strip(".") + " " + p["last"]).casefold()


# ------------------------------------------------------------------------------------------------ build
def _sessions():
    import pandas as pd

    p = ROOT / "data/raw/bars"
    files = sorted(p.glob("*raw*.parquet")) or sorted(p.glob("*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["symbol", "date"], filters=[("symbol", "==", "SPY")]) for f in files])
    return pd.DatetimeIndex(sorted(pd.to_datetime(d["date"]).dt.normalize().unique()))


def build() -> None:
    import pandas as pd

    from jevtrader.data.blinding import surname_eligible
    from jevtrader.programs.context import next_session_open_utc

    sessions = _sessions()
    listings = []
    for p in sorted(TENK.glob("*/*.json")):
        d = json.loads(p.read_text())
        sym = d["symbol"]
        usable = next_session_open_utc(sessions, d["filed"])
        if usable is None:
            continue
        seen = set()
        for src, rows in (("exec_section", exec_names(d.get("exec_section") or "")),
                          ("signatures", [(n, c) for n, c in signers(d.get("signatures") or "")
                                          if officer_capacity(c)]),
                          ("signatures_director", director_signers(d.get("signatures") or ""))):
            for raw, title in rows:
                pn = parse_name(raw)
                if pn is None or not (plausible_signer(pn) if src == "signatures_director" else plausible_person(pn)):
                    continue
                k = person_key(pn)
                if (k, src) in seen:
                    continue
                seen.add((k, src))
                listings.append({"symbol": sym, "person": k, "name": re.sub(r"\s+", " ", raw).strip(), "source": src,
                                 "title": title, "filed": d["filed"], "accession": d["accession"],
                                 "usable_from": usable.isoformat(), "url": d["url"], **{f"p_{x}": pn[x] for x in pn}})
    for src, rows in (("manual", _manual()), ("manual_founder", _manual(FOUNDERS_FILE))):
        for m in rows:
            pn = parse_name(m["name"])
            listings.append({"symbol": m["symbol"], "person": person_key(pn), "name": m["name"], "source": src,
                             "title": m["role"], "filed": m["valid_from"][:10], "accession": m.get("accession", ""),
                             "usable_from": pd.Timestamp(m["valid_from"], tz="UTC").isoformat(), "url": m["source"],
                             **{f"p_{x}": pn[x] for x in pn}})
    L = pd.DataFrame(listings)
    L["p_middle"] = L["p_middle"].map(lambda x: " ".join(x) if isinstance(x, list) else (x or ""))
    L = L.sort_values(["symbol", "person", "usable_from", "source"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_csv(OUT / "executive_listings.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"{len(L)} listings, {L.groupby(['symbol', 'person']).ngroups} (firm, person) pairs, "
          f"{L.symbol.nunique()} firms", flush=True)

    # one row per (firm, person): forms, the first usable time, and the surname statistics per listing
    people = {}
    for (sym, k), g in L.groupby(["symbol", "person"], sort=True):
        g = g.sort_values("usable_from")
        first_use: dict[str, str] = {}  # form -> usable time of the first listing that yields it (v4)
        for r in g.itertuples(index=False):
            pn = {"first": r.p_first, "nick": r.p_nick if isinstance(r.p_nick, str) and r.p_nick else None,
                  "middle": [m for m in str(r.p_middle or "").split() if m], "last": r.p_last,
                  "suffix": r.p_suffix if isinstance(r.p_suffix, str) else ""}
            for f in forms(pn):
                first_use.setdefault(f, r.usable_from)
        fs = sorted(first_use)
        last = g.iloc[0].p_last
        people[(sym, k)] = {"symbol": sym, "person": k, "name": g.iloc[-1]["name"], "last": last, "forms": fs,
                            "form_starts": [first_use[f] for f in fs],
                            "firsts": sorted({f.split()[0] for f in fs}), "starts": sorted(set(g.usable_from)),
                            "start": g.usable_from.min(), "sources": sorted(set(g.source)),
                            "titles": "; ".join(dict.fromkeys(t for t in g.title if isinstance(t, str) and t))[:200],
                            "roles": "+".join(sorted({ROLE_OF_SOURCE[x] for x in g.source})),
                            "first_accession": g.iloc[0].accession, "first_url": g.iloc[0].url}
    excluded = director_attribution(people)
    if excluded:
        X = pd.DataFrame([{"symbol": people[k]["symbol"], "person": people[k]["person"], "name": people[k]["name"],
                           "forms": "|".join(people[k]["forms"]), **x} for k, x in sorted(excluded.items())])
        X.to_csv(OUT / DIRECTOR_EXCLUSIONS_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
        people = {k: v for k, v in people.items() if k not in excluded}
        print(f"{len(X)} director rows not masked (named mostly elsewhere; {DIRECTOR_EXCLUSIONS_FILE}): "
              f"{', '.join(X.name)}", flush=True)
    stats = surname_stats(people)
    rows = []
    for key, v in people.items():
        s = stats.get(key, {})
        rows.append({"symbol": v["symbol"], "person": v["person"], "name": v["name"], "surname": v["last"],
                     "forms": "|".join(v["forms"]), "form_starts": "|".join(v["form_starts"]), "start": v["start"],
                     "surname_eligible": surname_eligible(v["last"]), "surname_word": surname_word(v["last"]),
                     "surname_from": s.get("surname_from", ""), "surname_items": s.get("items", 0),
                     "surname_share": s.get("share", ""), "surname_other": s.get("other", ""),
                     "sources": "+".join(v["sources"]), "roles": v["roles"], "titles": v["titles"],
                     "first_accession": v["first_accession"],
                     "first_source_url": v["first_url"]})
    A = pd.DataFrame(rows).sort_values(["symbol", "person"])
    A.to_csv(OUT / "executive_aliases.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"{len(A)} (firm, person) rows ({', '.join(f'{k} {v}' for k, v in A.roles.value_counts().items())}); "
          f"surname alone everywhere for {int((A.surname_from != '').sum())}", flush=True)


def _manual(name: str = "manual_executives.csv") -> list[dict]:
    """Rows of a hand list (manual_executives.csv; manual_founders.csv): symbol, name, role, valid_from, source, ..."""
    p = OUT / name
    if not p.exists():
        return []
    with open(p, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("symbol") and not r["symbol"].startswith("#")]
    for r in rows:
        if not r.get("source", "").startswith("https://www.sec.gov/") or not r.get("valid_from"):
            raise ValueError(f"{p}: {r.get('symbol')} {r.get('name')}: every row needs valid_from and an SEC source")
    return rows


def _archive_before(end: str):
    """News items (id-level) created and last edited before `end`: known time, tags, text (headline + summary)."""
    import numpy as np
    import pandas as pd

    news = pd.read_parquet(ROOT / "data/processed/news_table.parquet",
                           columns=["id", "symbol", "created_at", "updated_at", "headline", "summary"])
    news["created_at"] = pd.to_datetime(news["created_at"], utc=True)
    news["updated_at"] = pd.to_datetime(news["updated_at"], utc=True).fillna(news["created_at"])
    news = news[(news.created_at < pd.Timestamp(end, tz="UTC")) & (news.updated_at < pd.Timestamp(end, tz="UTC"))]
    g = news.groupby("id").agg(known=("updated_at", "max"), created=("created_at", "min"), tags=("symbol", frozenset),
                               headline=("headline", "first"), summary=("summary", "first"))
    g["known"] = np.maximum(g["known"].values, g["created"].values)
    return g


def director_attribution(people: dict) -> dict:
    """(freeze) The director-only rows that are *not* masked: a director whose full name appears in at least
    DIRECTOR_MIN_ITEMS archive items created and last edited before CORPUS_END, of which fewer than
    DIRECTOR_MIN_SHARE are tagged to the firm, is named mostly in another role or is someone else with the same name
    (another firm's CEO, a public official, a TV trader, an actor, "Jeffrey Epstein" for BKNG's Jeffery H. Epstein).
    Masking such a name hides nothing about the firm and removes what the text says, so the row is dropped (and
    listed with its counts in DIRECTOR_EXCLUSIONS_FILE). Founders (hand-verified) and executives are never dropped.
    The statistic reads only pre-W_clean text, so for W_clean it is point in time; for earlier weeks dropping a row
    only returns those names to the pre-freeze (DEV #2) masking."""
    import re as _re

    rows = {k: v for k, v in people.items() if v["roles"] == "director"}
    if not rows:
        return {}
    g = _archive_before(CORPUS_END)
    texts = (g["headline"].fillna("") + " \n " + g["summary"].fillna("")).tolist()
    tags = g["tags"].tolist()
    ids = g.index.tolist()
    by_form: dict[str, list] = defaultdict(list)
    for k, v in rows.items():
        for f in v["forms"]:
            if " " in f and len(f) >= 5:
                by_form[f].append(k)
    rx = _re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(map(_re.escape, by_form), key=len, reverse=True)) +
                     r")(?![A-Za-z])")
    items: dict[tuple, list[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        for key in {k for m in rx.finditer(t) for k in by_form[m.group(1)]}:
            items[key].append(i)
    out = {}
    for key, idx in items.items():
        sym = rows[key]["symbol"]
        tagged = sum(sym in tags[i] for i in idx)
        if len(idx) >= DIRECTOR_MIN_ITEMS and tagged / len(idx) < DIRECTOR_MIN_SHARE:
            out[key] = {"items": len(idx), "tagged": tagged, "tagged_share": round(tagged / len(idx), 3),
                        "example_item_ids": " ".join(str(ids[i]) for i in idx[:3]),
                        "rule": f">= {DIRECTOR_MIN_ITEMS} items before {CORPUS_END}, < {DIRECTOR_MIN_SHARE} tagged"}
    return out


def surname_stats(people: dict) -> dict:
    """Per (firm, person): the first listing time at which the surname is distinctive (module doc), with the counts at
    that time (or at the last listing if never)."""
    import numpy as np
    import pandas as pd

    from jevtrader.data.blinding import GIVEN_NAMES, surname_eligible

    news = pd.read_parquet(ROOT / "data/processed/news_table.parquet",
                           columns=["id", "symbol", "created_at", "updated_at", "headline", "summary"])
    news["created_at"] = pd.to_datetime(news["created_at"], utc=True)
    news["updated_at"] = pd.to_datetime(news["updated_at"], utc=True).fillna(news["created_at"])
    news = news[(news.created_at < pd.Timestamp(CORPUS_END, tz="UTC")) & (news.updated_at < pd.Timestamp(CORPUS_END,
                                                                                                       tz="UTC"))]
    g = news.groupby("id").agg(known=("updated_at", "max"), created=("created_at", "min"), tags=("symbol", frozenset),
                               headline=("headline", "first"), summary=("summary", "first"))
    g["known"] = np.maximum(g["known"].values, g["created"].values)
    texts = (g["headline"].fillna("") + " \n " + g["summary"].fillna("")).tolist()
    known = g["known"].tolist()
    tags = g["tags"].tolist()
    by_last: dict[str, list] = defaultdict(list)
    for key, v in people.items():
        if surname_eligible(v["last"]):
            by_last[v["last"]].append(key)
    if not by_last:
        return {}
    rx = re.compile(r"(?<![A-Za-z0-9'’\-])(" + "|".join(sorted(map(re.escape, by_last), key=len, reverse=True)) +
                    r")(?![A-Za-z0-9\-])")
    occ: dict[str, list[int]] = defaultdict(list)  # surname -> item indices
    for i, t in enumerate(texts):
        for s in {m.group(1) for m in rx.finditer(t)}:
            occ[s].append(i)
    given = {x.casefold() for x in GIVEN_NAMES}
    out = {}
    for last, keys in by_last.items():
        idx = occ.get(last, [])
        for key in keys:
            v = people[key]
            full = re.compile(r"(?<![A-Za-z])(?:" + "|".join(map(re.escape, v["forms"])) + r")(?![A-Za-z])")
            prev = re.compile(r"([A-Z][a-z]+)\s+(?:[A-Z]\.\s+)?" + re.escape(last) + r"(?![A-Za-z])")
            firsts = {f.casefold() for f in v["firsts"]}
            rec = []  # (known_at, attributable, occurrences after another given name, occurrences)
            for i in idx:
                t = texts[i]
                attributable = bool(full.search(t)) or v["symbol"] in tags[i]
                n_occ = len(re.findall(r"(?<![A-Za-z])" + re.escape(last) + r"(?![A-Za-z])", t))
                other = sum(1 for m in prev.finditer(t) if m.group(1).casefold() in given and
                            m.group(1).casefold() not in firsts)
                rec.append((known[i], attributable, other, n_occ))
            rec.sort(key=lambda x: x[0])
            ts = [x[0] for x in rec]
            res = {"items": 0, "share": "", "other": ""}
            for start in v["starts"]:
                s = pd.Timestamp(start)
                k = int(np.searchsorted(np.array(ts, dtype="datetime64[ns]"), np.datetime64(s.tz_convert(None)),
                                        side="left")) if ts else 0
                sub = rec[:k]
                n = len(sub)
                share = sum(a for _, a, _, _ in sub) / n if n else 0.0
                oth = sum(o for _, _, o, _ in sub) / max(1, sum(c for *_, c in sub))
                res = {"items": n, "share": round(share, 3), "other": round(oth, 3)}
                if n >= MIN_ITEMS and share >= MIN_SHARE and oth <= MAX_OTHER and not surname_word(last):
                    res["surname_from"] = start
                    break
            out[key] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["fetch", "build"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--redo-signature-table", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.workers, set(a.symbols.split(",")) if a.symbols else None, a.redo_signature_table)
    else:
        build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
