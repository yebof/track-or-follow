"""Entity, level and time blinding for System-One states (look-ahead control; design_v1 §5.0, validity M9/m7).

A model whose weights saw 2020-2026 market history can recognize "Apple, March 2023" and recall what
happened next. Blinding replaces company names and tickers with stable pseudonyms, masks index levels and
share prices, and turns absolute dates into offsets relative to the decision, so a judgment has to come from
the evidence in the state. It is necessarily imperfect (products, executives and rare events can still
identify a firm); the residual identifiability is measured by the D2 probe, not assumed (§13.7).

v1 extensions:
- pseudonyms have 8 hex characters of sha256(salt + symbol) and are asserted unique over the symbol set;
- *every* universe firm is masked in every text, not only the focal firm (co-mentions leak identity);
- aliases per firm: SEC current and former names, every historical ticker (`SymbolResolver.history`, e.g.
  FB for META), a manual brand list (`BRAND_ALIASES`, e.g. Google for GOOGL, and the feed's own short names such as
  "Applied Mat" or "Capital One Finl", found by `data.leak_probe`), and regex-discovered aliases (`AliasDiscovery`,
  rule `pit-alias-v3`: firm-specific product, brand and subsidiary names only, computed on expanding windows from
  items created and last updated before the anchor; docs/impl/day3_blind.md);
- stock-index levels and share prices are masked as `[level]`; absolute dates as `[date]`;
- web host names ("investor.apple.com", "Amazon.Com", "about.fb.com") are masked with the code name of the firm a
  label names (text-rules v2, docs/impl/day3_blind.md "Audit response").

Matching rules (all-universe masking makes false positives costly: "Raises Price Target" must not become two
firms' code names):
- names match case-insensitively but must start with a capital letter or digit in the text ("Apple", not
  "apple pie"), on alphanumeric boundaries, never right after "$" or inside a decimal ("$3M", "1.3M"). Two kinds of
  names match in any case: names registered with a lower-case first letter ("lululemon", "eBay", "onsemi") and
  multi-word names with at least two distinctive words ("lululemon athletica", "estee lauder companies");
- matching ignores diacritics ("Estée", "Mondelēz") and a comma or period before a space ("Becton, Dickinson",
  "U.S. Bancorp" against "U.S Bancorp"), on a length-preserving folded copy of the text, so the output keeps the
  original characters everywhere else;
- a name right after "@" (an X handle: "@nvidia") matches in any case;
- a legal or feed designator right after a masked name is masked with it ("UnitedHealth Group", "Royal Caribbean
  Gr", "Merck & Co", "Honeywell Intl", "Amazon.com"): such residues ("Company-X de Nemours", "Company-X Lauder
  Cos") re-identify a firm;
- **word-like names** (a single token that is an ordinary word, place or person name: Target, Visa, Delta,
  Match, Boston) are masked only in explicit contexts ("Target Corp", "Target (TGT)", "Visa (NYSE:V)") or in a
  text tagged to that firm (`tagged=`: an item tagged TGT masks "Target", except inside protected phrases such
  as "Price Target"). A single-token *former* name (Citigroup's TRAVELERS, Costco's PRICE, Match's IAC) is
  used only while it was the firm's name (SEC `formerNames` dates), because other firms use it today;
- a web host name (labels ending in a common top-level domain, "investors.target.com") is a firm's explicit context:
  it is masked, whole, with the code name of the firm that one of its labels names exactly (hyphens and case
  ignored: "t-mobile", "paloaltonetworks", "exxonmobil"; word-like names count here, "investors.target.com"; tickers
  only in the registrable label, "about.fb.com"), or, for a firm the text is tagged to, whose distinctive name is a
  prefix or suffix of a label ("fedexcares.com", "shopdisney.com"). Other host names stay (D0 strips URLs with a
  scheme or "www.");
- tickers of 2+ characters that are not words or common abbreviations match as whole uppercase tokens
  (`$AAPL`, `AAPL`, `GE`); one-letter tickers and word-like tickers (`A`, `F`, `ALL`, `NOW`, `IT`, `PM`) match
  only in explicit ticker contexts: `$F`, `(F)`, `NYSE: F`;
- an alias claimed by several firms (a renamed or merged firm whose predecessor and successor are both in
  the study set, e.g. DISCA/WBD, RTN/RTX) is never dropped: it is resolved point-in-time to one firm, by
  (tagged to the text, trading at the cutoff, claim valid at the cutoff, most recent claim, symbol). When in
  doubt it is masked with *some* claimant's code name, which still blinds.

Point-in-time: the masking at cutoff c depends only on the alias tables, dates <= c (name validity, trading
life, discovery anchor <= c), items created and updated before the anchor and, when discovery is given a membership
function, the firms that are members at the anchor. Compiles for different anchors are pure functions and share no
mutable state, so the order in which anchors are compiled never matters.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import html
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger("jevtrader.blinding")

SALT = "jev-www27"
# version of the text rules (names, hosts, levels, times) with discovery off; it enters every context description
# (`programs.context.RealData.describe`), so shards built under other rules are never resumed. v2 (audit repair,
# docs/impl/day3_blind.md "Audit response"): host names, options-alert and "$X Price Target" levels, half-year, CY,
# earnings-date and apostrophe-year times, NEE's FPL and ELV's Wellpoint
# v3 (docs/impl/dev2_blind.md): years 1800-2099 outside the exact-offset range (coarse), statute years, the curated
# alias list data/raw/aliases/manual_aliases.csv ("Visa", "3M", ... hard where verified), quantities vs "3M", and
# executives from the firms' 10-Ks masked as [executive]
# v4 (docs/impl/dev2_blind.md, audit response): a year next to another year is a year ("2023 2022% Change": v3 crashed
# F-A on DE's 10-K) and the year pass repeats until nothing changes; pre-/post-YYYY, EU statute numbers and standard
# editions; each executive name form from its own first 10-K; the surname alone never where it is someone or
# something else (another person's given name or initials, a public official, a place or organisation name), a
# common-word surname only next to a title or a verb of speech and never everywhere; dated curated aliases (E*TRADE),
# MS&Co.; a name right after "<number>." with no space
TEXT_RULES = "blind-text-v4"
LEVEL_MASK = "[level]"
DATE_MASK = "[date]"
EXEC_MASK = "[executive]"  # question bank conventions.masks.executive

_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|lp|llc|the|/[a-z]+/?|class [a-c]|de)\b\.?",
    re.I,
)

# Capitalized words too common to identify a firm on their own (first-word aliases, alias discovery).
COMMON_WORDS = frozenset(w.casefold() for w in """
a an the and or of for to in on at by with from as is are was were be been its it this that these those new
not no all any more most less other our your their his her we you they he she after before over under up down
out off into than then about amid amidst against between during through while why how what when where which who
will would can could may might should must says said say reports reported report update updates week weeks
day days month months year years quarter quarterly annual first second third fourth last next top best big
january february march april may june july august september october november december monday tuesday wednesday
thursday friday saturday sunday today yesterday tomorrow q1 q2 q3 q4 fy h1 h2
american america united states national international global general capital digital public southern
northern eastern western central pacific atlantic texas california new york air health healthcare energy
financial finance group holdings technologies technology systems services solutions resources industries
industrial motors motor airlines communications pharmaceuticals pharma realty trust street state royal bank
banks banking home chemical chemicals southwestern union continental consolidated universal standard advanced
applied analog automatic marathon charles eli walt johnson intercontinental insurance life mutual
properties property partners management investment investments securities exchange markets market stock
stocks shares share price prices target targets analyst analysts rating ratings upgrade upgrades downgrade
downgrades earnings revenue revenues sales profit profits guidance outlook forecast estimates estimate
consensus dividend dividends buyback deal deals merger acquisition acquisitions offering ceo cfo coo chairman
president chief officer executive board director directors inc corp co company companies ltd plc llc lp
nyse nasdaq amex sec fda ftc doj fed federal reserve treasury congress senate house white government court
us u.s. usa uk eu china chinese europe european japan japanese india canada mexico germany france russia
ukraine israel wall benzinga pro premium options option activity unusual whales movers mover trading trade
trades traders premarket after-hours hours session index indices etf etfs fund funds futures bitcoin crypto
oil gas gold data ai cloud software chip chips semiconductor semiconductors retail consumer consumers
electric power water food beverage beverages products product brands brand stores store network networks
media entertainment games game sports mobile wireless internet online platform platforms devices device
vehicle vehicles cars car truck trucks bullish bearish buy sell hold neutral outperform underperform
overweight underweight strong weak higher lower rises rise falls fall gains gain jumps jump drops drop
dollar dollars super morgan
""".split())

# Single-token firm names that are also ordinary words, places, person names or other firms' names in
# financial news. Curated once from the study set's SEC names: lowercase dictionary words whose lowercase use in
# pre-2023 Benzinga text is >= 1% of their capitalized use, plus places, person names and names other firms
# use. Such a name is masked only in explicit contexts or in text tagged to its firm (module docstring).
WORD_NAMES = frozenset("""
target price travelers booking electronic match progressive discovery delta align pioneer
constellation cadence microchip wells visa charter intuitive coherent workday vertex deer bristol penn
boston southwest norfolk devon norwegian
morgan philip simon edwards lam willis warner duke fidelity
3m freeport bd
""".split())

# Phrases whose words are never company names here ("Raises Price Target", "Chevron deference").
PROTECTED_PHRASES = ("price targets?", r"target prices?(?! targets?\b)", "chevron doctrine", "chevron deference", "delta variant",
                     "wells notice", "super bowl", "constellation brands", "fidelity investments", "warner music",
                     # text-rules v3: senses of names the manual list makes hard ("Visa", "Freeport", "Oxy",
                     # "Microchip", "Workday") or of a hard first word ("Palo"), found in the news archive before 2024-07
                     r"(?:h-?1b|h-?2b|l-?1|j-?1|f-?1|o-?1|eb-?5|student|work|travel|tourist|entry|exit|golden|transit) visas?",
                     (r"visas? (?:bans?|issues?|programs?|waivers?|holders?|applications?|requirements?|restrictions?|"
                      r"rules?|polic(?:y|ies)|fees?|processing|free|overstays?|sponsorship|lottery|status|interviews?|"
                      r"appointments?|denials?|curbs?|limits?|caps?|suspensions?)"),
                     r"(?:issues?|issued|grants?|granted|deny|denies|denied|revokes?|revoked) (?:a visa|visas)\b",
                     "visa-free",  # not "issue VISA®-branded cards" (FIS's 10-K)
                     r"palo alto(?! networks)", "freeport lng", r"freeport,? (?:texas|tx|bahamas|maine|illinois)",
                     "inspira technologies oxy",
                     (r"microchips? (?:shortages?|supply|supplies|crisis|industry|makers?|manufacturers?|production|"
                      r"demand|sector)"), r"\d+-hour workday",
                     # other entities named like a study firm, seen once F-A text masks the whole study set
                     "eaton towers", "monolithic memories")
_PROTECTED = "|".join(p.replace(" ", r"\s+") for p in PROTECTED_PHRASES)

# Tickers that are English words or common abbreviations: only masked in explicit ticker contexts.
COMMON_WORD_TICKERS = frozenset("""
ALL NOW LOW KEY HAS CAT ON IT MO PM DE MA HD MS AI DD ED EL HP IP IR SO RE NI PR UA OI US UK EU CEO CFO ABC ARE BIG
CARE CASH COST DAY DOW EAT FAST GOOD HAL HUM KEYS LIFE LOVE MAN MAT NEW NICE ONE OUT PAY PEAK REAL SEE SITE TECH
TRUE WELL WAT YOU FDA SEC GDP CPI EPS ETF IPO USA
""".split())

# Brand names that differ from the SEC name (design_v1 §5.0 "manual list"). Only symbols in the blinder's set
# are used. Products/executives are left to alias discovery and the P1 B.identifies_company selection.
BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "GOOGL": ("Google",), "GOOG": ("Google",), "META": ("Facebook",), "AMZN": ("Amazon",),
    "BRK.B": ("Berkshire",), "JPM": ("JPMorgan", "J.P. Morgan", "JP Morgan"), "BAC": ("BofA", "Bank of America"),
    "WFC": ("Wells Fargo",), "C": ("Citi", "Citibank", "Citigroup"), "GS": ("Goldman", "Goldman Sachs"),
    "MS": ("Morgan Stanley",), "KO": ("Coca-Cola", "Coke"), "PEP": ("PepsiCo", "Pepsi"),
    "PG": ("P&G", "Procter & Gamble"), "JNJ": ("J&J", "Johnson & Johnson"), "UNH": ("UnitedHealth", "UnitedHealthcare"),
    "HD": ("Home Depot",), "WMT": ("Walmart", "Wal-Mart"), "COST": ("Costco",), "DIS": ("Disney",),
    "NFLX": ("Netflix",), "NVDA": ("Nvidia",), "MSFT": ("Microsoft",), "TSLA": ("Tesla",), "INTC": ("Intel",),
    "CSCO": ("Cisco",), "ORCL": ("Oracle",), "CRM": ("Salesforce",), "ADBE": ("Adobe",), "QCOM": ("Qualcomm",),
    "AVGO": ("Broadcom",), "TXN": ("Texas Instruments",), "MU": ("Micron",), "T": ("AT&T",), "VZ": ("Verizon",),
    "TMUS": ("T-Mobile",), "CMCSA": ("Comcast", "NBCUniversal"), "XOM": ("Exxon", "ExxonMobil", "Exxon Mobil"),
    "CVX": ("Chevron",), "COP": ("ConocoPhillips",), "BA": ("Boeing",), "CAT": ("Caterpillar",),
    "LMT": ("Lockheed", "Lockheed Martin"), "RTX": ("Raytheon",), "HON": ("Honeywell",), "UPS": ("UPS",),
    "FDX": ("FedEx",), "MCD": ("McDonald's", "McDonalds"), "SBUX": ("Starbucks",), "NKE": ("Nike",),
    "LLY": ("Eli Lilly", "Lilly"), "PFE": ("Pfizer",), "MRK": ("Merck",), "ABBV": ("AbbVie",),
    "BMY": ("Bristol-Myers", "Bristol Myers", "Bristol-Myers Squibb", "Bristol Myers Squibb"), "AMGN": ("Amgen",),
    "GILD": ("Gilead",), "CVS": ("CVS",), "V": ("Visa",), "MA": ("Mastercard",), "PYPL": ("PayPal",),
    "AXP": ("American Express", "Amex"), "BLK": ("BlackRock",), "SCHW": ("Schwab", "Charles Schwab"),
    "UBER": ("Uber",), "ABNB": ("Airbnb",), "BKNG": ("Booking Holdings", "Priceline"), "CMG": ("Chipotle",),
    "LOW": ("Lowe's", "Lowes"), "MO": ("Altria",), "PM": ("Philip Morris",), "AMD": ("AMD",), "IBM": ("IBM",),
    "GE": ("General Electric",), "GM": ("General Motors",), "F": ("Ford",), "DAL": ("Delta Air Lines",),
    "UAL": ("United Airlines",), "AAL": ("American Airlines",), "LUV": ("Southwest Airlines",),
    "SPGI": ("S&P Global",), "MMM": ("3M",), "DE": ("John Deere", "Deere"), "TGT": ("Target Corp",),
    "PLTR": ("Palantir",), "SNOW": ("Snowflake",), "SHOP": ("Shopify",), "COIN": ("Coinbase",),
    "MRNA": ("Moderna",), "ZM": ("Zoom Video",), "SQ": ("Block Inc",), "XYZ": ("Block Inc",),
    "TWTR": ("Twitter",), "ATVI": ("Activision", "Activision Blizzard"), "PARA": ("Paramount",),
    "WBD": ("Warner Bros. Discovery", "Warner Bros Discovery"), "CHTR": ("Charter Communications",),
    "NOW": ("ServiceNow",), "INTU": ("Intuit",), "ISRG": ("Intuitive Surgical",), "TMO": ("Thermo Fisher",),
    "DHR": ("Danaher",), "ABT": ("Abbott",), "MDT": ("Medtronic",), "SMCI": ("Supermicro", "Super Micro"),
    "MSTR": ("MicroStrategy", "Strategy Inc"), "ANET": ("Arista",), "LRCX": ("Lam Research",),
    "AMAT": ("Applied Materials",), "KLAC": ("KLA Corp", "KLA"), "MRVL": ("Marvell",), "PANW": ("Palo Alto Networks",),
    "CRWD": ("CrowdStrike",), "DELL": ("Dell",), "HPQ": ("HP Inc",), "ENPH": ("Enphase",), "FSLR": ("First Solar",),
    "OXY": ("Occidental",), "SLB": ("Schlumberger",), "HAL": ("Halliburton",), "EOG": ("EOG",),
    # FPL (NEE's SEC name until 2010, its utility subsidiary Florida Power & Light since) and Wellpoint (ELV's name
    # until 2014, its health-plan brand again since 2022, archive item "... Health Plan Brand Wellpoint") are single-token
    # former names that would expire with the SEC name; listed here they are masked at every date
    "ELV": ("Anthem", "Elevance", "Wellpoint"),
    "NEE": ("FPL", "Florida Power & Light", "Florida Power and Light"),
}

# The feed's own short names that the SEC names and the brand list did not mask: found by the WS-C2 leak scan
# (`data.leak_probe`: the name each symbol carries in the archive's broker-action and options-activity headlines;
# `scripts/blinding_leak_scan.py`, 2026-09-27), then checked by hand. Merged into BRAND_ALIASES below.
FEED_SHORT_NAMES: dict[str, tuple[str, ...]] = {
    "AMAT": ("Applied Materials", "Applied Mat"), "COF": ("Capital One", "Capital One Finl", "Capital One Financial"),
    "DG": ("Dollar General", "Dollar Gen"), "EBAY": ("eBay",), "MMM": ("3M",),
    "IFF": ("Intl Flavors & Fragrances", "International Flavors & Fragrances", "Intl Flavors", "International Flavors"),
    "NXPI": ("NXP", "NXP Semiconductors"), "RCL": ("Royal Caribbean", "Royal Caribbean Group", "Royal Caribbean Gr",
                                                   "Royal Caribbean Cruises"),
    "USB": ("U.S. Bancorp", "US Bancorp"), "TFC": ("Truist", "Truist Financial", "Truist Finl", "BB&T"),
    "DD": ("DuPont de Nemours", "DuPont"), "EL": ("Estee Lauder", "Estee Lauder Cos", "Estee Lauder Companies"),
    "HPE": ("Hewlett Packard Enterprise", "Hewlett Packard"), "BDX": ("Becton Dickinson", "Becton, Dickinson", "BD"),
    "FIS": ("Fidelity National Info", "Fidelity National Information", "Fidelity National Information Services"),
    "CTSH": ("Cognizant", "Cognizant Tech Solns", "Cognizant Technology Solutions"),
    "STX": ("Seagate", "Seagate Tech", "Seagate Technology"), "MRVL": ("Marvell", "Marvell Tech", "Marvell Technology"),
    "PNC": ("PNC Financial Services", "PNC Finl Servs", "PNC Financial"), "GE": ("General Electric", "GE Aerospace",
                                                                              "GE Aero"),
    "WBD": ("Warner Bros. Discovery", "Warner Bros Discovery", "Warner Bros.Discovery"), "MDLZ": ("Mondelez",),
    "FCX": ("Freeport-McMoRan", "Freeport McMoRan", "Freeport"), "ON": ("onsemi", "ON Semiconductor"),
    "LULU": ("lululemon", "lululemon athletica"), "HON": ("Honeywell", "Honeywell Intl"),
    "PM": ("Philip Morris", "Philip Morris Intl", "Philip Morris International"),
    "MAR": ("Marriott", "Marriott Intl", "Marriott International"), "COIN": ("Coinbase", "Coinbase Glb"),
    "CRWD": ("CrowdStrike", "CrowdStrike Holdings"), "UAL": ("United Airlines", "United Airlines Holdings"),
    "AAL": ("American Airlines", "American Airlines Group"), "LOW": ("Lowe's", "Lowes", "Lowe's Companies"),
    "MCD": ("McDonald's", "McDonalds", "McDonald"),
}
BRAND_ALIASES = {s: tuple(dict.fromkeys((*BRAND_ALIASES.get(s, ()), *FEED_SHORT_NAMES.get(s, ()))))
                 for s in (*BRAND_ALIASES, *(k for k in FEED_SHORT_NAMES if k not in BRAND_ALIASES))}


def _one_word(name: str) -> bool:
    return len(re.split(r"[\s\-]+", name.strip())) == 1


def is_word_like(name: str) -> bool:
    """A single token that is an ordinary word (COMMON_WORDS or WORD_NAMES): masked only in explicit contexts."""
    return " " not in name.strip() and name.strip().casefold() in (COMMON_WORDS | WORD_NAMES)


# ---------------------------------------------------------------------------------- matching keys
@functools.lru_cache(maxsize=8192)
def _fold_char(ch: str) -> str:
    """A precomposed Latin letter with diacritics -> its base letter ("é" -> "e", "ē" -> "e"), a typographic
    apostrophe -> "'"; anything else is returned unchanged. One character in, one out, so spans on folded text are spans of the original."""
    if ch.isascii():
        return ch
    if ch in "\u2018\u2019\u02bc":  # typographic apostrophes: "McDonald’s" is "McDonald's"
        return "'"
    d = unicodedata.normalize("NFKD", ch)
    if len(d) > 1 and d[0].isascii() and d[0].isalpha() and all(unicodedata.combining(c) for c in d[1:]):
        return d[0]
    return ch


def fold_accents(text: str) -> str:
    """Length-preserving diacritic fold (`_fold_char` per character)."""
    return text if text.isascii() else "".join(_fold_char(c) for c in text)


_GAP = re.compile(r"[,.]?\s+")


def alias_key(name: str) -> str:
    """Lookup key of a name alias and of the text it matched: diacritics folded, a comma or period before a space
    dropped, whitespace collapsed, casefolded ("Becton, Dickinson" and "BECTON DICKINSON" share a key)."""
    return _GAP.sub(" ", fold_accents(name).strip()).casefold()


def _distinctive(word: str) -> bool:
    w = re.sub(r"[^\w]", "", word).casefold()
    return len(w) >= 3 and w not in COMMON_WORDS and w not in WORD_NAMES


def is_case_free(name: str) -> bool:
    """Names matched in any case: registered with a lower-case first letter ("lululemon", "eBay", "onsemi"), or
    multi-word with at least two distinctive (not common, not word-like) words ("estee lauder companies")."""
    n = name.strip()
    if not n or is_word_like(n):
        return False
    if n[0].islower():
        return True
    return sum(_distinctive(w) for w in n.split()) >= 2


# ---------------------------------------------------------------------------------- pseudonyms
def pseudonym(symbol: str, salt: str = SALT) -> str:
    """`Company-XXXXXXXX`: 8 upper-case hex characters of sha256(salt + symbol)."""
    return "Company-" + hashlib.sha256((salt + symbol).encode()).hexdigest()[:8].upper()


# code names no firm may receive: the content-free probe's focal code name (question bank
# conventions.content_free.code_name, QB-2; tests/test_integration_day2.py ties the two)
RESERVED_CODE_NAMES = frozenset({"Company-00000000"})


def assert_unique_pseudonyms(symbols: Iterable[str], salt: str = SALT) -> dict[str, str]:
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for s in symbols:
        p = pseudonym(s, salt)
        if p in RESERVED_CODE_NAMES:
            raise ValueError(f"pseudonym of {s} is the reserved code name {p}; change the salt")
        if p in seen and seen[p] != s:
            raise ValueError(f"pseudonym collision: {s} and {seen[p]} -> {p}; change the salt")
        seen[p], out[s] = s, p
    return out


# ---------------------------------------------------------------------------------- aliases
def clean_name(name: str) -> str:
    """SEC entity name without legal suffixes and incorporation tags ("CHASE MANHATTAN CORP /DE/" ->
    "CHASE MANHATTAN", "WELLS FARGO & COMPANY/MN" -> "WELLS FARGO", "IAC/INTERACTIVECORP" -> "IAC
    INTERACTIVECORP": a tag is 1-4 letters closed by '/', a space or the end)."""
    n = re.sub(r"/\s*[A-Za-z]{1,4}\s*(?:/|$|(?=\s))", " ", name)
    n = re.sub(r"[,.]", " ", n)
    n = _SUFFIX.sub(" ", n)
    n = re.sub(r"[/\\]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # trailing foreign legal forms once the dots are gone ("NXP Semiconductors N V", "KASLION Acquisition B V")
    n = re.sub(r"(?:\s+(?:n\s?v|b\s?v|s\s?a|s\s?e|a\s?g|s\s?p\s?a|asa|oyj))+$", "", n, flags=re.I).strip()
    return re.sub(r"(?:\s+(?:&|and))+$", "", n, flags=re.I).strip()


Interval = tuple["pd.Timestamp | None", "pd.Timestamp | None"]  # a claim is valid on [start, end); None = open


@dataclass(frozen=True)
class Executive:
    """A person the firm's own 10-K lists as an executive officer or officer signer (scripts/build_executive_aliases.py;
    docs/impl/dev2_blind.md §4), masked as EXEC_MASK. Each full-name form is masked from its own first listing 10-K's
    usable time (`form_starts`; `start` is the first listing's); the `surname` alone is masked from `start` in a text
    tagged to the firm or naming the person in full, and everywhere from `surname_from` (the surname is distinctive in
    the news before then; None: never), except where it is someone or something else (`_surname_is_someone_else`)
    and, for a common-word surname (`surname_word`), outside a person context and never everywhere (v4)."""

    name: str
    forms: tuple[str, ...]
    surname: str
    start: pd.Timestamp | None = None
    surname_from: pd.Timestamp | None = None
    # v4 (DEV #2 audit): each full-name form from the first 10-K that yields it (aligned with `forms`; empty: every form
    # from `start`), so a form first printed in a later 10-K is never masked retroactively
    form_starts: tuple[pd.Timestamp | None, ...] = ()
    # v4: the surname is a common English word ("Cook", "Smith", "Call"): masked alone only in a person context (a
    # title right before it or a verb of speech right after it), never by the everywhere rule
    surname_word: bool = False

    def form_start(self, i: int) -> pd.Timestamp | None:
        return self.form_starts[i] if i < len(self.form_starts) and self.form_starts[i] is not None else self.start


@dataclass(frozen=True)
class FirmAliases:
    names: tuple[str, ...]  # case-insensitive, capitalized in text, masked wherever they occur
    tickers: tuple[str, ...]  # case-sensitive
    word_names: tuple[str, ...] = ()  # ordinary words: explicit contexts or text tagged to the firm only
    # validity of this firm's claim on an alias (names by `alias_key`, tickers exact); a missing alias is undated
    valid_names: Mapping[str, Interval] = field(default_factory=dict, compare=False, hash=False)
    valid_tickers: Mapping[str, Interval] = field(default_factory=dict, compare=False, hash=False)
    executives: tuple[Executive, ...] = field(default=(), compare=False, hash=False)
    # v4: curated names that are this firm's only inside their validity (manual list `valid_from`/`valid_to`: a unit
    # acquired during the study period, "E*TRADE" of MS from 2020-10-02); outside it they are not masked at all
    gated_names: frozenset[str] = field(default=frozenset(), compare=False, hash=False)


def _name_variants(n: str) -> set[str]:
    out = {n}
    if " " in n and n.count(" ") <= 2:
        out.add(n.replace(" ", "-"))  # COCA COLA -> Coca-Cola
    return out


def _utc(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    t = pd.Timestamp(x)
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _widen(old: Interval | None, new: Interval) -> Interval:
    """Union hull of two validity intervals (None bounds are open)."""
    if old is None:
        return new
    s = None if old[0] is None or new[0] is None else min(old[0], new[0])
    e = None if old[1] is None or new[1] is None else max(old[1], new[1])
    return s, e


def company_aliases(
    symbol: str,
    sec_submission: dict | None = None,
    extra: Iterable[str] = (),
    history_tickers: Iterable[str | tuple] = (),
    *,
    hard: Iterable[str] = (),
    soft: Iterable[str] = (),
    executives: Iterable[Executive] = (),
    dated: Mapping[str, Interval] | None = None,
) -> FirmAliases:
    """SEC current and former names (cleaned, with their validity from SEC `formerNames`), the current name's
    distinctive first word, the symbol and its SEC and historical tickers (`(ticker, from, to)` tuples are
    dated claims), plus `extra` names (brand list, discovered aliases). Single-token names that are ordinary
    words go to `word_names`; a word-like former name keeps its validity, so it expires when the firm drops it.
    `hard`: curated names masked wherever they occur even though they are word-like (manual list mode "name", verified
    on the news archive: "Visa", "3M"); `soft`: curated names masked only in explicit contexts or tagged text even if
    they are not ordinary words (mode "word": "Chase"); `executives`: the firm's executives (masked as EXEC_MASK);
    `dated`: curated names valid only on an interval (masked only inside it, `FirmAliases.gated_names`)."""
    names: set[str] = set()
    words: set[str] = set()
    valid_n: dict[str, Interval] = {}
    tickers: set[str] = {symbol}
    valid_t: dict[str, Interval] = {}
    undated_n: set[str] = set()
    hard_keys = {alias_key(h) for h in hard if h}
    soft_keys = {alias_key(x) for x in soft if x} - hard_keys

    def add_name(n: str, interval: Interval | None, min_len: int = 3) -> None:
        if len(n) < min_len:  # SEC names need 3 characters; a curated brand may have 2 ("3M")
            return
        if (is_word_like(n) and alias_key(n) not in hard_keys) or alias_key(n) in soft_keys:
            words.add(n)
            variants = {n}
        else:
            variants = _name_variants(n)
            names.update(variants)
        for v in variants:
            k = alias_key(v)
            if interval is None or interval == (None, None):
                undated_n.add(k)
            else:
                valid_n[k] = _widen(valid_n.get(k), interval)

    if sec_submission:
        formers = [(clean_name(f.get("name", "") or ""), _utc(f.get("from")), _utc(f.get("to")))
                   for f in (sec_submission.get("formerNames") or [])]
        cur_start = max((e for _, _, e in formers if e is not None), default=None)
        cur = clean_name(sec_submission.get("name", "") or "")
        add_name(cur, (cur_start, None))
        for n, s, e in formers:
            add_name(n, (s, e))
        ws = cur.split()
        if len(ws) >= 2 and len(ws[0]) >= 4 and ws[0].isalpha() and ws[0].casefold() not in COMMON_WORDS:
            add_name(ws[0] if not ws[0].isupper() else ws[0].title(), (cur_start, None))
        tickers |= set(sec_submission.get("tickers", []) or [])
    for h in history_tickers:
        if isinstance(h, str):
            tickers.add(h)
        else:
            t, s, e = h
            tickers.add(t)
            if t != symbol:
                valid_t[t] = _widen(valid_t.get(t), (_utc(s), _utc(e)))
    for t in list(tickers):
        for v in ({t.replace(".", "-"), t.replace(".", "/")} if "." in t else set()) | \
                 ({t.replace("-", ".")} if "-" in t else set()):
            tickers.add(v)
            if t in valid_t:
                valid_t[v] = valid_t[t]
    dated = dict(dated or {})
    gated: set[str] = set()
    for e in (*extra, *hard, *soft):
        if e:
            add_name(e, dated.get(e), min_len=2)
            if e in dated:
                gated |= {alias_key(v) for v in _name_variants(e) | {e}}
    # a word-like SEC name that the manual list makes hard ("VISA" of VISA INC.) is a hard name too
    for w in list(words):
        if alias_key(w) in hard_keys:
            words.discard(w)
            names.add(w)
    # an all-caps name equal to a short or word-like ticker (IBM, UPS, CVS) is matched as a ticker only,
    # case-sensitively; longer ones (WYNN) stay names too, since "Wynn" is how text writes them
    drop = {n for n in names | words if n in tickers and (len(n) <= 3 or n in COMMON_WORD_TICKERS)}
    names -= drop
    words -= drop
    hardn = {alias_key(n) for n in names}
    words = {min(g) for g in _by_casefold(w for w in words if alias_key(w) not in hardn).values()}
    valid_n = {k: v for k, v in valid_n.items() if k not in undated_n}  # an undated claim covers all dates
    key = (lambda x: (-len(x), x))
    return FirmAliases(tuple(sorted(names, key=key)), tuple(sorted(tickers, key=key)), tuple(sorted(words, key=key)),
                       valid_n, valid_t, tuple(executives), frozenset(k for k in gated if k in valid_n))


def _by_casefold(xs: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for x in xs:
        out[alias_key(x)].append(x)
    return out


def _is_ticker_like(a: str) -> bool:
    return re.fullmatch(r"[A-Z]{1,5}(?:[.\-/][A-Z]{1,2})?", a) is not None


def _as_firm_aliases(v: FirmAliases | Iterable[str], symbol: str) -> FirmAliases:
    if isinstance(v, FirmAliases):
        return v
    items = list(v)
    tickers = {a for a in items if a == symbol or _is_ticker_like(a)}
    rest = [a for a in items if a not in tickers]
    return FirmAliases(tuple(a for a in rest if not is_word_like(a)), tuple(sorted(tickers | {symbol})),
                       tuple(a for a in rest if is_word_like(a)))


def _trie_pattern(words: Iterable[str]) -> str:
    """Regex matching any of `words`, factored as a trie (fast on thousands of aliases; prefers the longest)."""
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = {}

    def build(node: dict) -> str:
        end = "" in node
        alts = [re.escape(ch) + build(sub) for ch, sub in sorted(node.items()) if ch != ""]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        if end:
            return "(?:" + body + ")?"
        return body

    return build(trie)


_EXCHANGE = r"(?:NYSE American|NYSE Arca|NYSE|NASDAQ|Nasdaq|AMEX|Cboe|CBOE|BATS|OTC)"
# explicit company contexts for a word-like name: a legal suffix after it, or its own ticker in parentheses
_LEGAL_SUFFIX = re.compile(r"\s+(?:Inc|Corp|Corporation|Co|Cos|Company|Companies|Group|Holdings|Ltd|Limited|plc|PLC|"
                           r"LLC|N\.V|S\.A)\b")
_PAREN_TICKER = re.compile(rf"\s*\(\s*(?:{_EXCHANGE}\s*:\s*)?([A-Z]{{1,5}}(?:[.\-/][A-Z]{{1,2}})?)\s*\)")
# legal and feed designators masked together with the name they follow (module doc); case-sensitive
_DESIGNATOR = (r"(?:,?\s+(?:Inc|Corp|Corporation|Co|Cos|Company|Companies|Group|Gr|Grp|Holdings?|Hldgs|Ltd|Limited|"
               r"plc|PLC|LLC|LP|L\.P|N\.\s?V|NV|S\.\s?A|Intl|Glb|Finl|Servs|Svcs|Solns)\.?"
               r"|\s+(?:&|and)\s+Cos?\.?|\.com)(?![A-Za-z0-9\-])")
# not inside a word, after "$" or inside a decimal ("$3M", "1.3M"); v4: a name that starts with a letter may follow a
# number and a period with no space ("... roughly $51,000.Exxon Mobil Corp.": a missing space, DEV #2 audit)
_NAME_START = r"(?<![A-Za-z0-9$])(?:(?<![0-9][.,])|(?=[A-Za-z]))"
# a web host name: dot-separated labels ending in a common top-level domain (generic ones in any case, "Amazon.Com";
# country codes lower-case only, so a sentence run together, "sales.In", is not a host)
_TLD = r"(?:(?i:com|net|org)|io|co|ai|tv|us|biz|info|app|gov|edu|uk|ca|de|jp|cn|fr|eu|au|ch|nl|kr|in)"
_HOST = rf"(?<![A-Za-z0-9.\-_])(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{{0,61}}[A-Za-z0-9])?\.)+{_TLD}(?![A-Za-z0-9\-_])"
# labels that never name a firm (second-level public suffixes, "www")
_HOST_SKIP = frozenset("com net org co io ai tv us biz info app gov edu ac www www2 m".split())
# a tagged firm's distinctive name matches as a label prefix from 4 characters ("ebay"inc, "fedex"cares) and as a
# suffix from 6 ("shop"disney, "about"amazon; a 5-letter suffix such as "apple" would also match "pineapple")
AFFIX_MIN, SUFFIX_MIN = 4, 6


# a name that starts with a digit ("3M") is a quantity ("3M people", "more than 3M", "2M or 3M", "3M+") unless the text
# is tagged to its firm or the context is explicit ("3M Co", "3M (MMM)"); verified on the news archive before 2024-07
# (docs/impl/dev2_blind.md §2: 661 of 688 items with "3M" are tagged MMM; the others are counts)
_QTY_BEFORE = re.compile(r"(?i)(?:\b(?:than|over|about|nearly|almost|around|some|roughly|approximately|approx\.?|least|"
                         r"extra|another|additional|up to|total|record|just|only)|\d[\d.,]*\s*[KMB]?\s+(?:or|to|and)|"
                         r"[~≈+<>])\s*$")
_QTY_AFTER = re.compile(
    r"(?i)(?:\+|-plus\b|\s*-\s*\d|\s*/)|\s+(?:people|persons|tests|vehicles|cars|doses|units|users?|courses|jobs?|acres|"
    r"barrels|bpd|passengers|customers|subscribers|subs|members|accounts|homes|households|visitors|views|downloads|"
    r"devices|phones|iphones|boxes|locations|tons|tonnes|miles|square|sq|fewer|more|new|additional|covid(?:-19)?|"
    r"vaccinations?|vaccines?|roubles|rubles|euros|pounds|yen|won|yuan|rupees|dollars|usd|per|each|annually|"
    r"shares\s+(?:in|of)|of\s+(?!its\b|the\b))\b")


def _own_name_token(token: str, e: Executive) -> bool:
    """Whether `token` can be one of the executive's own given or middle names: a token of a full-name form other
    than the surname, or a word whose initial is one of the form's middle initials ("Thomas" for "Brian T. Moynihan")."""
    t = token.casefold()
    last = alias_key(e.surname).split()
    for f in e.forms:
        ws = alias_key(f).replace(".", " ").split()
        ws = ws[:len(ws) - len(last)] if ws[len(ws) - len(last):] == last else ws[:-1]
        if t in ws or any(len(w) == 1 and t[:1] == w for w in ws[1:]):
            return True
    return False


def _own_initial(ch: str, e: Executive) -> bool:
    """Whether a capital initial can be one of the executive's own first or middle initials."""
    c = ch.casefold()
    last = alias_key(e.surname).split()
    for f in e.forms:
        ws = alias_key(f).replace(".", " ").split()
        ws = ws[:len(ws) - len(last)] if ws[len(ws) - len(last):] == last else ws[:-1]
        if any(w[:1] == c for w in ws):
            return True
    return False


# Surname-alone guards (text rules v4, DEV #2 audit of the executive rules). A surname alone is someone or something
# else when it follows another person's given name, possibly with middle initials ("Fed Governor Lisa D. Cook"), or
# initials that are not the executive's own ("A.O. Smith"); when a public official's title precedes it ("Fed's Cook
# Says", "Senator Warren"; the person's given name is read by the first rule: "Fed Governor Lisa D. Cook"); or when it
# is part of a place or organisation name ("Cook County", "Cook Inlet", "Solomon Islands", "Hastings Entertainment",
# "Battle of Hastings", "Moynihan Train Hall", "Fraser Institute"). EXEC_PROTECTED lists compounds these rules cannot
# see ("MSCI Barra", "Hannon Armstrong", "Dorsey & Whitney": an "&" rule would also unmask "Musk & Board").
_PERSON_BEFORE = re.compile(r"(?:\b([A-Za-z][a-z]+)\s+)?((?:(?<![A-Za-z])[A-Z]\.\s*){0,3})$")  # not "FSD. Musk"
_OFFICIAL_BEFORE = re.compile(
    r"(?:\bFed(?:'s|’s)?|\bFederal Reserve(?:'s|’s)?|\b(?:Governor|Gov\.|Senator|Sen\.|Rep\.|Representative|"
    r"Congressman|Congresswoman|Judge|Justice|Minister|Mayor|Ambassador|Commissioner))"
    r"(?:\s+(?:Governor|Chair|Vice Chair|President|Board|Bank|of [A-Z][a-z]+))*(?:\s+[A-Z]\.){0,2}\s+$")
_COMPOUND_AFTER = re.compile(
    r"\s+(?:County|Inlet|Islands?|Institute|University|College|School|Hall|Train|Medical|Entertainment|"
    r"Mines?|Street|Avenue|Center|Centre|River|Park|Bay|Road|Airport|Stadium|Hospital|Brothers|Valley|Agency|World|"
    r"Sustainable|Partners|Associates|LLP|Prize|Award|Doctrine|Index)\b")
_COMPOUND_BEFORE = re.compile(r"\b(?:Battle|Siege|Isle|Cape|Port|Fort|Mount|Mt\.|Saint|St\.|King|Queen|Lake)\s+"
                              r"(?:of\s+)?$")
EXEC_PROTECTED = ("MSCI Barra", "Barra risk", "Hannon Armstrong", "Cook Up", "Neil Armstrong", "Dorsey & Whitney",
                  "Dorsey and Whitney")
_EXEC_PROTECTED = re.compile(r"(?<![A-Za-z])(?:" + "|".join(p.replace(" ", r"\s+") for p in EXEC_PROTECTED) +
                             r")(?![A-Za-z])")
# a person context for a surname that is a common word: a title right before it, a firm's possessive ("Apple's Cook";
# a public official's "Fed's" is caught first by _OFFICIAL_BEFORE), an age in apposition ("Cook, 61,"), or a verb of
# speech or of a person's action right after it
_TITLE_BEFORE = re.compile(r"\b(?:CEO|CFO|COO|CTO|CIO|EVP|SVP|VP|Chairman|Chairwoman|Chairperson|Chair|President|"
                           r"Chief|Founder|Co-Founder|Cofounder|founder|co-founder|boss|Boss|Executive|executive|Mr\.|"
                           r"Ms\.|Mrs\.|Dr\.)\s+$|\b[A-Z][\w&.-]*(?:'s|’s)\s+$")
_AGE_AFTER = re.compile(r",\s*\d{2}\s*,")
_SPEECH_AFTER = re.compile(
    r"(?i)(?:'s|’s)?,?\s+(?:says|said|say|told|tells|wrote|writes|added|adds|noted|notes|warned|warns|warning|stated|"
    r"states|explained|explains|argued|argues|predicted|predicts|expects|expected|sees|saw|believes|believed|thinks|"
    r"thought|announced|announces|confirmed|confirms|commented|comments|remarked|remarks|tweeted|tweets|posted|posts|"
    r"responded|responds|replied|replies|claimed|claims|admitted|admits|insisted|insists|urged|urges|reiterated|"
    r"reiterates|acknowledged|acknowledges|emphasized|emphasizes|stressed|stresses|touted|touts|praised|praises|"
    r"criticized|criticizes|defended|defends|discussed|discusses|revealed|reveals|joked|jokes|testified|testifies|"
    r"spoke|speaks|asked|asks|answered|answers|hinted|hints|signaled|signals|signalled|outlined|outlines|described|"
    r"describes|takes|took|meets|met|visits|visited|spends|spent|shared|sells|sold|buys|bought|leads|led|joins|joined|"
    r"hosts|hosted|unveils|unveiled|launches|launched|earns|earned|aims|hopes|wants|plans|is|was|has|had|will|would)\b"
    r"|(?:'s|’s)\s")  # "Cook is the only one", "which Cook had initiated", "Cook's comments"


def _surname_is_someone_else(folded: str, start: int, end: int, e: Executive) -> bool:
    """The guards above, for the surname alone at folded[start:end] and the executive `e`."""
    before = folded[max(0, start - 48):start]
    m = _PERSON_BEFORE.search(before)
    if m and m.group(1) and m.group(1).casefold() in GIVEN_NAMES and not _own_name_token(m.group(1), e):
        return True  # another person's given name, possibly with middle initials in between ("Lisa D. Cook")
    if m and m.group(2) and not (m.group(1) and _own_name_token(m.group(1), e)):
        if not _own_initial(m.group(2).strip()[:1], e):
            return True  # initials that are not the executive's own ("A.O. Smith")
    if _OFFICIAL_BEFORE.search(before) or _COMPOUND_BEFORE.search(before) or _COMPOUND_AFTER.match(folded, end):
        return True
    return False


def _person_context(folded: str, start: int, end: int) -> bool:
    return bool(_TITLE_BEFORE.search(folded[max(0, start - 24):start]) or _SPEECH_AFTER.match(folded, end)
                or _AGE_AFTER.match(folded, end))


def _counted(text: str, start: int, end: int) -> bool:
    return bool(_QTY_BEFORE.search(text[max(0, start - 24):start]) or _QTY_AFTER.match(text, end))


def _compact(s: str) -> str:
    """Letters and digits of a folded, casefolded name or host label ("T-Mobile" and "t-mobile" -> "tmobile")."""
    return re.sub(r"[^0-9a-z]", "", fold_accents(s).casefold())


def _name_pattern(names: Iterable[str]) -> str:
    """Trie over the folded names; a space in a name matches a run of whitespace, optionally after "," or "."."""
    # lower-cased so that one trie branch holds every casing of a name and the longest alias always wins (the
    # pattern is matched under (?i); a capital first letter is required separately)
    return _trie_pattern(sorted({fold_accents(n).replace("\\", "").lower() for n in names})).replace(
        "\\ ", r"[,.]?\s+")


@dataclass(frozen=True)
class Claim:
    """One firm's claim on an alias string."""

    symbol: str
    soft: bool = False  # word-like name: explicit context or tagged text only
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    own: bool = False  # the ticker is the firm's own symbol
    expires: bool = False  # a single-token former name (IAC, TRAVELERS): only while it was the firm's name
    gated: bool = False  # v4: a dated curated name ("E*TRADE" of MS): only while the claim is valid

    def status(self, c: pd.Timestamp | None) -> int:
        """3 valid at c, 2 undated, 1 not yet valid, 0 expired. Without a cutoff every dated claim with an end
        counts as expired (conservative for word-like former names)."""
        if self.start is None and self.end is None:
            return 2
        if c is None:
            return 0 if self.end is not None else 3
        if self.end is not None and c >= self.end:
            return 0
        if self.start is not None and c < self.start:
            return 1
        return 3


@dataclass(frozen=True)
class _Compiled:
    pattern: re.Pattern | None
    names: Mapping[str, tuple[Claim, ...]]  # casefolded alias -> claims
    tickers: Mapping[str, tuple[Claim, ...]]
    hosts: Mapping[str, tuple[Claim, ...]] = field(default_factory=dict)  # compact name -> claims (any host label)
    host_tickers: Mapping[str, tuple[Claim, ...]] = field(default_factory=dict)  # compact ticker (registrable label)
    affix: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)  # symbol -> (compact, alias key)
    # executives (docs/impl/dev2_blind.md §4): a separate first pass, so that "Michael Dell" is one person and never
    # "Michael Company-X"; full-name form key -> (symbol, executive), eligible surname key -> (symbol, executive)
    exec_pattern: re.Pattern | None = None
    exec_forms: Mapping[str, tuple[tuple[str, Executive, pd.Timestamp | None], ...]] = field(default_factory=dict)
    exec_surnames: Mapping[str, tuple[tuple[str, Executive], ...]] = field(default_factory=dict)


class Blinder:
    """Masks every universe firm with its pseudonym, then levels, then absolute times."""

    def __init__(
        self,
        firms: Mapping[str, FirmAliases | Iterable[str]],
        salt: str = SALT,
        discovery: AliasDiscovery | None = None,
        lives: Mapping[str, tuple] | None = None,
    ):
        """`lives`: symbol -> (first, last) trading session, used only to resolve an alias that several firms
        claim (the one trading at the cutoff wins); see `lives_from_bars`."""
        self.salt = salt
        self.firms = {s: _as_firm_aliases(v, s) for s, v in firms.items()}
        self.code_names = assert_unique_pseudonyms(self.firms, salt)
        self.discovery = discovery
        self.lives = {s: (_utc(a), _utc(b) + pd.Timedelta(days=1)) for s, (a, b) in (lives or {}).items()
                      if _utc(a) is not None and _utc(b) is not None}
        self._static = self._compile({}, None)
        self._by_anchor: dict[pd.Timestamp, _Compiled] = {}
        # aliases several firms claim (resolved per cutoff, never dropped); for reports and tests
        self.shared = {k: tuple(sorted({c.symbol for c in v})) for m in (self._static.names, self._static.tickers)
                       for k, v in m.items() if len({c.symbol for c in v}) > 1}

    @property
    def ambiguous(self) -> set[str]:
        """v0 name: the static aliases claimed by several firms (now resolved point-in-time, not dropped)."""
        return set(self.shared)

    def code_name(self, symbol: str) -> str:
        return self.code_names[symbol]

    # -------------------------------------------------------------- regex construction
    def _compile(self, discovered: Mapping[str, Iterable[str]], anchor: pd.Timestamp | None) -> _Compiled:
        """Pure function of the firm tables and the aliases discovered at `anchor` (no shared state)."""
        names: dict[str, list[Claim]] = defaultdict(list)
        strings: set[str] = set()
        case_free: set[str] = set()
        # alias keys whose counted surface forms include a camel-case one ("macOS" beside "MacOS"): matched in any case
        # whichever form is the most frequent at this anchor
        cf_keys = getattr(self.discovery, "case_free_keys", None)
        cf = cf_keys(anchor) if cf_keys is not None and anchor is not None else frozenset()
        for s, f in self.firms.items():
            for soft, group in ((False, f.names), (True, f.word_names)):
                for n in group:
                    k = alias_key(n)
                    start, end = f.valid_names.get(k, (None, None))
                    names[k].append(Claim(s, soft, start, end, expires=end is not None and _one_word(n),
                                          gated=k in f.gated_names))
                    strings.add(n)
                    if not soft and is_case_free(n):
                        case_free.add(n)
            for n in discovered.get(s, ()):
                # a discovered claim starts at the anchor that first accepted it (the most recent claim wins a shared
                # alias, e.g. a brand that changed owner)
                first_seen = getattr(self.discovery, "first_seen", None)
                start = first_seen(s, n, anchor) if first_seen is not None else None
                names[alias_key(n)].append(Claim(s, is_word_like(n), start or anchor, None))
                strings.add(n)
                if n[:1].islower() or alias_key(n) in cf:  # "iPhone", "eBay": the capital-first branch would miss them
                    case_free.add(n)
        tickers: dict[str, list[Claim]] = defaultdict(list)
        for s, f in self.firms.items():
            for t in f.tickers:
                start, end = f.valid_tickers.get(t, (None, None))
                tickers[t].append(Claim(s, False, start, end, own=(t == s)))
        names_t = {k: tuple(sorted(set(v), key=lambda c: (c.symbol, c.soft))) for k, v in names.items()}
        ticks_t = {k: tuple(sorted(set(v), key=lambda c: (c.symbol, not c.own))) for k, v in tickers.items()}
        plain = [t for t in ticks_t if len(t) >= 2 and t not in COMMON_WORD_TICKERS]
        # host labels: every name key (word-like names too: a host name is an explicit context), tickers of 2+
        # characters that are not words, and each firm's distinctive (not word-like) names for the tagged-firm affix
        hosts: dict[str, set[Claim]] = defaultdict(set)
        affix: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for k, v in names_t.items():
            ck = _compact(k)
            if len(ck) >= 2 and ck not in _HOST_SKIP:
                hosts[ck].update(v)
            if len(ck) >= AFFIX_MIN:
                for x in v:
                    if not x.soft:
                        affix[x.symbol].add((ck, k))
        host_tickers: dict[str, set[Claim]] = defaultdict(set)
        for t in plain:
            ck = _compact(t)
            if ck not in _HOST_SKIP:
                host_tickers[ck].update(ticks_t[t])
        hosts_t = {k: tuple(sorted(v, key=lambda c: (c.symbol, c.soft))) for k, v in hosts.items()}
        host_ticks_t = {k: tuple(sorted(v, key=lambda c: (c.symbol, not c.own))) for k, v in host_tickers.items()}
        affix_t = {s: tuple(sorted(v, key=lambda x: (-len(x[0]), x))) for s, v in affix.items()}
        parts = [rf"(?P<host>{_HOST})(?P<host_sfx>(?:{_DESIGNATOR})*)",
                 rf"(?P<keep>(?<![A-Za-z0-9])(?i:{_PROTECTED})(?![A-Za-z0-9]))"]
        if names_t:
            parts.append(rf"(?P<name>{_NAME_START}(?=[A-Z0-9])(?i:{_name_pattern(strings)})(?![A-Za-z0-9]))"
                         rf"(?P<name_sfx>(?:{_DESIGNATOR})*)")
        if case_free:
            parts.append(rf"(?P<namei>{_NAME_START}(?i:{_name_pattern(case_free)})(?![A-Za-z0-9]))"
                         rf"(?P<namei_sfx>(?:{_DESIGNATOR})*)")
        if names_t:  # an X/Twitter handle names the firm in any case ("@nvidia", "@netflix")
            parts.append(rf"(?<=@)(?P<handle>(?i:{_name_pattern(strings)}))(?![A-Za-z0-9_])")
        if ticks_t:
            tp = _trie_pattern(ticks_t)
            parts.append(rf"\$(?P<dollar>{tp})(?![A-Za-z0-9])")
            parts.append(rf"(?<=\()(?P<paren>{tp})(?=\))")
            parts.append(rf"(?P<xpre>\b{_EXCHANGE}\s*:\s*)(?P<exch>{tp})(?![A-Za-z0-9])")
        if plain:  # v4: also a source tail after " -" ("'... Natural Gas' -GE Newsroom"), never inside "T-Mobile"
            parts.append(rf"(?:(?<![A-Za-z0-9$\-])|(?<=\s-))(?P<plain>{_trie_pattern(plain)})(?![A-Za-z0-9])"
                         rf"(?P<plain_sfx>(?:{_DESIGNATOR})*)")  # "KKR & Co", "IBM Corp": a ticker used as the name
        ex_pat, ex_forms, ex_surnames = self._compile_executives(names_t)
        return _Compiled(re.compile("|".join(parts)), names_t, ticks_t, hosts_t, host_ticks_t, affix_t,
                         ex_pat, ex_forms, ex_surnames)

    def _compile_executives(self, names_t: Mapping[str, tuple[Claim, ...]]):
        """The executive pass: full-name forms and eligible surnames of every firm's executives. A form that is itself a
        firm's name, or the start of one ("Charles Schwab" is SCHW's name), stays with the firm; a surname that is a
        word of any firm's name ("Dell", "Ford", "Fargo") is never masked alone."""
        firm_keys = set(names_t)
        firm_prefixes = {" ".join(k.split()[:i]) for k in firm_keys for i in range(2, len(k.split()) + 1)}
        firm_tokens = {w for k in firm_keys for w in re.split(r"[\s\-]+", k)}
        forms: dict[str, list[tuple[str, Executive, pd.Timestamp | None]]] = defaultdict(list)
        surnames: dict[str, list[tuple[str, Executive]]] = defaultdict(list)
        strings: set[str] = set()
        for s, f in self.firms.items():
            for e in f.executives:
                for i, x in enumerate(e.forms):
                    k = alias_key(x)
                    if len(k) < 5 or " " not in k or k in firm_keys or k in firm_prefixes or k in STOP_PHRASES:
                        continue
                    forms[k].append((s, e, e.form_start(i)))  # v4: each form from its own first 10-K
                    strings.add(x)
                k = alias_key(e.surname)
                if surname_eligible(e.surname) and k not in firm_tokens and k not in firm_keys:
                    surnames[k].append((s, e))
                    strings.add(e.surname)
        if not strings:
            return None, {}, {}
        # not inside a word, a handle or an apostrophe name ("D'Amico"); v4: a name may follow a quote ("'Tim Cook signs
        # ...'") or a hyphen ("Musk-Tim Cook Spat"); a surname alone right after "<letter>-" is part of a hyphenated
        # name ("Wilson-Thompson") unless it is distinctive (`_mask_executives`)
        pat = re.compile(rf"(?<![A-Za-z0-9@#_$.])(?<![A-Za-z0-9]['’])(?=[A-Z])(?i:{_name_pattern(strings)})"
                         rf"(?![A-Za-z0-9_@])")
        return (pat, {k: tuple(v) for k, v in forms.items()}, {k: tuple(v) for k, v in surnames.items()})

    def _mask_executives(self, text: str, folded: str, comp: _Compiled, c: pd.Timestamp | None,
                         tagged: frozenset[str]) -> tuple[str, str]:
        """EXEC_MASK for every executive of the blinder's firms named in the text (the `Executive` rules): a full-name
        form from its own first 10-K; the surname alone from the start in a text tagged to the firm or naming the
        person in full, else from `surname_from`. v4 (DEV #2 audit): the surname alone never where it is someone or
        something else (`_surname_is_someone_else`: another given name or other initials before it, a public
        official's title, a place or organisation name, EXEC_PROTECTED), and a common-word surname (`surname_word`:
        "Cook", "Smith") only in a person context (`_person_context`) and never by the everywhere rule."""
        if comp.exec_pattern is None:
            return text, folded

        def live(t: pd.Timestamp | None) -> bool:
            return t is not None and (c is None or t <= c)

        ms = list(comp.exec_pattern.finditer(folded))
        if not ms:
            return text, folded
        named = {id(e) for m in ms for _, e, fs in comp.exec_forms.get(alias_key(m.group(0)), ()) if live(fs)}
        protected = [p.span() for p in _EXEC_PROTECTED.finditer(folded)]
        out, fout, pos = [], [], 0
        for m in ms:
            k = alias_key(m.group(0))
            start = m.start()
            ok = any(live(fs) for _, _, fs in comp.exec_forms.get(k, ()))
            if not ok and k in comp.exec_surnames and not any(a <= m.start() < b for a, b in protected):
                hyphenated = bool(re.search(r"[A-Za-z0-9]-$", folded[max(0, m.start() - 2):m.start()]))
                for s, e in comp.exec_surnames[k]:
                    if not live(e.start) or _surname_is_someone_else(folded, m.start(), m.end(), e):
                        continue
                    if hyphenated and not (live(e.surname_from) and not e.surname_word):
                        continue  # "Kathleen Wilson-Thompson": part of another person's name
                    if e.surname_word and not _person_context(folded, m.start(), m.end()):
                        continue
                    if s in tagged or id(e) in named or (live(e.surname_from) and not e.surname_word):
                        ok = True
                        # the person's own given and middle names right before it go with it ("Brian Thomas")
                        for _ in range(3):
                            w = re.search(r"([A-Z][A-Za-z]*\.?)\s+$", folded[max(pos, start - 24):start])
                            if not w or not _own_name_token(w.group(1).rstrip("."), e):
                                break
                            start -= len(w.group(0))
                        break
            if not ok:
                continue
            out.append(text[pos:start])
            fout.append(folded[pos:start])
            out.append(EXEC_MASK)
            fout.append(EXEC_MASK)
            pos = m.end()
        out.append(text[pos:])
        fout.append(folded[pos:])
        return "".join(out), "".join(fout)

    def _compiled(self, cutoff: pd.Timestamp | None) -> _Compiled:
        if self.discovery is None or cutoff is None:
            return self._static
        anchor = self.discovery.anchor_for(cutoff)
        if anchor not in self._by_anchor:
            self._by_anchor[anchor] = self._compile(self.discovery.aliases_at(anchor), anchor)
        return self._by_anchor[anchor]

    # -------------------------------------------------------------- resolution
    def _live(self, symbol: str, c: pd.Timestamp | None) -> bool:
        life = self.lives.get(symbol)
        return c is not None and life is not None and life[0] <= c < life[1]

    def _resolve(self, claims: Iterable[Claim], c: pd.Timestamp | None, tagged: frozenset[str]) -> str | None:
        """The firm an alias denotes at cutoff c: tagged to the text, then trading at c, then its own ticker,
        then a claim valid at c, then the most recent claim, then the symbol (deterministic)."""
        cl = list(claims)
        if not cl:
            return None
        if len({x.symbol for x in cl}) == 1:
            return cl[0].symbol
        far = pd.Timestamp.min.tz_localize("UTC")

        def rank(x: Claim):
            return (x.symbol in tagged, self._live(x.symbol, c), x.own, x.status(c),
                    x.start if x.start is not None and (c is None or x.start <= c) else far)

        best = max(rank(x) for x in cl)
        return min(x.symbol for x in cl if rank(x) == best)

    def _host_owner(self, host: str, comp: _Compiled, c: pd.Timestamp | None, tagged: frozenset[str]) -> str | None:
        """The firm a web host name names (module docstring): a label equal to one of its names (the registrable
        label first, then the subdomains right to left) or, in the registrable label only, to one of its tickers; else
        a tagged firm whose distinctive name is a prefix or suffix of a label. None if no label names a firm."""
        labels = [x for x in (_compact(p) for p in host.split(".")[:-1]) if x and x not in _HOST_SKIP]
        for i, lab in enumerate(reversed(labels)):
            ok = [x for x in comp.hosts.get(lab, ()) if not (x.expires and x.status(c) == 0)
                  and not (x.gated and x.status(c) != 3)]
            if not ok and i == 0:
                ok = list(comp.host_tickers.get(lab, ()))
            sym = self._resolve(ok, c, tagged)
            if sym is not None:
                return sym
        for s in sorted(tagged):
            for ck, k in comp.affix.get(s, ()):
                if any(lab != ck and (lab.startswith(ck) or (len(ck) >= SUFFIX_MIN and lab.endswith(ck)))
                       for lab in labels) and any(
                        x.symbol == s and not (x.expires and x.status(c) == 0) and not (x.gated and x.status(c) != 3)
                        for x in comp.names.get(k, ())):
                    return s
        return None

    @staticmethod
    def _explicit(text: str, end: int, tickers: Iterable[str]) -> bool:
        if _LEGAL_SUFFIX.match(text, end):
            return True
        m = _PAREN_TICKER.match(text, end)
        return m is not None and m.group(1) in set(tickers)

    # -------------------------------------------------------------- masking
    def blind_entities(self, text: str, cutoff=None, tagged: Iterable[str] = ()) -> str:
        """Replace every universe firm's names and tickers with its pseudonym. With a discovery object and a
        cutoff, aliases discovered from items before that cutoff are masked too. `tagged`: the firms the text is
        tagged to (a news item's tickers), whose word-like names are masked wherever they appear."""
        if not text:
            return text
        comp = self._compiled(cutoff)
        c = _utc(cutoff)
        tagged = frozenset(tagged)
        folded = fold_accents(text)  # same length: match on the folded text, copy unmatched spans from the original
        text, folded = self._mask_executives(text, folded, comp, c, tagged)

        def rep(m: re.Match, soft_ok: frozenset[str]) -> str | None:
            """The replacement of one match; `soft_ok`: the firms whose word-like names count here (tagged or named)."""
            g = m.lastgroup
            if g in ("name_sfx", "namei_sfx", "plain_sfx", "host_sfx"):  # the (possibly empty) designator group
                g = g[:-4]  # closes the match
            if g == "keep":
                return None
            if g == "host":
                sym = self._host_owner(m.group("host"), comp, c, tagged)
                return None if sym is None else self.code_names[sym]
            if g in ("name", "namei", "handle"):
                if g != "handle" and m.group(g)[:1].isdigit() and _counted(m.string, m.start(g), m.end(g)):
                    ok = [x for x in comp.names[alias_key(m.group(g))] if x.symbol in tagged or
                          self._explicit(m.string, m.end(g), self.firms[x.symbol].tickers)]
                    sym = self._resolve(ok, c, tagged)  # "3M people", "more than 3M": a quantity unless tagged/explicit
                    return None if sym is None else self.code_names[sym]
                ok = [x for x in comp.names[alias_key(m.group(g))]
                      if not (x.expires and x.status(c) == 0) and not (x.gated and x.status(c) != 3)
                      and (not x.soft or x.symbol in soft_ok or self._explicit(m.string, m.end(g), self.firms[x.symbol].tickers))]
                sym = self._resolve(ok, c, tagged)
                return None if sym is None else self.code_names[sym]
            t = m.group("exch" if g == "exch" else g)
            sym = self._resolve(comp.tickers[t], c, tagged)
            return (text[m.start("xpre"):m.end("xpre")] if g == "exch" else "") + self.code_names[sym]

        matches = list(comp.pattern.finditer(folded))
        # v3: a firm the text names (a hard name, a ticker, an explicit context) counts as tagged for its word-like names
        # in the rest of the text: a defined term re-uses the short name ('JPMorgan Chase Bank, N.A. ("Chase")',
        # 'Target Corporation ("Target")'), and the short name would re-identify the masked firm next to it
        sym_of = {v: k for k, v in self.code_names.items()}
        named = {sym_of[x.group(0)] for m in matches if (r := rep(m, tagged)) is not None for x in _CODE.finditer(r)}
        soft_ok = tagged | frozenset(named)
        out, pos = [], 0
        for m in matches:
            r = rep(m, soft_ok)
            if r is None:
                continue
            out.append(text[pos:m.start()])
            out.append(r)
            pos = m.end()
        out.append(text[pos:])
        return "".join(out)

    def blind_entity(self, text: str, symbol: str | None = None) -> str:
        """v0 name kept for callers; masks all universe firms (the focal one included)."""
        return self.blind_entities(text, tagged=() if symbol is None else (symbol,))

    @staticmethod
    def mask_levels(text: str) -> str:
        return mask_levels(text)

    @staticmethod
    def blind_time(text: str, decision_year: int) -> str:
        return blind_time(text, decision_year)

    def __call__(self, text: str, symbol: str | None = None, decision_year: int | None = None, *, cutoff=None,
                 tagged: Iterable[str] = ()) -> str:
        """Entities (all universe firms; `symbol` and `tagged` are the firms the text is about), then `[level]`
        masks, then `[date]` masks and relative years (when a year is given)."""
        tags = set(tagged) | ({symbol} if symbol else set())
        text = self.mask_levels(self.blind_entities(text, cutoff, tags))
        return self.blind_time(text, decision_year) if decision_year is not None else text


# ---------------------------------------------------------------------------------- time blinding
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
_MON = rf"(?:{_MONTHS})\.?"
_DAYN = r"\d{1,2}(?:st|nd|rd|th)?"
_YYYY = r"(?:18|19|20)\d{2}"  # text-rules v3: 1800-2099 (founding years, old statutes, long-dated targets)
# absolute dates, most specific first; each becomes [date] before years are made relative
_DATE_FORMS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(rf"\b{_YYYY}-\d{{1,2}}-\d{{1,2}}(?:[T ]\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?(?:Z|[+-]\d{{2}}:?\d{{2}})?)?(?![\w-])"), DATE_MASK),
    (re.compile(rf"\b{_YYYY}/\d{{1,2}}/\d{{1,2}}\b"), DATE_MASK),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)?\d{2}\b"), DATE_MASK),  # 01/03/2023, 1/3/23, 01-03-2023
    (re.compile(rf"\b{_DAYN}\s+(?:of\s+)?{_MON}(?:,?\s+{_YYYY})?(?!\w)"), DATE_MASK),  # 5 March 2024, 5th of March
    (re.compile(rf"\b{_MON}\s+{_DAYN}(?:,?\s*{_YYYY})?(?!\w)"), DATE_MASK),  # March 5, 2024; Sept. 5 2023; March 5
    (re.compile(rf"\b{_MON},?\s+{_YYYY}\b"), DATE_MASK),  # March 2024
    # a bare month/day after a date preposition ("as of 3/14"); plain fractions ("1/3 of sales") stay
    (re.compile(r"(?i)\b(on|as of|since|through|thru|until|dated|ending|ended)(\s+)\d{1,2}/\d{1,2}\b"), r"\1\2" + DATE_MASK),
    # a bare month/day as an earnings date (options alerts: "Earnings 9/19 Before Open [est]"; "Reports 8/8 AH") or
    # after a weekday ("Thursday's (6/3)")
    (re.compile(r"(?i)\b(earnings(?:\s+date)?:?)(\s+)\d{1,2}/\d{1,2}\b(?!/)"), r"\1\2" + DATE_MASK),
    (re.compile(r"(?<![\d/.])\b\d{1,2}/\d{1,2}(?=\s+(?:\[est\]\s+)?(?:Before (?:the )?(?:Market )?Open|"
                r"After (?:the )?(?:Market )?Close|BMO|AMC|AH|Pre-?[Mm]arket|After[- ][Hh]ours)\b)"), DATE_MASK),
    (re.compile(r"(?i)\b((?:mon|tues|wednes|thurs|fri|satur|sun)day(?:'s)?)(\s*[,(]?\s*)\d{1,2}/\d{1,2}\b(?!/)"),
     r"\1\2" + DATE_MASK),
    # a month with an abbreviated year ("Until Feb. '21")
    (re.compile(rf"\b{_MON},?\s+'\d{{2}}\b(?!')"), DATE_MASK),
    # v4 (DEV #2 audit, 10-K tables): a month and a year joined by a hyphen ("July-2022") or numerically ("4/2017",
    # "12/2019": KKR's fund tables)
    (re.compile(rf"\b{_MON}-{_YYYY}\b"), DATE_MASK),
    (re.compile(rf"\b{_MON}\s+of\s+{_YYYY}\b"), DATE_MASK),  # "incorporated in December of 1925" (SPGI's 10-K)
    (re.compile(r"(?<![\d/.,$])\b(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b(?![/\d])"), DATE_MASK),
)
# quarter, fiscal-year, half-year and calendar-year labels, with an optional analyst "E" (estimate) and, for fiscal
# and calendar years, an optional range ("FY2023-24", "CY19-21E"); half-years need a separator after "H1"/"H2"
# ("H2'20", "H1 2021") so that product names ("H100") never match
_QTR = re.compile(r"\b(Q[1-4]|[1-4]Q)\s*'?\s*((?:19|20)?\d{2})(E)?\b", re.I)
_RANGE_TAIL = r"(?:[-–/]'?((?:19|20)?\d{2})(E)?)?\b(?!%|[.,]\d)"
_FY = re.compile(r"\b(FY|fiscal(?:\s+year)?)\s*'?\s*((?:19|20)?\d{2})(E)?" + _RANGE_TAIL, re.I)
_HALF = re.compile(r"\b(?:(H[12])(?:\s*'\s*|\s+)|([12]H)\s*'?\s*)((?:19|20)?\d{2})(E)?\b(?!%|[.,]\d)")
_CY = re.compile(r"\b(CY)\s*'?\s*((?:19|20)?\d{2})(E)?" + _RANGE_TAIL)
# abbreviated years 2000-2039 ("By '25", "Late '20 To Early '21", "'19/'20 Wheat", "'20 EBITDA"), only after a year
# preposition or position word, in a pair, or before an accounting noun; a quote that starts with a number ("'23
# injured") or an out-of-range year ("'60 Minutes", "X-Men '97") stays
_APOS_END = r"(?![\w'%]|[.,]\d)"
_APOS_PAIR = re.compile(r"(?<![\w'])'([0-3]\d)\s*/\s*'?([0-3]\d)" + _APOS_END)
_APOS_CUE = re.compile(r"(?i)\b(in|into|by|beyond|since|until|till|from|to|through|during|before|after|early|late|mid|"
                       r"end|of|for|summer|spring|fall|autumn|winter|sees?|expects?)(\s+)'([0-3]\d)" + _APOS_END)
_APOS_NOUN = re.compile(r"(?<![\w'])'([0-3]\d)(?=\s+(?:EBITDA|EBIT|EPS|P/E|PE|revenues?|sales|GDP|guidance|"
                        r"estimates?|outlook|budget|capex|FCF)\b)", re.I)
_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")  # a calendar year (level masks leave these to blind_time)
# mid-2023, pre-2014, post-2020 (v4: pre/post, DEV #2 audit: "pre-2014 PUDs" stayed in the clear)
_YEAR_PREFIXED = re.compile(r"(?i)\b(mid|early|late|end|pre|post)-(19[89]\d|20[0-3]\d)\b(?![.,:/-]?\d)")
_YEAR_RANGE = re.compile(r"(?<![\d.,$/:-])\b(19[89]\d|20[0-3]\d)\s*[-–/]\s*((?:19|20)?\d{2})\b(?![.,:/-]?\d)")
_YEAR_TOKEN = re.compile(r"(?<![\d.,$/:-])\b(19[89]\d|20[0-3]\d)\b(?![.,:/-]?\d)")
# a bare 1980-2039 is a year unless it counts something ("cut 2000 jobs") and no year preposition precedes it
_YEAR_CUE = re.compile(r"(?i)\b(?:in|since|by|of|for|from|to|through|until|till|during|before|after|early|late|mid|"
                       r"end|fiscal|calendar|year|summer|spring|fall|autumn|winter|q[1-4]|h[12])[\s-]*$")
_COUNT_AFTER = re.compile(
    r"\s*(?:jobs|employees|workers|people|staff|positions|roles|layoffs|stores|shops|locations|restaurants|branches|"
    r"outlets|units|vehicles|cars|trucks|planes|aircraft|jets|ships|homes|houses|apartments|rigs|wells|patients|"
    r"customers|clients|users|subscribers|members|accounts|shares|contracts|points|pts|bps|basis|tons|tonnes|"
    r"barrels|bbl|megawatts|MW|GW|acres|miles|kilometers|km|feet|sq|square|beds|seats|flights|routes|machines|"
    r"servers|pieces|items|patents|lawsuits|transactions|orders|times|x|%|percent)(?![A-Za-z])")
# text-rules v4 (DEV #2 audit, the DE 10-K crash): a year next to another year is a year whatever follows it. A
# table header "2023 2022 % Change" (or "2023\n2022\n% Change") has a year before "%"; v3 read "2022%" as a count,
# left it, and at decision year 2023 the rewritten "this year" before it became a year cue: the envelope refused the
# F-A state and the run aborted (at decision year 2024 "1 year ago 2022%" passed: a silent leak). The neighbour is a
# 1980-2039 year token or a relative year this module writes ("this year", "2 years ago", "(1 year ahead)").
# The neighbour may carry a "%" (a header "2017% 2016%") or be joined by "and", "or", "to", "through" ("1 year ago and
# 2019 items"); "2017% of total" is a year column header too.
_YEAR_SEQ_BEFORE = re.compile(r"(?:(?<![\d.,$/:-])\b(?:19[89]\d|20[0-3]\d)%?|\bthis year|\byears? (?:ago|ahead))\)?"
                              r"(?:\s+(?:and|or|to|through|versus|vs\.?))?\s+$")
_YEAR_SEQ_AFTER = re.compile(r"%?\s+(?:(?:and|or|to|through|versus|vs\.?)\s+)?(?:19[89]\d|20[0-3]\d)\b(?![.,:/-]?\d)"
                             r"|%\s+of\s+total\b")
# residues of pre-fix blinding ("1 year ago-01-03", "01/03/1 year ago"): never produced now, still refused
_RESIDUE = re.compile(r"(?:ago|ahead|this year)-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/(?:this year|\d+ years? (?:ago|ahead))")

# text-rules v3 (docs/impl/dev2_blind.md §1). A year in the name of a statute or agreement ("Securities Exchange Act of
# 1934", "the 2017 Tax Act") names the law, not a time of the text; as an offset from the review ("Act of 4 years
# ago") it would pin the decision year, so it becomes [date] whatever its value
_LAW = r"(?:Acts?|Law|Code|Treaty|Accords?|Agreement|Directive|Regulations?|Convention|Protocol|Amendments?|Bill)"
_LAW_YEAR_AFTER = re.compile(rf"(\b{_LAW}\s+of\s+)(?:18|19|20)\d{{2}}\b(?![.,:/-]?\d)")
_LAW_YEAR_BEFORE = re.compile(r"(?<![\d.,$/:-])\b((?:18|19|20)\d{2})(?=(?:\s+[A-Z][\w&.'’-]*){0,4}\s+" + _LAW +
                              r"\b)")
# v4: the year of an EU-style statute number ("Regulation (EU) 2016/679", "Regulation EU 2017/745", "Directive
# 2014/65/EU" -> "[date]/679"; "Regulation (EU) No 536/2014" -> "No 536/[date]") and a standard's edition year
# ("ISO 14001:2015", "ISO/IEC 27001:2013", '("ISO") 13485:2016', "14064-1:2018(E)" -> "14001:[date]")
_LAW_NUMBER = re.compile(rf"(\b{_LAW}\s+(?:\(?(?:EU|EC|EEC)\)?\s+)?(?:No\.?\s+)?)(?:18|19|20)\d{{2}}(?=/\d)"
                         r"|(\b(?:No\.?|EC|EU|EEC)\s+\d{1,4}/)(?:19|20)\d{2}\b(?![.,:/-]?\d)"
                         r"|((?<![\w:.,/-])(?:[A-Z]{1,4})?\d{3,5}(?:-\d{1,2})?[®™]?:)(?:19|20)\d{2}\b(?![.,:/-]?\d)"
                         # a deal or trust series named by its year ("Mortgage Loan Trust 2006-14SL", "STACK 2006-1")
                         # (a single digit or digits with letters: "2023-24" stays a range of years)
                         r"|((?<![\w.,$/:-]))(?:19[89]\d|20[0-3]\d)(?=-(?:\d[A-Z]{0,4}|\d{2}[A-Z]{1,4})\b(?![.,:/-]?\d))")
# years outside the exact-offset range 1980-2039: 1800-1979 (founding and incorporation years, old events) and
# 2040-2099 (long-dated targets, licences, patents). As exact offsets they would still anchor the text ("founded 46
# years ago" = a known founding year), so they become a coarse relative phrase, and only in a year-like context: after
# a time cue, after a determiner and before a word that is not a count ("a 1973 job application"), in a year range, or
# as a decade ("the 1970s"). Counts ("1850 employees"), amounts ("$1,954"), decimals and bare numbers stay.
_FAR = r"(?:18\d{2}|19[0-7]\d|20[4-9]\d)"
FAR_PAST, FAR_AHEAD = "decades ago", "more than a decade ahead"
_FAR_TOKEN = re.compile(rf"(?<![\d.,$/:#-])\b({_FAR})\b(?![.,:/-]?\d)(?!%)")
_FAR_CUE = re.compile(
    r"(?i)(?:\b(?:in|since|by|from|to|through|thru|until|till|during|before|after|circa|ca\.|c\.|around|early|late|"
    r"mid|year|years|fiscal|calendar|founded|established|incorporated|organized|organised|formed|built|opened|began|"
    r"begun|started|dating|dates|back|as of|between|beyond|due|expir(?:e|es|ed|ing)|matur(?:e|es|ed|ing)|target|"
    r"goal|spring|summer|fall|autumn|winter)|(?:class|summer|spring|fall|autumn|winter|end|beginning|start|year) of"
    r"|(?:\b(?:18|19|20)\d{2}|ago|ahead|this year|\[date\])\s+(?:and|or|to|through|until))[\s,-]*$")
_FAR_DET = re.compile(r"(?i)\b(?:a|an|the|its|his|her|their|our|this|that)\s+$")
_FAR_RANGE = re.compile(r"(?<![\d.,$/:#-])\b((?:18|19|20)\d{2})\s*[-–]\s*((?:18|19|20)\d{2})\b(?![.,:/-]?\d)")
_FAR_DECADE = re.compile(r"(?<![\d.,$/:#-])\b(?:the\s+)?(18\d0|19[0-7]0|20[4-9]0)'?s\b")


def _far(y: int, decision_year: int | None = None) -> str:
    return FAR_PAST if y < 1980 else FAR_AHEAD


def _is_far_year(text: str, m: re.Match) -> bool:
    """A 1800-1979 or 2040-2099 number in a year-like context (see above); never a count or a unit amount."""
    if _COUNT_AFTER.match(text, m.end()):
        return False
    before = text[max(0, m.start() - 32):m.start()]
    if _FAR_CUE.search(before):
        return True
    return bool(_FAR_DET.search(before)) and re.match(r"\s+[A-Za-z]", text[m.end():m.end() + 2]) is not None


def _in_range(y: int) -> bool:
    return 1980 <= y <= 2039


def _is_far_range(m: re.Match) -> bool:
    """"1954-1978", "1975–2020": an ascending span of years with at least one end outside 1980-2039, not a count."""
    a, b = int(m.group(1)), int(m.group(2))
    return a < b <= a + 150 and not (_in_range(a) and _in_range(b)) and not _COUNT_AFTER.match(m.string, m.end())


def _is_year(text: str, m: re.Match) -> bool:
    """A 1980-2039 token is a year unless it counts something ("cut 2000 jobs", "2000%") and neither a year cue nor
    another year (a table header "2023 2022 % Change", v4) is next to it. `blind_time` and `absolute_time` both decide
    with this one test, on the text they see; `blind_time` repeats its year pass until nothing changes, so a year that
    becomes a neighbour's cue once rewritten is rewritten too."""
    before = text[max(0, m.start() - 24):m.start()]
    return (bool(_YEAR_CUE.search(before)) or not _COUNT_AFTER.match(text, m.end())
            or bool(_YEAR_SEQ_BEFORE.search(before)) or bool(_YEAR_SEQ_AFTER.match(text, m.end())))


def _rel(y: int, decision_year: int) -> str:
    k = y - decision_year
    return "this year" if k == 0 else (f"{-k} year{'s' if k != -1 else ''} ago" if k < 0 else f"{k} year{'s' if k != 1 else ''} ahead")


def _yy(s: str, like: int | None = None) -> int:
    v = int(s)
    if v >= 100:
        return v
    return (like // 100) * 100 + v if like is not None else v + 2000


def _span(m: re.Match, first: int, decision_year: int) -> str:
    """"<rel>" or "<rel> to <rel>" for a label's year (group `first`) and optional range end (group `first + 2`), with
    ", estimate" if either carries an analyst "E"."""
    a = _yy(m.group(first))
    out = _rel(a, decision_year)
    if m.re.groups >= first + 2 and m.group(first + 2):
        out += f" to {_rel(_yy(m.group(first + 2), a), decision_year)}"
    est = m.group(first + 1) or (m.group(first + 3) if m.re.groups >= first + 3 else None)
    return out + (", estimate" if est else "")


def blind_time(text: str, decision_year: int) -> str:
    """Absolute dates -> [date] (ISO, numeric, day-first, month-first with or without a comma or year,
    month-year, earnings-date month/day), then quarter, fiscal-year, half-year and calendar-year labels, abbreviated
    years, year ranges and bare years relative to the decision year ("Q1 (this year)", "fiscal year (1 year ago)",
    "H2 (1 year ahead)", "calendar year (this year)", "by 1 year ahead", "2 years ago to this year"). A count that
    looks like a year ("cut 2000 jobs") is left alone. `absolute_time(blind_time(x, y))` is always None."""
    if not text:
        return text
    for pat, repl in _DATE_FORMS:
        text = pat.sub(repl, text)
    # v3: statute years, then years outside 1980-2039 in year-like contexts (ranges, decades, single years)
    text = _LAW_YEAR_AFTER.sub(r"\1" + DATE_MASK, text)
    text = _LAW_YEAR_BEFORE.sub(DATE_MASK, text)
    text = _LAW_NUMBER.sub(lambda m: next(g for g in m.groups() if g is not None) + DATE_MASK, text)
    text = _FAR_RANGE.sub(lambda m: " to ".join(dict.fromkeys(
        _rel(y, decision_year) if _in_range(y) else _far(y) for y in (int(m.group(1)), int(m.group(2)))))
        if _is_far_range(m) else m.group(0), text)
    text = _FAR_DECADE.sub(lambda m: _far(int(m.group(1))), text)
    text = _FAR_TOKEN.sub(lambda m: _far(int(m.group(1))) if _is_far_year(m.string, m) else m.group(0), text)
    text = _QTR.sub(lambda m: f"{m.group(1).upper()} ({_span(m, 2, decision_year)})", text)
    text = _FY.sub(lambda m: f"fiscal year{'s' if m.group(4) else ''} ({_span(m, 2, decision_year)})", text)
    text = _HALF.sub(lambda m: f"{m.group(1) or m.group(2)} ({_span(m, 3, decision_year)})", text)
    text = _CY.sub(lambda m: f"calendar year{'s' if m.group(4) else ''} ({_span(m, 2, decision_year)})", text)
    text = _APOS_PAIR.sub(lambda m: f"{_rel(_yy(m.group(1)), decision_year)}/{_rel(_yy(m.group(2)), decision_year)}",
                          text)
    text = _APOS_CUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_rel(_yy(m.group(3)), decision_year)}", text)
    text = _APOS_NOUN.sub(lambda m: _rel(_yy(m.group(1)), decision_year), text)
    text = _YEAR_PREFIXED.sub(lambda m: f"{m.group(1)} ({_rel(int(m.group(2)), decision_year)})", text)
    text = _YEAR_RANGE.sub(lambda m: f"{_rel(int(m.group(1)), decision_year)} to "
                                     f"{_rel(_yy(m.group(2), int(m.group(1))), decision_year)}", text)
    # Year tokens and far years until nothing changes: a relative year written by one pass can be the cue or the year
    # neighbour of a token the pass left ("2023 2022% Change" at 2023 -> "this year 2022%": "year" is a cue; "2024-24
    # 1979" -> "... year 1979"). Every changing pass removes a 4-digit token (a relative year has at most 3 digits),
    # so the loop ends; `absolute_time` then sees exactly the context the last pass saw (v4, DEV #2 audit).
    while True:
        nxt = _YEAR_TOKEN.sub(lambda m: _rel(int(m.group(1)), decision_year) if _is_year(m.string, m) else m.group(0),
                              text)
        nxt = _FAR_TOKEN.sub(lambda m: _far(int(m.group(1))) if _is_far_year(m.string, m) else m.group(0), nxt)
        if nxt == text:
            return text
        text = nxt


def absolute_time(text: str) -> str | None:
    """The first absolute date or calendar year in `text` (anything `blind_time` would rewrite, plus residues of
    the pre-fix rewriting), or None. The state envelope refuses text for which this is not None (§5.0)."""
    if not text:
        return None
    for pat, _ in _DATE_FORMS:
        m = pat.search(text)
        if m:
            return m.group(0)
    for pat in (_LAW_YEAR_AFTER, _LAW_YEAR_BEFORE, _LAW_NUMBER, _FAR_DECADE):
        m = pat.search(text)
        if m:
            return m.group(0)
    for m in _FAR_RANGE.finditer(text):
        if _is_far_range(m):
            return m.group(0)
    for m in _FAR_TOKEN.finditer(text):
        if _is_far_year(text, m):
            return m.group(0)
    for pat in (_QTR, _FY, _HALF, _CY, _APOS_PAIR, _APOS_CUE, _APOS_NOUN, _YEAR_PREFIXED, _YEAR_RANGE, _RESIDUE):
        m = pat.search(text)
        if m:
            return m.group(0)
    for m in _YEAR_TOKEN.finditer(text):
        if _is_year(text, m):
            return m.group(0)
    return None


# ---------------------------------------------------------------------------------- level masks
_INDEX_NAME = re.compile(
    r"\b(?:S&P\s?500|S&P|SPX|Dow Jones Industrial Average|Dow Jones|DJIA|Dow|Nasdaq[-\s]100|Nasdaq Composite|Nasdaq|"
    r"NDX|Russell\s?2000|Russell\s?1000|Russell|NYSE Composite|VIX)(?![A-Za-z])"
)
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{3,}(?:\.\d+)?|\d+\.\d+)(?![\w,]*\d)")
_CHANGE_UNIT = re.compile(r"\s*(?:%|percent\b|per cent\b|points?\b|pts?\b|bps\b|basis points\b|-point\b)", re.I)
_DOLLAR = re.compile(r"\$\s?(\d(?:[\d,]*\d)?(?:\.\d+)?)(?!\s*(?:million|billion|trillion|bn|mln|mn|m|b|k)\b)(?![\d,.]*\d)", re.I)
_PRICE_CUE = re.compile(
    r"(?i)\b(?:shares?|stock|price target|PT|target price|traded|trading|trades|closed|closing|opened|opening|"
    r"priced|last sale|hit|reached|fell to|rose to|move to|52-week (?:high|low)|all-time (?:high|low))\b"
    r"|Company-[0-9A-F]{8}"
)
# a price right after "@" or "Ref=" (options alerts: "500 @ $1.23 vs 86 OI; Ref=$264.21")
_PRICE_CUE_NEXT = re.compile(r"(?:@|\b[Rr]ef\s*=)\s*$")
# a price named by what follows it, possibly as a list: "$205 Price Target", "$427.87 price target", "$101 Target On",
# option strikes "$270 Calls", "$52, $54 Calls"
_PRICE_AFTER = re.compile(r"(?:\s*(?:,|/|-|and|or)?\s*\$\s?\d(?:[\d,]*\d)?(?:\.\d+)?)*\s*"
                          r"(?:price targets?|target prices?|PT|targets?|strikes?|calls?|puts?)\b", re.I)
_SENTENCE_END = re.compile(r"[.!?;]\s")


def mask_levels(text: str) -> str:
    """Mask stock-index levels after an index name ("S&P 500 closed at 5,432" -> "[level]") and share prices
    after a price cue ("shares closed at $123.45", "price target to $250", "Ref=$16.13", "@ $1.23") or before a price
    noun ("$205 Price Target", "$270 Calls"). Changes ("up 12 points", "1.2%") and dollar magnitudes ("$24.1 billion")
    are kept."""
    if not text:
        return text
    spans: list[tuple[int, int]] = []
    for m in _INDEX_NAME.finditer(text):
        end_m = _SENTENCE_END.search(text, m.end())
        stop = min(m.end() + 60, end_m.start() + 1 if end_m else len(text))
        for n in _NUMBER.finditer(text, m.end(), stop):
            if _CHANGE_UNIT.match(text, n.end()) or text[n.start() - 1:n.start()] in ("-", "$"):
                continue
            if _YEAR.fullmatch(n.group(0)):  # a calendar year is time, handled by blind_time
                continue
            spans.append(n.span())
    for m in _DOLLAR.finditer(text):
        start_s = max(0, m.start() - 80)
        before = text[start_s:m.start()]
        last_end = None
        for e in _SENTENCE_END.finditer(before):
            last_end = e.end()
        window = before[last_end:] if last_end is not None else before
        if _PRICE_CUE.search(window) or _PRICE_CUE_NEXT.search(window) or _PRICE_AFTER.match(text, m.end()):
            spans.append(m.span())
    if not spans:
        return text
    out, pos = [], 0
    for a, b in sorted(set(spans)):
        if a < pos:
            continue
        out.append(text[pos:a])
        out.append(LEVEL_MASK)
        pos = b
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------------- alias discovery
DISCOVERY_RULE = "pit-alias-v3"

# Stoplists of the discovery rule (fixed a priori, docs/impl/day3_blind.md §2). They can only reject a candidate,
# never create an alias, so a string they list stays as it is in every state.
# Given names: a candidate starting with one is a person ("Tim Cook", "Joe Biden"), and a single token that
# mostly follows one or a title is a surname ("Musk", "Powell"). People are never aliases of a firm.
GIVEN_NAMES = frozenset("""
aaron abby abigail adam adrian adrienne aidan al alan albert alberto alec alex alexander alexandra alexis alfred
ali alice alicia alison allan allen allison alvin amanda amber amy ana andrea andreas andrew andy angela angelo
anita ann anna anne annie anthony antonio april archie arnold art arthur arvind ash ashley audrey austin ava
barack barbara barry bart ben benjamin bernard bernie beth betsy betty beverly bill billy blake bob bobby bonnie
boris brad bradley brandon brenda brendan brent brett brian bridget brittany brooke bruce bryan bud byron
caitlin calvin cameron candace cara carl carla carlos carmen carol caroline carolyn carrie casey cathie cathy
catherine cecilia chad charlene charles charlie charlotte chelsea cheryl chris christian christina christine
christopher chuck cindy claire clara clarence claudia clay clayton cliff clifford clint clyde cody colin
connie connor craig cristiano curtis cynthia dale dan dana daniel danielle danny dara darius darren dave david
dawn dean deb debbie deborah debra denise dennis derek diana diane dick dina dominic don donald donna doris
doug douglas duane dustin dwight earl ed eddie edgar edith edward edwin eileen elaine eli elizabeth ella ellen
elliott elon emily emma emmanuel eric erica erik erin ernest ethan eugene eva evan evelyn faye felicia felix
fernando florence floyd frances francis frank franklin fred frederick gabriel gail gary gavin gene george
gerald gil gilbert gina ginni gladys glen glenn gordon grace graham grant greg gregory gwen hal hank hannah
harold harry harvey heather hector heidi helen henry herbert herman hillary holly howard hugh hunter ian igor
ira irene iris irving isaac isabel ivan jack jackie jacob jacqueline jake james jamie jan jane janet janice jared
jason jay jean jeanne jeff jeffrey jen jenna jennifer jenny jensen jerome jerry jesse jessica jill jim jimmy jo
joan joanna joanne joe joel john johnny jon jonathan jordan jorge jose joseph josh joshua joyce juan judith judy
julia julian julie justin kamala karen karl kate katherine kathleen kathryn kathy katie katy kay keith kelly
ken kenneth kent kerry kevin kim kimberly kirk kristen kristin kurt kyle lance larry laura lauren laurence
lawrence leah lee leo leon leonard leslie lewis lily linda lindsay lindsey lisa liz lloyd logan lois lori
lorraine louis louise lucas lucy luis luke lynn madeline mandy marc marcus margaret maria marie marilyn mario
marion mark marsha marshall martha martin marty marvin mary mason matt matthew maureen max maxine megan melanie
melissa melvin meredith mia michael michele michelle miguel mike mildred miles milton mitch mitchell molly
monica morris muhammad mustafa myron nancy naomi natalie nathan nathaniel neal neil nelson nicholas nick nicole
nina noah norma norman oliver olivia omar oscar owen pam pamela pat patricia patrick patty paul paula pauline
peggy penny pete peter phil philip phillip phyllis pierre rachel ralph ramon randall randy ray raymond rebecca
reed regina reggie renee rex rhonda ricardo rich richard rick ricky rita rob robert roberta robin rod rodney
roger roland ron ronald ronnie rosa rose ross roy ruby russ russell ruth ryan sally sam samantha samuel sandra
sandy sara sarah satya scott sean sergey seth shannon sharon shawn sheila shelby shelley sheryl shirley sid
sidney simon sophia stacy stan stanley stephanie stephen steve steven stuart sue sundar susan suzanne sylvia
tamara tammy tanya tara ted teresa terrence terry thelma theodore theresa thomas tiffany tim timothy tina todd
tom tommy toni tony tracy travis trevor troy tyler valerie vanessa vera veronica vicki victor victoria vincent
virginia vivian vlad vladimir volodymyr wade walt walter wanda warren wayne wendy wesley whitney will william
willie wilson xi yolanda yvonne zach zachary zoe
abhishek ajay akio amit anand andre anil antony ashok chen dario demis fumio gautam giorgia hock hiroshi indra
jen-hsun jinping jun kazuo keir kenichiro kiyoshi lip-bu masayoshi mukesh narendra nikesh olaf pony rishi safra
sanjay shantanu shinzo shou tadashi takeshi ugur vikram vivek xavier yoshihiro yuri
""".split())

# words that introduce a person's name ("CEO Musk", "Chair Powell", "Designer Jony Ive", "Founder Vitalik")
TITLE_CUES = frozenset("""
ceo cfo coo cto cio chairman chairwoman chair chairperson president vice vp sen senator rep representative
secretary sec gov governor mayor minister premier chancellor mr ms mrs dr judge justice founder co-founder cofounder
analyst analysts investor investors billionaire commissioner director chief leader speaker ambassador boss
executive exec host economist strategist trader manager prince king queen pope sheikh designer former officer
reporter journalist columnist editor leaker tipster influencer professor prof coach entrepreneur tycoon mogul
magnate activist lawmaker congressman congresswoman candidate nominee spokesperson spokesman spokeswoman musician
singer rapper artist actor actress girlfriend boyfriend wife husband
""".split())

# acronyms and category words that are never one firm's name: technologies, product categories, business and
# finance acronyms, regulators, macro terms, indices, index/sector ETFs, crypto assets
GENERIC_TERMS = frozenset("""
gpu gpus cpu cpus ev evs ai agi llm llms api apis saas paas iaas pc pcs oled led leds lcd ssd ssds dram nand hbm
asic asics fpga soc socs av avs adas ota esg eps gaap ebitda ceo cfo coo cto covid covid-19 coronavirus pandemic
omicron sars mrna vaccine vaccines otc nyse nasdaq amex sec fda faa ftc doj fcc cdc nih hhs cms epa irs nhtsa ntsb
cfpb occ fdic fhfa finra cftc fomc ecb boj pboc boe snb rba imf wto who opec opec+ nato un eu uk us usa gdp cpi
ppi pce nfp ism pmi qe qt ipo ipos spac spacs etf etfs nft nfts dao defi web3 metaverse 5g 4g 6g lte wifi wi-fi
bluetooth usb hdmi ar vr xr mr iot ml nlp 3d 2d cad chatbot chatbots robotaxi robotaxis cybersecurity
semiconductor semiconductors chipmaker chipmakers automaker automakers carmaker carmakers streaming smartphone
smartphones tablet tablets laptop laptops wearables wearable drone drones satellite satellites hydrogen lithium
nickel cobalt uranium copper steel aluminum solar wind nuclear biotech fintech insurtech proptech edtech healthtech
medtech telehealth ecommerce e-commerce m&a r&d q&a pre-ipo pre-market premarket after-hours afterhours midday
intraday ytd yoy qoq mom wow tbd ttm bps ppp arpu arr mau maus dau daus gmv aum nav
dtc d2c b2b b2c cob cagr capex opex fcf roi roe roa roic ebit sga cogs kpi kpis okr dei hr ir erp cx ux ui gtm tam
mou loi nda ip jv jvs ciso cmo chro evp svp md bod agm egm faq eta fyi aka tba ltm ntm oem oems odm ems sku skus asp
asps mrr nrr ltv cac cpm cpc ctr roas r&d pt pts ath atl
equal-weight reiterates reiterated maintains maintained initiates initiated resumes assumes raises lowers lifts
boosts trims cuts perform outperformer in-line accumulate reduce sector-perform market-perform peer-perform
spx spy qqq dia iwm vix vxx tqqq sqqq soxl soxs soxx smh xlk xlf xle xlv xli xly xlp xlu xlb xlre xlc arkk arkg
arkw arkq uvxy gld slv uso ung tlt hyg lqd ief shy emb eem efa vti voo ivv vgt vug schd jepi
bitcoin btc ethereum ether eth dogecoin doge xrp ripple solana sol cardano ada litecoin ltc shiba inu tether usdt
usdc binance bnb polygon matic avalanche avax polkadot dot chainlink stablecoin stablecoins memecoin memecoins
crypto cryptos cryptocurrency cryptocurrencies blockchain token tokens altcoin altcoins
dow djia s&p s&p500 nasdaq-100 russell ftse dax cac stoxx nikkei sensex kospi topix hang seng msci
faang fang maang mamaa magnificent mag7
""".split())

# demonyms and nationality adjectives ("Ethiopian" crash, "Swiss" regulator)
DEMONYMS = frozenset("""
british chinese european japanese german french indian korean russian ukrainian israeli saudi canadian mexican
brazilian italian spanish swiss dutch swedish danish irish australian taiwanese vietnamese thai turkish iranian
iraqi egyptian ethiopian nigerian african asian latin arab arabian emirati qatari kuwaiti polish greek finnish
belgian austrian singaporean malaysian indonesian filipino pakistani bangladeshi argentine argentinian chilean
colombian peruvian venezuelan scottish welsh english hispanic nordic scandinavian baltic balkan persian
""".split())

# single tokens that make a candidate an official, institution, union or place: unambiguous surnames of
# government and central-bank officials, institutions, countries and territories, US states, regions and basins,
# large and headquarters cities
STOP_TOKENS = frozenset("""
trump biden obama harris pence vance clinton powell yellen brainard waller bernanke greenspan mnuchin bessent
raimondo lutnick lighthizer gensler pelosi schumer mcconnell mccarthy ocasio-cortez desantis newsom cuomo hochul
putin zelensky zelenskyy sunak starmer macron merkel scholz trudeau modi netanyahu lagarde draghi kuroda ueda
vestager erdogan bolsonaro milei sheinbaum fauci buttigieg blinken pompeo hegseth kudlow navarro jinping kashkari
bostic mester bullard clarida quarles harker goolsbee barkin kanter garland
fed feds fomc treasury treasuries congress senate senators congressional democrats democrat democratic
republicans republican gop capitol pentagon kremlin politburo parliament whitehouse doj ftc fcc faa fda cdc nih hhs
cms epa irs nhtsa ntsb cfpb occ fdic fhfa finra cftc usda doe dot dod ustr cfius nlrb osha nasa ecb boj pboc boe
imf wto opec nato un eu sec uaw teamsters
afghanistan albania algeria andorra angola antigua argentina armenia australia austria azerbaijan bahamas bahrain
bangladesh barbados belarus belgium belize benin bhutan bolivia bosnia botswana brazil brunei bulgaria burkina
burundi cambodia cameroon canada chad chile china colombia comoros congo croatia cuba cyprus czech czechia denmark
djibouti dominica ecuador egypt eritrea estonia eswatini ethiopia fiji finland france gabon gambia georgia germany
ghana greece grenada guatemala guinea guyana haiti honduras hungary iceland india indonesia iran iraq ireland israel
italy jamaica japan jordan kazakhstan kenya kiribati korea kosovo kuwait kyrgyzstan laos latvia lebanon lesotho
liberia libya liechtenstein lithuania luxembourg madagascar malawi malaysia maldives mali malta mauritania mauritius
mexico micronesia moldova monaco mongolia montenegro morocco mozambique myanmar namibia nauru nepal netherlands
nicaragua niger nigeria macedonia norway oman pakistan palau palestine panama paraguay peru philippines poland
portugal qatar romania russia rwanda samoa senegal serbia seychelles singapore slovakia slovenia somalia spain
sudan suriname sweden switzerland syria taiwan tajikistan tanzania thailand togo tonga trinidad tunisia turkey
turkmenistan tuvalu uganda ukraine uruguay uzbekistan vanuatu vatican venezuela vietnam yemen zambia zimbabwe
britain england scotland wales europe asia africa antarctica oceania eurasia bermuda cayman greenland gaza crimea
donbas kashmir tibet xinjiang hokkaido bavaria catalonia quebec ontario alberta siberia patagonia scandinavia
balkans caribbean mediterranean permian bakken appalachia stabroek
alabama alaska arizona arkansas california colorado connecticut delaware florida hawaii idaho illinois indiana iowa
kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska
nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia wisconsin wyoming
beijing shanghai shenzhen guangzhou hangzhou wuhan chengdu tianjin zhengzhou nanjing suzhou taipei tokyo osaka
seoul mumbai bangalore bengaluru hyderabad chennai delhi dubai riyadh doha moscow kyiv kiev london paris berlin
frankfurt munich stuttgart wolfsburg brussels amsterdam zurich geneva stockholm oslo copenhagen helsinki dublin
madrid barcelona rome milan vienna warsaw prague budapest istanbul cairo lagos nairobi johannesburg toronto
vancouver montreal ottawa sydney melbourne jakarta manila bangkok hanoi seattle chicago boston austin dallas houston
atlanta miami denver phoenix detroit philadelphia baltimore pittsburgh cleveland cincinnati nashville charlotte
portland minneapolis omaha cupertino redmond fremont sacramento oakland brooklyn manhattan hollywood bentonville
armonk berkeley stanford harvard princeton yale oxford cambridge davos grunheide gruenheide
""".split())

# whole candidates (alias keys) that are never a study firm's name: officials and commentators, institutions,
# programmes, places, holidays and events, media and data providers, brokers and research firms, unions, and
# well-known companies that are not SEC registrants (the registrant list covers the others)
STOP_PHRASES = frozenset(p.strip() for p in """
jackson hole|donald trump|joe biden|barack obama|kamala harris|mike pence|jd vance|hillary clinton|bill clinton|
george bush|jerome powell|janet yellen|lael brainard|christopher waller|michelle bowman|philip jefferson|lisa cook|
john williams|neel kashkari|raphael bostic|mary daly|loretta mester|james bullard|richard clarida|randal quarles|
patrick harker|austan goolsbee|thomas barkin|susan collins|lorie logan|kevin warsh|kevin hassett|stephen miran|
ben bernanke|alan greenspan|steven mnuchin|scott bessent|wilbur ross|gina raimondo|howard lutnick|
robert lighthizer|katherine tai|jamieson greer|jay clayton|gary gensler|paul atkins|lina khan|andrew ferguson|
merrick garland|william barr|pam bondi|jonathan kanter|nancy pelosi|chuck schumer|mitch mcconnell|kevin mccarthy|
mike johnson|elizabeth warren|bernie sanders|josh hawley|ted cruz|marco rubio|alexandria ocasio-cortez|ron wyden|
sherrod brown|maxine waters|pat toomey|tim scott|rand paul|marjorie taylor greene|ro khanna|gavin newsom|
ron desantis|greg abbott|andrew cuomo|kathy hochul|xi jinping|vladimir putin|volodymyr zelensky|boris johnson|
rishi sunak|keir starmer|emmanuel macron|angela merkel|olaf scholz|justin trudeau|mark carney|narendra modi|
benjamin netanyahu|christine lagarde|mario draghi|haruhiko kuroda|kazuo ueda|andrew bailey|ursula von der leyen|
margrethe vestager|thierry breton|kim jong un|mohammed bin salman|recep tayyip erdogan|jair bolsonaro|javier milei|
claudia sheinbaum|anthony fauci|pete buttigieg|elaine chao|antony blinken|mike pompeo|pete hegseth|lloyd austin|
mark esper|jake sullivan|larry kudlow|peter navarro|kevin o'leary|jim cramer|cathie wood|dan ives|gene munster|
katy huberty|toni sacconaghi|adam jonas|gary black|ross gerber|michael burry|bill ackman|carl icahn|ken griffin|
ray dalio|chamath palihapitiya|dan nathan|jon najarian|pete najarian|tom lee|josh brown|troy teslike|
sawyer merritt|pierre ferragu|david einhorn|stanley druckenmiller|david tepper|nelson peltz|paul tudor jones|
jeremy siegel|mohamed el-erian|larry summers|nouriel roubini|peter schiff|kevin paffrath|meet kevin|
federal reserve|white house|supreme court|justice department|commerce department|state department|
labor department|defense department|treasury department|energy department|wall street|main street|capitol hill|
silicon valley|european commission|european union|european parliament|bank of japan|bank of england|
people's bank|world bank|federal trade commission|fast money|fast money halftime|halftime report|mad money|
squawk box|power lunch|closing bell|wall street journal|financial times|new york times|washington post|
yahoo finance|seeking alpha|motley fool|business insider|fox business|the verge|the information|
new york|washington|los angeles|san francisco|hong kong|mexico city|sao paulo|tel aviv|las vegas|kuala lumpur|
ho chi minh|new delhi|abu dhabi|san jose|santa clara|palo alto|mountain view|menlo park|redwood city|san diego|
salt lake city|new jersey|north carolina|south carolina|north dakota|south dakota|west virginia|new hampshire|
new mexico|rhode island|puerto rico|united kingdom|united states|saudi arabia|south korea|north korea|
south africa|new zealand|latin america|middle east|south china sea|gulf coast|
super bowl|black friday|cyber monday|prime day|world cup|olympics|christmas|thanksgiving|halloween|easter|
omicron variant|jedi|
uaw|united auto workers|teamsters|afl-cio|sag-aftra
""".replace("\n", "").split("|") if p.strip())

# companies (and their media and research arms) that are not SEC registrants, plus the media and brokers most
# quoted in the feed: never a study firm's alias, and an item naming one counts towards the "names another
# company" rate of a candidate (SpaceX's Starship in items tagged to Tesla)
NON_SEC_COMPANIES = frozenset(p.strip() for p in """
openai|chatgpt|gpt-4|gpt-4o|gpt-5|anthropic|xai|grok|spacex|space x|starlink|neuralink|boring company|tiktok|
bytedance|huawei|xiaomi|byd|catl|foxconn|hon hai|samsung|samsung electronics|tsmc|sk hynix|hynix|lg|lg energy|
tencent|shein|temu|volkswagen|vw|bmw|mercedes|mercedes-benz|daimler|porsche|audi|toyota|nissan|hyundai|kia|renault|
geely|great wall|airbus|bombardier|siemens|bosch|roche|bayer|novo|novo nordisk|aramco|saudi aramco|gazprom|rosneft|
glencore|nestle|lvmh|hermes|kering|adidas|puma|uniqlo|zara|inditex|ikea|aldi|lidl|tesco|deliveroo|
essilorluxottica|luxottica|mistral|deepmind|stripe|klarna|revolut|epic games|valve|ubisoft|tata|reliance|infosys|
telegram|discord|wechat|kakao|nintendo|playstation|fortnite|juul|juul labs|ozempic|wegovy|glj|glj research|
bloomberg|reuters|cnbc|cnn|bbc|benzinga|barron|barron's|wsj|politico|axios|nikkei|caixin|xinhua|yonhap|electrek|
teslarati|macrumors|appleinsider|9to5mac|9to5google|techcrunch|engadget|wired|zdnet|cnet|theinformation|
stocktwits|swaggy|swaggystocks|wallstreetbets|reddit|
wedbush|needham|mizuho|jefferies|piper sandler|raymond james|evercore|oppenheimer|stifel|cowen|bernstein|keybanc|
baird|loop capital|rosenblatt|canaccord|macquarie|nomura|daiwa|clsa|btig|benchmark|guggenheim|ubs|barclays|hsbc|
deutsche|deutsche bank|credit suisse|morningstar|zacks|argus|cfra|tigress|wolfe|wolfe research|moffettnathanson|
redburn|berenberg|exane|susquehanna|da davidson|new street|new street research|atlantic equities|bmo|rbc|td cowen|
truist securities|citi research|bofa securities|ark|ark invest
""".replace("\n", "").split("|") if p.strip())
STOP_PHRASES = STOP_PHRASES | NON_SEC_COMPANIES

_MONTH_DAY_WORDS = frozenset(w.casefold() for w in _MONTHS.split("|")) | frozenset(
    "monday tuesday wednesday thursday friday saturday sunday".split())


def surname_eligible(surname: str) -> bool:
    """A surname that may be masked alone (an executive's, `Executive`): at least 3 letters, a single name token (a
    particle prefix such as "de" allowed), and not an ordinary word, given name, title, official, place, demonym,
    month or generic term (so "Price", "Grant", "James", "Powell" are never masked alone)."""
    s = fold_accents(surname).strip()
    if not re.fullmatch(r"(?:(?:van|von|de|del|der|den|da|di|du|la|le|mac|st\.?) )?[A-Za-z][A-Za-z'’\-]{2,}", s, re.I):
        return False
    k = s.casefold()
    last = k.split()[-1]
    return not any(x in bank for x in (k, last) for bank in (
        COMMON_WORDS, WORD_NAMES, GIVEN_NAMES, TITLE_CUES, STOP_TOKENS, DEMONYMS, GENERIC_TERMS, _MONTH_DAY_WORDS,
        SURNAME_WORDS)) and k not in STOP_PHRASES


# English words that are also surnames: never masked alone (their capitalized use in headlines is mostly the word)
SURNAME_WORDS = frozenset("""
black white brown green gray grey young king long short little rich wise sharp strong good best noble bright
page rice cash bond banks gold silver stone wood woods hill hills field fields lane hunt chase grant bell rose fox
west north south east love hope joy church hall house street star sun moon swift reed moss lamb fish burns
walker baker miller turner parker carter cooper mason porter fisher hunter marshall butler taylor
day may march price spring summer winter frost snow rain storm cross close case power powers holiday
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'\-]*")
_CAMEL = re.compile(r"[a-z]{1,4}[A-Z][A-Za-z0-9]*")  # iPhone, eBay, macOS: a name despite the lower-case start
_PLAIN_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'\-]*")
_SENT_START = re.compile(r"[.!?:;\"'(\[—–\-]\s*$")  # the next token starts a sentence or a quote
# after a token: a corporate designator or an exchange-ticker parenthesis ("Deliveroo plc", "iRobot (NASDAQ:IRBT)")
_COMPANY_CUE = re.compile(
    r"(?:'s)?(?:,?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Ltd|Limited|plc|PLC|LLC|LP|AG|SA|SE|NV|N\.V|GmbH|"
    r"Holdings|Group|Technologies|Therapeutics|Biotherapeutics|Pharmaceuticals|Biosciences|Labs|Ventures|Capital|"
    r"Partners|Airlines|Motors|Research|Securities|Advisors|Associates|Investments)\b|\s*\(\s*(?:[A-Z]{2,6}\s*:\s*)?[A-Z]{1,5}(?:\.[A-Z]{1,2})?\s*\))")
_PERSON_CUES = GIVEN_NAMES | TITLE_CUES


def _common(w: str) -> bool:
    return w.casefold() in COMMON_WORDS or w.casefold() in WORD_NAMES


def _norm_token(t: str) -> str:
    return re.sub(r"'s$", "", t).rstrip("-'&")


def _is_cap(t: str) -> bool:
    return t[:1].isupper() or _CAMEL.fullmatch(t) is not None


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(token without a possessive or trailing '-&', start, end)."""
    return [(_norm_token(m.group(0)), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _grams(text: str, toks: list[tuple[str, int, int]], max_n: int) -> set[str]:
    out: set[str] = set()
    run: list[str] = []
    last_end = -2

    def flush():
        for i in range(len(run)):
            for n in range(1, max_n + 1):
                if i + n <= len(run):
                    g = run[i:i + n]
                    if _common(g[0]) or _common(g[-1]):
                        continue
                    s = " ".join(g)
                    if len(s) >= 3 and not s.isdigit():
                        out.add(s)

    for tok, a, b in toks:
        if not tok:
            flush()
            run, last_end = [], b
            continue
        cap = _is_cap(tok)
        if cap and run and text[last_end:a] == " ":
            run.append(tok)
        else:
            flush()
            run = [tok] if cap else []
        last_end = b
    flush()
    return out


def candidate_ngrams(text: str, max_n: int = 3) -> set[str]:
    """Capitalized 1..max_n-grams (runs of consecutive capitalized or camel-case tokens separated by single spaces;
    HTML entities decoded, possessives dropped) whose first and last tokens are not common or word-like."""
    text = html.unescape(text or "")
    return _grams(text, _tokens(text), max_n)


def _parts(g: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", g) if t]


@dataclass(frozen=True)
class DiscoveryRule:
    """Thresholds of discovery rule `pit-alias-v3` (docs/impl/day3_blind.md §2). v2's thresholds were set on the dev
    half of the firms (sha256('alias-split-www27' + symbol), even first byte); v3's added mechanisms answer the error
    classes the audit of v2 found on the other half (§3)."""

    min_own: int = 8  # discovery-fold items with the n-gram that carry the firm's identifiers (support)
    min_share: float = 0.8  # co-occurrence with the firm's identifiers in the discovery fold (exclusivity)
    min_heldout: int = 3  # held-out-fold items with the n-gram
    min_heldout_share: float = 0.7  # the same co-occurrence re-measured on the held-out fold (precision check)
    max_rival: float = 0.3  # the n-gram's items carrying another member firm's identifiers
    max_extra: float = 0.5  # the n-gram's items also tagged to a ticker outside the study set (another company)
    max_other_named: float = 0.5  # the n-gram's items naming a company outside the study set (SpaceX, Sony, ...)
    # proper-noun evidence of an edge token: its add-one smoothed lower-case share in sentence-case text,
    # (lower + 1) / (occurrences + 2), is at most max_lower (so >= 8 occurrences, all capitalized, at 0.1)
    max_lower: float = 0.1
    max_person: float = 0.3  # a token mostly after a given name, initials or a title is (part of) a person's name
    max_company: float = 0.1  # the last token followed by a corporate designator or "(TICKER)" is another company
    max_source: float = 0.2  # a token mostly inside a trailing source attribution ("... -Tesmanian") is a medium
    hysteresis: float = 0.1  # an alias accepted at the previous anchor is kept while the share rules hold within this
    # the shares weight each item by 2^(t / half_life): the window still expands (every item before the anchor), but a
    # brand that changed owner (HBO: AT&T until 2022-04, WBD after) follows the recent items; support is unweighted
    half_life_days: float = 365.0
    min_corpus: int = 30_000  # no alias before this many items are counted (corpus statistics are unreliable)
    max_n: int = 3
    fold_salt: str = "alias-fold-www27"


@dataclass(frozen=True)
class OtherRegistrants:
    """SEC registrants outside the study set (`data/raw/sec/company_tickers.json` minus the study CIKs): a candidate
    that is another company's cleaned name, contains the distinctive first word of one, or contains its ticker is
    never a study firm's alias ("Nio", "Alibaba", "Sony", "Rivian", "Mizuho"). The file is a 2026 snapshot; the
    list can only keep a string unmasked, never mask one, so it cannot put future information into a state."""

    keys: frozenset[str] = frozenset()
    first_words: frozenset[str] = frozenset()
    tickers: frozenset[str] = frozenset()

    def __len__(self) -> int:
        return len(self.keys)


def load_other_registrants(sec_dir: Path, study_symbols: Iterable[str]) -> OtherRegistrants:
    """`OtherRegistrants` from `<sec_dir>/company_tickers.json`, excluding the CIKs of `<sec_dir>/cik_map.json`'s
    study symbols (share classes of a study firm are the firm). Empty if the file is missing."""
    sec_dir = Path(sec_dir)
    p = sec_dir / "company_tickers.json"
    if not p.exists():
        return OtherRegistrants()
    study = set(study_symbols)
    rows = json.loads(p.read_text())
    rows = list(rows.values()) if isinstance(rows, dict) else list(rows)
    cik_path = sec_dir / "cik_map.json"
    ciks = {int(v) for k, v in json.loads(cik_path.read_text()).items() if k in study} if cik_path.exists() else set()
    ciks |= {int(r["cik_str"]) for r in rows if r.get("ticker") in study}
    keys, first, tickers = set(), set(), set()
    for r in rows:
        if int(r["cik_str"]) in ciks:
            continue
        tickers.add(str(r["ticker"]))
        c = clean_name(r.get("title") or "")
        if not c:
            continue
        keys.add(alias_key(c))
        w = c.split()[0]
        if len(w) >= 4 and w.isalpha() and w.casefold() not in COMMON_WORDS:
            first.add(w.casefold())
    return OtherRegistrants(frozenset(keys), frozenset(first), frozenset(tickers))


# a trailing source attribution of a headline or summary ("... -Electrek Report", "... - The Verge", "via Bloomberg")
_SOURCE_TAIL = re.compile(r"(?:^|\s|['\"”’])[-–—]{1,2}\s*((?:[A-Z0-9][\w&.'’]*\s?){1,4}?)"
                          r"(?:\s*(?:Report|Reports|Article|Articles|News|Story|Art))?\W*$")
_SOURCE_LEAD = re.compile(r"\b(?:[Aa]ccording [Tt]o|[Vv]ia|[Cc]iting|[Rr]eported [Bb]y)\s+((?:[A-Z0-9][\w&.'’]*\s?){1,4})")
_INITIALS = re.compile(r"(?:[A-Z]\.){1,3}|[A-Z]{2}")  # "JB Straubel", "J.D. Vance"


class _TokenStats:
    """Corpus statistics of tokens (casefolded), accumulated over the counted items' headlines and summaries."""

    def __init__(self, known: frozenset[str] = frozenset()):
        self.known = known  # tickers and names: "GM Ultium" is not a pair of initials and a surname
        self.all: Counter = Counter()
        self.sc: Counter = Counter()  # occurrences inside sentence-case text, not at a sentence start
        self.sc_lower: Counter = Counter()  # ... of which written all lower-case
        self.person: Counter = Counter()  # right after a given name, initials or a title
        self.company: Counter = Counter()  # followed by a corporate designator or "(TICKER)"
        self.source: Counter = Counter()  # inside a source attribution

    def add(self, seg: str) -> None:
        toks = _tokens(seg)
        long = [t for t, _, _ in toks[1:] if sum(c.isalpha() for c in t) >= 4]
        # sentence-case text: >= 3 long tokens, most of them lower-case (a Title-Case headline, or a segment too
        # short to tell, gives no proper-noun evidence)
        sentence_case = len(long) >= 3 and sum(not t.islower() for t in long) / len(long) < 0.6
        prev: tuple[str, str, int] | None = None
        for j, (tok, a, b) in enumerate(toks):
            if not tok:
                prev = None
                continue
            k = tok.casefold()
            self.all[k] += 1
            if prev is not None and seg[prev[2]:a] == " " and (
                    prev[1] in _PERSON_CUES or (_INITIALS.fullmatch(prev[0]) and prev[1] not in self.known)):
                self.person[k] += 1
            if sentence_case and j > 0 and not _SENT_START.search(seg[max(0, a - 3):a]):
                self.sc[k] += 1
                if tok.islower():
                    self.sc_lower[k] += 1
            if _COMPANY_CUE.match(seg, b):
                self.company[k] += 1
            prev = (tok, k, b)
        for m in (_SOURCE_TAIL.search(seg), *_SOURCE_LEAD.finditer(seg)):
            if m is not None:
                for t, _, _ in _tokens(m.group(1)):
                    self.source[t.casefold()] += 1

    def proper(self, w: str, rule: DiscoveryRule) -> bool:
        """Capitalized in sentence-case text (not only in Title-Case headlines): a proper noun, not an ordinary word.
        The lower-case share is add-one smoothed, so thin early evidence ("Surface": 5 of 5 capitalized in 2019, 16 of
        48 lower-case by 2024) does not pass."""
        k = w.casefold()
        return (self.sc_lower[k] + 1) / (self.sc[k] + 2) <= rule.max_lower

    def share(self, counter: Counter, w: str) -> float:
        k = w.casefold()
        return counter[k] / max(1, self.all[k])


class AliasDiscovery:
    """Point-in-time alias discovery on expanding windows (design_v1 §5.0, validity m7), rule `pit-alias-v3`.

    Aliases are firm-specific names only: products, brands, subsidiaries, programmes; never people, places,
    institutions, other companies, media or ordinary words (docs/impl/day3_blind.md). At monthly anchors A (UTC month
    starts), counting only items created and last edited before A (V-m10) and, with `members`, only items tagged
    MARKET or to a firm that is a member at A (the news a real-time collector would hold at A), a capitalized
    1-3-gram g (counted case-insensitively by its alias key, as the blinder masks it) is accepted as an alias of a
    member firm s that is trading at A (`lives`) iff, once >= `min_corpus` items are counted,
    - **support and identifier co-occurrence**: items are split into a discovery and a held-out fold by
      sha256(fold_salt + id). In the discovery fold, >= `min_own` items with g carry s's identifiers (s's ticker as a
      feed tag on an item with <= 3 tags, or one of s's known names or tickers in the text), and they are
      >= `min_share` of all items with g. Every share in the rule weights an item by 2^(t / `half_life_days`), so
      the window expands but follows a brand that changed owner; support counts are unweighted;
    - **held-out precision**: in the held-out fold g has >= `min_heldout` items, and >= `min_heldout_share` of them
      carry s's identifiers;
    - **exclusivity**: no other member firm's identifiers co-occur with g in more than `max_rival` of g's items; at
      most `max_extra` of g's items are also tagged to a ticker outside the study set, and at most `max_other_named`
      name a company outside the study set (another company's news);
    - **shape**: every token is a plain name token; the first and last tokens have >= 3 letters, are not common
      words, given names or titles, and are proper nouns in the corpus (add-one smoothed lower-case share in
      sentence-case text <= 10%, so Title-Case headline words such as "Information" or "Has" fail); no token is a known
      alias of any firm or a proper fragment of a known name ("Fargo", "Depot"); g does not start with a given name;
      no token mostly follows a given name, initials or a title (people: "Musk", "Andrej", "Ming-Chi"), and a single
      token does not mostly follow such a first name (surnames: "Karpathy", "Kuo"); its last token is not mostly
      followed by a corporate designator or ticker ("Deliveroo plc"); no token mostly appears in a source attribution
      ("... -Tesmanian");
    - **stoplists**: g is not, and contains no token from, the lists of officials, institutions, unions, places,
      demonyms, generic terms, indices, crypto assets, media, brokers and non-SEC companies above, and is not another
      SEC registrant's name, first word or ticker (`OtherRegistrants`).
    An alias accepted at the previous anchor is kept while the share rules hold within `hysteresis` (the shape and
    stoplist rules are not relaxed), so masking does not flicker around a threshold, yet an alias whose evidence
    fades (a divested brand) or whose firm stops trading is dropped. The result at A is a pure function of the items
    known before A and of the membership and trading lives at anchors <= A, so a decision at cutoff c, which uses the
    latest anchor <= c, depends on no item created or edited at or after c. `members` must be monotone
    (`programs.context.StudyMembers.asof` is).
    """

    def __init__(
        self,
        items: pd.DataFrame,
        symbols: Iterable[str],
        known: Mapping[str, FirmAliases] | None = None,
        *,
        members: Callable[[pd.Timestamp], Iterable[str]] | None = None,
        other: OtherRegistrants | None = None,
        rule: DiscoveryRule | None = None,
        lives: Mapping[str, tuple] | None = None,
        salt: str = SALT,
    ):
        """`items`: news rows (id, symbol, created_at, updated_at, headline, summary; optional n_tags, the item's
        number of feed tickers), all rows of an item, including tags outside `symbols` (MARKET). `known`: the firms'
        static aliases (names and tickers found in the text count as identifiers; candidates overlapping them are
        refused). `members(anchor)`: the firms that are members at the anchor (None: every symbol, always). `lives`:
        symbol -> (first, last) trading session (a firm gets no alias at an anchor outside its life; None: always)."""
        missing = {"id", "symbol", "created_at", "updated_at", "headline", "summary"} - set(items.columns)
        if missing:
            raise ValueError(f"discovery items need columns {sorted(missing)} (updated_at: V-m10)")
        self.symbols = frozenset(symbols)
        self.rule = rule or DiscoveryRule()
        self.members = members
        self.other = other or OtherRegistrants()
        self.lives = {s: (_utc(a), _utc(b) + pd.Timedelta(days=1)) for s, (a, b) in (lives or {}).items()
                      if _utc(a) is not None and _utc(b) is not None}
        known = {s: _as_firm_aliases(v, s) for s, v in (known or {}).items()}
        # static aliases only: identifiers in the text (firm names and tickers; executives play no part in discovery)
        self._mention = Blinder({s: replace(f, executives=()) for s, f in known.items()}, salt=salt) if known else None
        self._code_of = {v: k for k, v in self._mention.code_names.items()} if self._mention else {}
        keys = {alias_key(a) for f in known.values() for a in (*f.names, *f.word_names)}
        keys |= {t.casefold() for f in known.values() for t in f.tickers}
        self.known = frozenset(keys | {re.sub(r"'s$", "", k) for k in keys})
        spans: set[str] = set()
        for k in keys:
            ws = _parts(k)
            if len(ws) > 1:
                spans |= {" ".join(ws[i:j]) for i in range(len(ws)) for j in range(i + 1, len(ws) + 1)
                          if (i, j) != (0, len(ws))}
        self._spans = frozenset(spans)  # proper fragments of known multi-word names
        self._frag_tokens = frozenset(w for s in spans for w in s.split() if w not in COMMON_WORDS)
        # names of companies outside the study set, for the "names another company" rate
        self._other_named = frozenset((set(self.other.keys) | NON_SEC_COMPANIES) - self.known)
        df = items.copy()
        created = pd.to_datetime(df.created_at, utc=True)
        updated = pd.to_datetime(df.updated_at, utc=True).fillna(created)
        df["known_at"] = created.where(created >= updated, updated)  # max(created_at, updated_at)
        df[["headline", "summary"]] = df[["headline", "summary"]].fillna("")
        if "n_tags" not in df.columns:
            df["n_tags"] = df.groupby("id")["symbol"].transform("size")
        g = df.groupby("id").agg(known_at=("known_at", "max"), tags=("symbol", frozenset), n_tags=("n_tags", "max"),
                                 headline=("headline", "first"), summary=("summary", "first"))
        g = g.sort_values("known_at", kind="stable")
        self._items = list(zip(g.index, g.known_at, g.tags, g.n_tags.astype(int), g.headline, g.summary))
        self._times = [x[1] for x in self._items]
        self._reset()

    # -------------------------------------------------------------- state
    def _reset(self) -> None:
        self._pos = 0  # next item (in time order) not yet looked at
        self._n_counted = 0
        self._counted = bytearray(len(self._items))
        self._pending: dict[str, list[int]] = defaultdict(list)  # items tagged only to firms not yet members
        self._members_seen: frozenset[str] = frozenset()
        self._n = (Counter(), Counter())  # items with key g, per fold (support)
        self._w = (Counter(), Counter())  # the same, recency-weighted (shares)
        self._x: Counter = Counter()  # weighted items with g also tagged to a ticker outside the study set
        self._ox: Counter = Counter()  # weighted items with g naming a company outside the study set
        self._forms: dict[str, Counter] = defaultdict(Counter)  # key -> surface forms (the most frequent is shown)
        # key -> firm -> [fold-0, fold-1 items carrying the firm's identifiers, the same recency-weighted]
        self._by_g: dict[str, dict[str, list[float]]] = defaultdict(dict)
        self._bigrams: dict[str, set[str]] = defaultdict(set)  # last token -> two-token keys (surname evidence)
        self._hot: set[tuple[str, str]] = set()  # (key, s) with fold-0 support >= min_own
        self._stats = _TokenStats(self.known)
        self._static: dict[str, str | None] = {}
        self._current: dict[str, dict[str, tuple[pd.Timestamp, int]]] = {}  # s -> key -> (first anchor, support)
        self._results: dict[pd.Timestamp, dict[str, tuple[str, ...]]] = {}
        self._first: dict[pd.Timestamp, dict[tuple[str, str], pd.Timestamp]] = {}
        self._case_free: dict[pd.Timestamp, frozenset[str]] = {}  # accepted keys with a camel-case form, per anchor
        self._last: pd.Timestamp | None = None

    @staticmethod
    def anchor_for(cutoff) -> pd.Timestamp:
        c = pd.Timestamp(cutoff)
        c = c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
        return pd.Timestamp(year=c.year, month=c.month, day=1, tz="UTC")

    def aliases_asof(self, cutoff) -> dict[str, tuple[str, ...]]:
        return self.aliases_at(self.anchor_for(cutoff))

    def aliases_at(self, anchor: pd.Timestamp) -> dict[str, tuple[str, ...]]:
        """Aliases per member firm at `anchor` (a month start; other times are floored to one), most supported
        first."""
        anchor = self.anchor_for(anchor)
        if anchor in self._results:
            return self._results[anchor]
        if not self._items or anchor <= self.anchor_for(self._times[0]):
            self._first.setdefault(anchor, {})
            self._case_free.setdefault(anchor, frozenset())
            return self._results.setdefault(anchor, {})  # no item is known before the first anchor
        if self._last is not None and anchor < self._last:
            # every month after the first item has been stepped through, so this cannot happen; recount to be safe
            self._reset()
        a = self._last + pd.DateOffset(months=1) if self._last is not None else self.anchor_for(self._times[0])
        while a <= anchor:
            self._step(pd.Timestamp(a))
            a = a + pd.DateOffset(months=1)
        return self._results[anchor]

    def first_seen(self, symbol: str, alias: str, anchor: pd.Timestamp | None = None) -> pd.Timestamp | None:
        """The anchor from which `alias` has been continuously accepted for `symbol` at `anchor` (default: the latest
        stepped anchor); None if it is not accepted there."""
        a = self.anchor_for(anchor) if anchor is not None else self._last
        return self._first.get(a, {}).get((symbol, alias_key(alias))) if a is not None else None

    def case_free_keys(self, anchor) -> frozenset[str]:
        """Alias keys accepted at `anchor` that were counted in a camel-case surface form with a lower-case start
        ("macOS", "iPhone"). The blinder matches them in any case, so the mask does not depend on which form happens
        to be the most frequent at the anchor ("MacOS" in some months, "macOS" in others)."""
        anchor = self.anchor_for(anchor)
        self.aliases_at(anchor)
        return self._case_free.get(anchor, frozenset())

    def describe(self) -> dict:
        return {"rule": DISCOVERY_RULE, **asdict(self.rule), "members": self.members is not None,
                "lives": bool(self.lives), "other_registrants": len(self.other), "n_items": len(self._items)}

    # -------------------------------------------------------------- counting
    def _members_at(self, anchor: pd.Timestamp) -> frozenset[str]:
        return self.symbols if self.members is None else frozenset(self.members(anchor)) & self.symbols

    def _alive(self, s: str, anchor: pd.Timestamp) -> bool:
        life = self.lives.get(s)
        return life is None or life[0] <= anchor < life[1]

    def _step(self, anchor: pd.Timestamp) -> None:
        members = self._members_at(anchor)
        batch: list[int] = []
        for s in sorted(members - self._members_seen):  # new members: their earlier items become collectable
            batch += self._pending.pop(s, [])
        self._members_seen = self._members_seen | members
        stop = bisect.bisect_left(self._times, anchor)  # items known strictly before the anchor
        for idx in range(self._pos, stop):
            tags = self._items[idx][2]
            firms = tags & self.symbols
            if len(firms) < len(tags) or firms & members:  # MARKET (or another non-firm tag) or a member's item
                batch.append(idx)
            else:
                for s in firms:
                    self._pending[s].append(idx)
        self._pos = max(self._pos, stop)
        for idx in sorted(set(batch)):
            if not self._counted[idx]:
                self._counted[idx] = 1
                self._n_counted += 1
                self._count(self._items[idx])
        self._accept(anchor, members)
        self._last = anchor

    def _fold(self, item_id) -> int:
        return hashlib.sha256(f"{self.rule.fold_salt}{item_id}".encode()).digest()[0] & 1

    def _count(self, item) -> None:
        item_id, known_at, tags, n_tags, headline, summary = item
        h, s = html.unescape(headline), html.unescape(summary)
        self._stats.add(h)
        self._stats.add(s)
        text = f"{h}. {s}"
        raw = _grams(text, _tokens(text), self.rule.max_n)
        keys: dict[str, str] = {}
        for g in raw:
            if self._static_check(g) is None:
                keys.setdefault(alias_key(g), g)
        if not keys:
            return
        firms = tags & self.symbols
        named: frozenset[str] = frozenset()
        if self._mention is not None:
            out = self._mention.blind_entities(text, cutoff=known_at, tagged=firms)
            named = frozenset(self._code_of[c] for c in _CODE.findall(out) if c in self._code_of)
        about = (firms if n_tags <= 3 else frozenset()) | named
        f = self._fold(item_id)
        w = self._weight(known_at)
        extra = len(firms) == len(tags) and n_tags > len(tags)  # a firm item also tagged outside the study set
        other_named = any(alias_key(g) in self._other_named for g in raw)
        for k, g in keys.items():
            if self._n[0][k] + self._n[1][k] == 0 and " " in k and k.count(" ") == 1:
                self._bigrams[k.split(" ")[1]].add(k)
            self._n[f][k] += 1
            self._w[f][k] += w
            self._forms[k][g] += 1
            if extra:
                self._x[k] += w
            if other_named:
                self._ox[k] += w
            per = self._by_g[k]
            for sym in about:
                c = per.get(sym)
                if c is None:
                    c = per[sym] = [0, 0, 0.0, 0.0]
                c[f] += 1
                c[2 + f] += w
                if f == 0 and c[0] == self.rule.min_own:
                    self._hot.add((k, sym))

    def _weight(self, t: pd.Timestamp) -> float:
        """2^((t - first item) / half-life): recency weight of an item (only ratios of weights are used)."""
        return 2.0 ** ((t - self._times[0]).total_seconds() / 86400.0 / self.rule.half_life_days)

    # -------------------------------------------------------------- the rule
    def _static_check(self, g: str) -> str | None:
        """The corpus-independent part of the shape and stoplist rules (cached per surface form)."""
        if g not in self._static:
            self._static[g] = self._static_reason(g)
        return self._static[g]

    def _static_reason(self, g: str) -> str | None:
        toks = g.split()
        parts = [t.casefold() for t in _parts(g)]
        key = alias_key(g)
        if key in STOP_PHRASES or key in self.other.keys:
            return "stoplist"
        if key in self._spans:
            return "fragment"
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                if " ".join(parts[i:j]) in self.known:
                    return "known"
        for t in _parts(g):
            k = t.casefold()
            if k in STOP_TOKENS or k in GENERIC_TERMS or k in DEMONYMS:
                return "stoplist"
            if k in self.other.first_words or (t.isupper() and t in self.other.tickers):
                return "other registrant"
        if any(not _PLAIN_TOKEN.fullmatch(t) or t.endswith("&") for t in toks):
            return "residue"
        if toks[0].casefold() in GIVEN_NAMES:
            return "person"
        for t in {toks[0], toks[-1]}:
            k = t.casefold()
            if sum(c.isalpha() for c in t) < 3:
                return "short"
            if k in COMMON_WORDS or k in WORD_NAMES or k in GIVEN_NAMES or k in TITLE_CUES:
                return "common"
        return None

    def _person_token(self, t: str) -> bool:
        return self._stats.share(self._stats.person, t) >= self.rule.max_person

    def _surname(self, t: str) -> bool:
        """`t` mostly follows a first name learned from the corpus ("Andrej Karpathy": "Andrej" follows titles)."""
        k = t.casefold()
        n = sum(self._n[0][b] + self._n[1][b] for b in self._bigrams.get(k, ()) if self._person_token(b.split(" ")[0]))
        return n / max(1, self._stats.all[k]) >= self.rule.max_person

    def _corpus_check(self, g: str) -> str | None:
        """The corpus-dependent part of the shape rule (token statistics of the items counted so far)."""
        r, st = self.rule, self._stats
        toks = g.split()
        for t in {toks[0], toks[-1]}:
            if not st.proper(t, r):
                return "not a proper noun"
        for t in _parts(g):
            if t.casefold() in self._frag_tokens and st.proper(t, r):
                return "fragment"
            if self._person_token(t):
                return "person"
            if st.share(st.source, t) >= r.max_source:
                return "source"
        if len(toks) == 1 and self._surname(toks[0]):
            return "person"
        if st.share(st.company, toks[-1]) >= r.max_company:
            return "company"
        return None

    def _evaluate(self, k: str, s: str, members: frozenset[str], relax: float = 0.0) -> str | None:
        """Why the alias key `k` is not accepted for `s` at the current counts (None: accepted). `relax` loosens the
        share rules (hysteresis for an alias accepted at the previous anchor)."""
        r = self.rule
        per = self._by_g.get(k, {})
        a0, _a1, v0, v1 = per.get(s, (0, 0, 0.0, 0.0))
        w0, w1 = self._w[0][k], self._w[1][k]
        if a0 < r.min_own or w0 <= 0:
            return "support"
        if v0 / w0 < r.min_share - relax:
            return "co-occurrence"
        if self._n[1][k] < r.min_heldout or v1 / w1 < r.min_heldout_share - relax:
            return "held-out precision"
        w = w0 + w1
        if self._x[k] / w > r.max_extra + relax:
            return "other company tagged"
        if self._ox[k] / w > r.max_other_named + relax:
            return "other company named"
        rival = max((c[2] + c[3] for s2, c in per.items() if s2 != s and s2 in members), default=0.0)
        if rival / w > r.max_rival + relax:
            return "exclusivity"
        g = self.form(k)
        return self._static_check(g) or self._corpus_check(g)

    def explain(self, alias: str, symbol: str, anchor: pd.Timestamp | None = None) -> str | None:
        """Why `alias` is not an alias of `symbol` at `anchor` (default: the latest stepped anchor): the first failed
        rule, or None if it is accepted. Counts are those of the latest stepped anchor, so `anchor` must be it."""
        if anchor is not None:
            self.aliases_at(anchor)
            if self.anchor_for(anchor) != self._last:
                raise ValueError("explain is available for the latest stepped anchor only")
        if self._last is None:
            return "no anchor stepped"
        k = alias_key(alias)
        if k in self._current.get(symbol, {}):
            return None
        if symbol not in self._members_at(self._last):
            return "not a member"
        if not self._alive(symbol, self._last):
            return "not trading"
        if self._n_counted < self.rule.min_corpus:
            return "corpus too small"
        static = self._static_check(self.form(k)) if k in self._forms else self._static_check(alias)
        return static or self._evaluate(k, symbol, self._members_at(self._last)) or "not accepted"

    def form(self, k: str) -> str:
        """The most frequent surface form of an alias key."""
        forms = self._forms.get(k)
        return max(forms.items(), key=lambda kv: (kv[1], kv[0]))[0] if forms else k

    def _accept(self, anchor: pd.Timestamp, members: frozenset[str]) -> None:
        prev = self._current
        cur: dict[str, dict[str, tuple[pd.Timestamp, int]]] = {}
        if self._n_counted >= self.rule.min_corpus:
            for k, s in self._hot:
                if s not in members or not self._alive(s, anchor):
                    continue
                kept = prev.get(s, {}).get(k)
                if self._evaluate(k, s, members, self.rule.hysteresis if kept else 0.0) is not None:
                    continue
                a0, a1 = self._by_g[k][s][:2]
                cur.setdefault(s, {})[k] = (kept[0] if kept else anchor, a0 + a1)
        self._current = cur
        self._results[anchor] = {s: tuple(self.form(k) for k, _ in sorted(v.items(), key=lambda kv: (-kv[1][1], kv[0])))
                                 for s, v in cur.items() if v}
        self._first[anchor] = {(s, k): first for s, v in cur.items() for k, (first, _) in v.items()}
        self._case_free[anchor] = frozenset(k for v in cur.values() for k in v
                                            if any(f[:1].islower() for f in self._forms.get(k, ())))

    def evidence(self, anchor: pd.Timestamp) -> pd.DataFrame:
        """The accepted (firm, alias) pairs at `anchor` with the anchor they have been accepted since and the counts at
        `anchor` (for audits)."""
        anchor = self.anchor_for(anchor)
        self.aliases_at(anchor)
        if anchor != self._last:
            raise ValueError("evidence is available for the latest stepped anchor only")
        rows = []
        for s, v in self._current.items():
            for k, (first, support) in v.items():
                a0, a1, v0, v1 = self._by_g[k].get(s, (0, 0, 0.0, 0.0))
                n0, n1 = self._n[0][k], self._n[1][k]
                w0, w1 = self._w[0][k], self._w[1][k]
                w = max(1e-12, w0 + w1)
                rows.append({"symbol": s, "alias": self.form(k), "key": k, "since": first, "items": n0 + n1,
                             "own_items": a0 + a1, "share_discovery": v0 / max(1e-12, w0),
                             "share_heldout": v1 / max(1e-12, w1), "extra_share": self._x[k] / w,
                             "other_named_share": self._ox[k] / w})
        return pd.DataFrame(rows, columns=["symbol", "alias", "key", "since", "items", "own_items", "share_discovery",
                                           "share_heldout", "extra_share", "other_named_share"])


_CODE = re.compile(r"Company-[0-9A-F]{8}")


# ---------------------------------------------------------------------------------- loading
def lives_from_bars(bars: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """symbol -> (first, last) session with a bar: which of two claimants of a shared alias traded at a cutoff."""
    d = pd.to_datetime(bars["date"])
    g = d.groupby(bars["symbol"]).agg(["min", "max"])
    return {s: (r["min"], r["max"]) for s, r in g.iterrows()}


# the curated alias tables (docs/impl/dev2_blind.md §2 and §4): data/raw/aliases next to data/raw/sec
MANUAL_ALIASES_FILE = "manual_aliases.csv"
EXECUTIVES_FILE = "executive_aliases.csv"
ALIAS_MODES = ("name", "word")


def aliases_dir(sub_dir: Path) -> Path:
    """data/raw/aliases for the submissions directory data/raw/sec/submissions."""
    return Path(sub_dir).resolve().parents[1] / "aliases"


def load_manual_aliases(path: Path) -> dict[str, dict[str, str]]:
    """symbol -> {alias: mode} from the curated list (one row per alias with its source and corpus check; columns
    symbol, alias, mode, ...). mode "name": masked wherever it occurs capital-first, even if word-like; "word": only in
    explicit contexts or text tagged to the firm. Rows whose mode is neither are refused (a typo must not silently
    drop an alias). A missing file gives {} (logged). Validity (optional columns valid_from, valid_to):
    `load_manual_alias_dates`."""
    return {s: {a: m for a, (m, _) in v.items()} for s, v in _manual_rows(path).items()}


def load_manual_alias_dates(path: Path) -> dict[str, dict[str, Interval]]:
    """symbol -> {alias: (valid_from, valid_to)} for the curated aliases with a validity (v4: a subsidiary acquired or
    sold during the study period, such as "E*TRADE" of MS from its closing on 2020-10-02). Such an alias is masked
    only inside its validity (`FirmAliases.gated_names`)."""
    return {s: {a: iv for a, (_, iv) in v.items() if iv is not None} for s, v in _manual_rows(path).items()
            if any(iv is not None for _, iv in v.values())}


def _manual_rows(path: Path) -> dict[str, dict[str, tuple[str, Interval | None]]]:
    p = Path(path)
    if not p.exists():
        log.warning("no manual alias list at %s: only the built-in BRAND_ALIASES are used", p)
        return {}
    df = pd.read_csv(p, dtype=str, keep_default_na=False, comment=None)
    out: dict[str, dict[str, tuple[str, Interval | None]]] = defaultdict(dict)
    for r in df.to_dict("records"):
        sym, alias, mode = r["symbol"].strip(), r["alias"].strip(), r["mode"].strip()
        if not sym or sym.startswith("#") or not alias:
            continue
        if mode not in ALIAS_MODES:
            raise ValueError(f"{p}: alias {alias!r} of {sym} has mode {mode!r}, not one of {ALIAS_MODES}")
        a, b = (r.get("valid_from") or "").strip(), (r.get("valid_to") or "").strip()
        out[sym][alias] = (mode, (_utc(a or None), _utc(b or None)) if a or b else None)
    return dict(out)


def load_executives(path: Path) -> dict[str, tuple[Executive, ...]]:
    """symbol -> executives from `scripts/build_executive_aliases.py`'s table (columns symbol, name, surname, forms
    '|'-joined, start, surname_from; v4: form_starts '|'-joined and aligned with forms, surname_word). A missing file
    gives {} (logged); a table without the v4 columns gives every form the person's start and no common-word flag."""
    p = Path(path)
    if not p.exists():
        log.warning("no executive table at %s: executives are not masked", p)
        return {}
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    out: dict[str, list[Executive]] = defaultdict(list)
    for r in df.to_dict("records"):
        forms = tuple(x for x in r["forms"].split("|") if x)
        starts: tuple[pd.Timestamp | None, ...] = ()
        if r.get("form_starts"):
            starts = tuple(_utc(x or None) for x in r["form_starts"].split("|"))
            if len(starts) != len(forms):
                raise ValueError(f"{p}: {r['symbol']} {r['name']}: {len(starts)} form starts for {len(forms)} forms")
        out[r["symbol"]].append(Executive(r["name"], forms, r["surname"], _utc(r["start"] or None),
                                          _utc(r["surname_from"] or None), starts,
                                          str(r.get("surname_word", "")).strip().lower() == "true"))
    return {s: tuple(v) for s, v in out.items()}


def aliases_sha256(sub_dir: Path) -> str:
    """Identity of the curated alias tables (enters the context description: a table change changes every state it
    touches)."""
    h = hashlib.sha256()
    for name in (MANUAL_ALIASES_FILE, EXECUTIVES_FILE):
        p = aliases_dir(sub_dir) / name
        h.update(f"{name}\t{hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'missing'}\n".encode())
    return h.hexdigest()[:16]


def load_blinder(
    sub_dir: Path,
    symbols: Iterable[str],
    extra_aliases: Mapping[str, Iterable[str]] | None = None,
    resolver=None,
    history: tuple[date, date] | None = None,
    brand_aliases: Mapping[str, Iterable[str]] = BRAND_ALIASES,
    discovery_items: pd.DataFrame | None = None,
    salt: str = SALT,
    lives: Mapping[str, tuple] | None = None,
    members: Callable[[pd.Timestamp], Iterable[str]] | None = None,
    other: OtherRegistrants | None = None,
    rule: DiscoveryRule | None = None,
    manual_aliases: Mapping[str, Mapping[str, str]] | None = None,
    executives: Mapping[str, Iterable[Executive]] | None = None,
) -> Blinder:
    """Blinder over the full symbol set (every PIT universe member of the study, not only one week's U_t).

    `resolver` + `history`: historical tickers (dated) from `SymbolResolver.history(symbol, *history)`.
    `discovery_items`: news rows (id, symbol, created_at, updated_at, headline, summary, optional n_tags) for PIT alias
    discovery: every row of the archive, MARKET included (items of other firms are what exclusivity is measured on).
    `members(anchor)`: the study firms that are members at a discovery anchor (PIT; None: all symbols).
    `other`: SEC registrants outside the study set (default: `load_other_registrants(sub_dir.parent, symbols)`).
    `lives`: symbol -> (first, last) trading session (`lives_from_bars`), to resolve shared aliases.
    `manual_aliases`: symbol -> {alias: mode} (default: `aliases_dir(sub_dir)/manual_aliases.csv`; {} for none), on
    top of `brand_aliases`. `executives`: symbol -> executives (default: `aliases_dir(sub_dir)/executive_aliases.csv`;
    {} for none).
    """
    symbols = list(symbols)
    manual_dates: Mapping[str, Mapping[str, Interval]] = {}
    if manual_aliases is None:
        manual_aliases = load_manual_aliases(aliases_dir(sub_dir) / MANUAL_ALIASES_FILE)
        manual_dates = load_manual_alias_dates(aliases_dir(sub_dir) / MANUAL_ALIASES_FILE)
    if executives is None:
        executives = load_executives(aliases_dir(sub_dir) / EXECUTIVES_FILE)
    firms: dict[str, FirmAliases] = {}
    for s in symbols:
        f = Path(sub_dir) / f"{s}.json"
        sub = json.loads(f.read_text()) if f.exists() else None
        hist = [(t, a, b) for t, a, b in resolver.history(s, *history)] if resolver is not None and history else []
        man = manual_aliases.get(s, {}) or {}
        extra = [*(brand_aliases.get(s, ()) or ()), *((extra_aliases or {}).get(s, ()) or ())]
        firms[s] = company_aliases(s, sub, extra, hist, hard=[a for a, m in man.items() if m == "name"],
                                   soft=[a for a, m in man.items() if m == "word"],
                                   executives=executives.get(s, ()) or (), dated=manual_dates.get(s))
    disc = None
    if discovery_items is not None:
        if other is None:
            other = load_other_registrants(Path(sub_dir).parent, symbols)
        disc = AliasDiscovery(discovery_items, symbols, firms, members=members, other=other, rule=rule, lives=lives,
                              salt=salt)
    b = Blinder(firms, salt=salt, discovery=disc, lives=lives)
    if b.shared:
        log.info("%d aliases claimed by several firms (resolved per cutoff): %s", len(b.shared),
                 dict(list(sorted(b.shared.items()))[:20]))
    return b
