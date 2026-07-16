from __future__ import annotations

import html
import json
import os
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Callable, Optional

from rdflib import Graph, URIRef, Literal, XSD

DCTERMS_CREATED = URIRef("http://purl.org/dc/terms/created")
DCTERMS_MODIFIED = URIRef("http://purl.org/dc/terms/modified")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; PrivacyPolicyComplianceAnalyzer/1.0; "
    "+https://example.org/privacy-policy-compliance-analyzer)"
)

# ---------------------------------------------------------------------------
# Agent 1 now uses real web search discovery instead of guessed URL paths.
# ---------------------------------------------------------------------------



@dataclass
class DateCandidate:
    raw: str
    parsed: Optional[datetime]
    label: str
    confidence: str
    source: str  # regex | llm | stored
    evidence: str = ""


class _VisibleTextParser(HTMLParser):
    """Small dependency-free HTML visible-text extractor."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    BLOCK_TAGS = {
        "p", "div", "section", "article", "header", "footer", "main", "br",
        "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        data = html.unescape(data or "").strip()
        if data:
            self.parts.append(data)

    def get_text(self) -> str:
        text = " ".join(self.parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()


#agent1 url fetcher

def fetch_policy_source(url: str, timeout: int = 20) -> dict[str, Any]:
    """Fetch a URL and return content bytes + metadata."""
    if not url or not str(url).strip():
        raise ValueError("policy_url is required")

    req = urllib.request.Request(
        str(url).strip(),
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/pdf,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read()
        content_type = resp.headers.get("content-type", "")
        last_modified = resp.headers.get("last-modified")
        final_url = resp.geturl()

    return {
        "url": final_url,
        "content": content,
        "content_type": content_type,
        "last_modified_header": last_modified,
    }


def extract_text_from_html(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    parser = _VisibleTextParser()
    parser.feed(raw)
    return clean_policy_text(parser.get_text())


def extract_text_from_pdf_bytes(raw: bytes) -> str:
    """Optional PDF support. Uses PyPDF2 when available."""
    try:
        from io import BytesIO
        from PyPDF2 import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PDF extraction requires PyPDF2") from exc

    reader = PdfReader(BytesIO(raw))
    return clean_policy_text("\n\n".join(page.extract_text() or "" for page in reader.pages))


def extract_policy_text(fetched: dict[str, Any]) -> str:
    content_type = (fetched.get("content_type") or "").lower()
    content = fetched.get("content") or b""
    if "pdf" in content_type or str(fetched.get("url", "")).lower().endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    return extract_text_from_html(content)


def clean_policy_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Date parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

_DATE_PATTERNS = [
    # Last Updated: August 12, 2025
    rf"(?P<label>last\s+(?:updated|modified|revised)|effective\s+(?:date|as\s+of)|updated\s+on|revised\s+on|date\s+of\s+last\s+revision)\s*[:\-–—]?\s*(?P<date>(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}})",
    # Effective Date: 2025-08-12
    r"(?P<label>last\s+(?:updated|modified|revised)|effective\s+(?:date|as\s+of)|updated\s+on|revised\s+on|date\s+of\s+last\s+revision)\s*[:\-–—]?\s*(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    # Effective Date: 08/12/2025
    r"(?P<label>last\s+(?:updated|modified|revised)|effective\s+(?:date|as\s+of)|updated\s+on|revised\s+on|date\s+of\s+last\s+revision)\s*[:\-–—]?\s*(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})",
    # This Privacy Policy was last updated on August 12, 2025
    rf"(?P<label>privacy\s+policy\s+was\s+last\s+(?:updated|modified|revised)\s+on)\s*(?P<date>(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}})",
]


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_date_guess(value: str | None) -> Optional[datetime]:
    """Parse common policy date formats without adding a heavy dependency."""
    if not value:
        return None
    s = str(value).strip()
    s = re.sub(r"^(as of|on)\s+", "", s, flags=re.I).strip()
    s = s.replace("Sept.", "Sep").replace("Sept ", "Sep ")

    # ISO / numeric patterns
    numeric_formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
        "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",
    ]
    for fmt in numeric_formats:
        try:
            return _as_utc(datetime.strptime(s, fmt))
        except ValueError:
            pass

    # Month-name patterns
    month_formats = ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%B %Y", "%b %Y"]
    for fmt in month_formats:
        try:
            return _as_utc(datetime.strptime(s, fmt))
        except ValueError:
            pass

    # RFC/HTTP date header
    try:
        return _as_utc(parsedate_to_datetime(s))
    except Exception:
        return None


def format_date(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return _as_utc(dt).date().isoformat()


#agent2 data extracter

def extract_policy_date_regex(policy_text: str, last_modified_header: str | None = None) -> DateCandidate:
    """Use deterministic patterns first. Returns confidence none/low/high."""
    text = clean_policy_text(policy_text)
    candidates: list[DateCandidate] = []

    search_area = text[:8000]  # most policies put update date near the top
    for pattern in _DATE_PATTERNS:
        for m in re.finditer(pattern, search_area, flags=re.I):
            raw_date = m.group("date").strip(" .")
            parsed = parse_date_guess(raw_date)
            if parsed:
                label = re.sub(r"\s+", " ", m.group("label").strip())
                evidence = m.group(0).strip()
                candidates.append(DateCandidate(raw_date, parsed, label, "high", "regex", evidence))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Pick newest but mark low confidence so LLM can validate if enabled.
        newest = max(candidates, key=lambda c: c.parsed or datetime.min.replace(tzinfo=timezone.utc))
        newest.confidence = "low"
        newest.evidence = "Multiple policy date candidates found; newest candidate: " + newest.evidence
        return newest

    header_date = parse_date_guess(last_modified_header)
    if header_date:
        return DateCandidate(last_modified_header or "", header_date, "HTTP Last-Modified", "low", "regex", last_modified_header or "")

    return DateCandidate("", None, "", "none", "regex", "")


def _json_from_llm(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def llm_extract_policy_date(
    policy_text: str,
    llm_client: Any = None,
    llm_model: str = "llama-3.1-8b-instant",
    llm_callable: Optional[Callable[[str, str], str]] = None,
) -> DateCandidate:
    """LLM fallback for unclear policy dates. It extracts only; it does not update the KG."""
    clipped = clean_policy_text(policy_text)[:12000]
    system = (
        "You extract the effective date or last-updated date from privacy policies. "
        "Return JSON only. Do not invent dates. If no reliable policy update date is present, return null."
    )
    user = f"""
Find the date that belongs to the privacy policy itself, such as Last Updated, Effective Date, Last Modified, or Revised.
Ignore copyright years, footer years, blog dates, cookie banner dates, and unrelated dates.

Return exactly this JSON shape:
{{
  "date_found": "YYYY-MM-DD or null",
  "date_label": "Last Updated / Effective Date / null",
  "confidence": "high / medium / low / none",
  "evidence": "short quote from the policy, or empty string"
}}

PRIVACY POLICY TEXT:
{clipped}
""".strip()

    if llm_callable is not None:
        raw = llm_callable(system, user)
    elif llm_client is not None:
        resp = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=220,
        )
        raw = resp.choices[0].message.content.strip()
    else:
        return DateCandidate("", None, "", "none", "llm", "No LLM client configured")

    try:
        data = _json_from_llm(raw)
    except Exception:
        return DateCandidate("", None, "", "none", "llm", "LLM returned invalid JSON")

    date_found = data.get("date_found")
    parsed = parse_date_guess(date_found)
    if not parsed:
        return DateCandidate("", None, str(data.get("date_label") or ""), "none", "llm", str(data.get("evidence") or ""))

    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in {"high", "medium", "low", "none"}:
        confidence = "low"
    return DateCandidate(str(date_found), parsed, str(data.get("date_label") or "LLM extracted date"), confidence, "llm", str(data.get("evidence") or ""))


def extract_policy_date_agent(
    policy_text: str,
    last_modified_header: str | None = None,
    llm_client: Any = None,
    llm_model: str = "llama-3.1-8b-instant",
    llm_callable: Optional[Callable[[str, str], str]] = None,
    force_llm: bool = False,
) -> tuple[DateCandidate, bool]:
    """Regex first; LLM only when regex is unclear or forced."""
    regex_candidate = extract_policy_date_regex(policy_text, last_modified_header)
    if regex_candidate.confidence == "high" and not force_llm:
        return regex_candidate, False

    llm_candidate = llm_extract_policy_date(policy_text, llm_client, llm_model, llm_callable)
    if llm_candidate.parsed and llm_candidate.confidence in {"high", "medium"}:
        return llm_candidate, True

    return regex_candidate, bool(force_llm or regex_candidate.confidence in {"none", "low"})


#agent3 judge

def get_stored_company_date(g: Graph, manufacturer_iri: URIRef) -> DateCandidate:
    """Prefer dcterms:modified; fall back to dcterms:created."""
    for pred, label in [(DCTERMS_MODIFIED, "KG modified"), (DCTERMS_CREATED, "KG created")]:
        vals = list(g.objects(manufacturer_iri, pred))
        if vals:
            raw = str(vals[0])
            parsed = parse_date_guess(raw[:10]) or parse_date_guess(raw)
            return DateCandidate(raw, parsed, label, "high" if parsed else "none", "stored", raw)
    return DateCandidate("", None, "", "none", "stored", "")


def compare_policy_dates(stored: DateCandidate, live: DateCandidate) -> dict[str, Any]:
    if live.parsed is None:
        return {"status": "date_unknown", "should_update": False, "reason": "No reliable live policy date found."}
    if stored.parsed is None:
        return {"status": "outdated", "should_update": True, "reason": "No stored KG date found; live policy date is available."}
    if _as_utc(live.parsed).date() > _as_utc(stored.parsed).date():
        return {"status": "outdated", "should_update": True, "reason": "Live policy date is newer than stored KG date."}
    return {"status": "current", "should_update": False, "reason": "Stored KG policy date is current or newer."}


#agent4 policy updater than passes to the timestamp funct

def _report(
    *, company_name: str, policy_url: str, stored: DateCandidate, live: DateCandidate,
    status: str, action_taken: str, llm_used: bool, reason: str,
    previous_policy_saved: bool = False, update_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "target_type": "company_policy",
        "company": company_name,
        "policy_url": policy_url,
        "stored_date": format_date(stored.parsed),
        "stored_date_raw": stored.raw or None,
        "live_date": format_date(live.parsed),
        "live_date_raw": live.raw or None,
        "date_label": live.label or None,
        "date_source": live.source,
        "date_confidence": live.confidence,
        "date_evidence": live.evidence or None,
        "llm_used": llm_used,
        "status": status,
        "reason": reason,
        "action_taken": action_taken,
        "previous_policy_saved": previous_policy_saved,
        "update_info": update_info or {},
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def check_and_update_company_policy(
    *,
    g: Graph,
    manufacturer_iri: URIRef,
    policy_prop: URIRef,
    policy_url: str,
    company_name: str,
    timestamp_module: Any,
    ontology_path: Optional[str] = None,
    llm_client: Any = None,
    llm_model: str = "llama-3.1-8b-instant",
    llm_callable: Optional[Callable[[str, str], str]] = None,
    force_llm: bool = False,
) -> dict[str, Any]:
    """
    Main policy-only agentic update pipeline.

    This updates the KG only when the live policy date is clearly newer.
    The old policy is preserved through timestamp_module.upsert_policy().
    """
    fetched = fetch_policy_source(policy_url)
    live_policy_text = extract_policy_text(fetched)
    if not live_policy_text:
        raise ValueError("Could not extract readable policy text from the source URL.")

    live_date, llm_used = extract_policy_date_agent(
        live_policy_text,
        last_modified_header=fetched.get("last_modified_header"),
        llm_client=llm_client,
        llm_model=llm_model,
        llm_callable=llm_callable,
        force_llm=force_llm,
    )
    stored_date = get_stored_company_date(g, manufacturer_iri)
    decision = compare_policy_dates(stored_date, live_date)
    checked_at = timestamp_module.mark_checked(g, manufacturer_iri)


    if not decision["should_update"]:
        action = "No KG update was made."
        if decision["status"] == "current":
            action = "No KG update needed; stored policy is current."
        return _report(
            company_name=company_name,
            policy_url=policy_url,
            stored=stored_date,
            live=live_date,
            status=decision["status"],
            action_taken=action,
            llm_used=llm_used,
            reason=decision["reason"],
        )

    update_info = timestamp_module.upsert_policy(g, manufacturer_iri, policy_prop, live_policy_text)
    if ontology_path:
        g.serialize(destination=str(ontology_path), format="xml")

    return _report(
        company_name=company_name,
        policy_url=policy_url,
        stored=stored_date,
        live=live_date,
        status="updated",
        action_taken="Knowledge graph policy_description updated before scoring.",
        llm_used=llm_used,
        reason=decision["reason"],
        previous_policy_saved=bool(update_info.get("previous_policy")),
        update_info=update_info,
    )


# =============================================================================
# Agent 1 — Dynamic manufacturer-name source discovery
# =============================================================================

def _company_slug(company_name: str) -> str:
    """Normalize a user-entered manufacturer name into a domain-like slug."""
    name = (company_name or "").lower().strip()
    # Remove common legal suffixes that do not usually appear in domains.
    name = re.sub(r"\b(inc|inc\.|llc|ltd|corp|corporation|company|co|co\.)\b", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)



# ---------------------------------------------------------------------------
# REAL WEB SEARCH DISCOVERY (replaces the old guess-only implementation)
# ---------------------------------------------------------------------------
# The old implementation tried to guess URLs like:
#   https://www.company.com/privacy-policy
# That fails for companies such as Control4 whose policy may live on
# my.control4.com/PrivacyPolicy.aspx or another official subdomain.
#
# This implementation performs a real web search for the manufacturer name,
# collects candidate policy URLs, fetches those pages, validates the content,
# and then lets the LLM select the most official privacy-policy source.

BLOCKED_RESULT_DOMAINS = (
    "duckduckgo.com", "bing.com", "google.com", "yahoo.com", "wikipedia.org",
    "youtube.com", "youtu.be", "facebook.com", "instagram.com", "x.com",
    "twitter.com", "linkedin.com", "reddit.com", "pinterest.com", "github.com",
)

BAD_URL_HINTS = (
    "blog", "news", "press", "article", "forum", "community", "reviews",
    "login", "signin", "signup", "careers", "jobs", "support/article",
    "youtube", "facebook", "linkedin", "reddit",
)


def _strip_tags(value: str) -> str:
    value = re.sub(r"<script.*?</script>", " ", value or "", flags=re.I | re.S)
    value = re.sub(r"<style.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return clean_policy_text(html.unescape(value))


def _clean_search_url(raw_url: str, base_url: str = "") -> Optional[str]:
    """Normalize result links from search-engine HTML pages."""
    if not raw_url:
        return None

    url = html.unescape(str(raw_url).strip())
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/") and base_url:
        url = urllib.parse.urljoin(base_url, url)

    # DuckDuckGo redirects often look like /l/?uddg=<encoded-url>
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        url = qs["uddg"][0]
    elif "q" in qs and qs["q"] and parsed.netloc.lower() in {"www.google.com", "google.com"}:
        url = qs["q"][0]

    url = html.unescape(url).strip()
    if not url.startswith(("http://", "https://")):
        return None

    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    if not host:
        return None
    if any(blocked in host for blocked in BLOCKED_RESULT_DOMAINS):
        return None

    # Remove fragments and obvious tracking noise.
    cleaned = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path.rstrip("/"),
        "",
        parsed.query,
        "",
    ))
    return cleaned


def _fetch_search_page(search_url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(
        search_url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def _parse_search_results(html_text: str, base_url: str) -> list[dict[str, Any]]:
    """Dependency-free search result parser for DuckDuckGo/Bing-like HTML."""
    results: list[dict[str, Any]] = []

    # First parse anchor tags so title text travels with the URL.
    for m in re.finditer(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html_text, flags=re.I | re.S):
        url = _clean_search_url(m.group(1), base_url=base_url)
        if not url:
            continue
        title = _strip_tags(m.group(2))[:240]
        results.append({"url": url, "title": title, "snippet": ""})

    # De-dupe by URL while preserving first useful title.
    dedup: dict[str, dict[str, Any]] = {}
    for r in results:
        if r["url"] not in dedup:
            dedup[r["url"]] = r
    return list(dedup.values())


def _search_result_score(company_name: str, url: str, title: str = "", snippet: str = "") -> int:
    """Score search-result metadata before fetching the page."""
    slug = _company_slug(company_name)
    company_terms = [t for t in re.split(r"\s+", re.sub(r"[^A-Za-z0-9 ]+", " ", company_name.lower())) if t]
    low_url = (url or "").lower()
    low_title = (title or "").lower()
    low_snip = (snippet or "").lower()
    host = urllib.parse.urlparse(low_url).netloc.replace("www.", "")
    joined = f"{low_url} {low_title} {low_snip}"

#score apparatus
    score = 0
    if ".us" in low_url:
        score += 5
    if "us." in low_url:
        score += 5
    if "/us/" in low_url:
        score += 5
    if "/pages/privacy-policy" in low_url:
        score += 3
    if "privacy" in low_url:
        score += 5
    if any(x in low_url for x in (
            "/eu/",
            "/uk/",
            "/en-gb/",
            "europe",
            "gdpr",
        )):
        score -= 10
    if "privacy policy" in joined or "privacy notice" in joined or "privacy statement" in joined:
        score += 5
    if slug and slug in host.replace(".", ""):
        score += 4
    elif slug and slug in low_url.replace(".", ""):
        score += 2
    if company_terms and any(term in joined for term in company_terms):
        score += 2
    if "official" in joined:
        score += 1
    if low_url.startswith("https://"):
        score += 1
    if any(bad in low_url for bad in BAD_URL_HINTS):
        score -= 4
    if "cookie" in low_url and "privacy" not in low_url:
        score -= 3
    return score

#thehint
INDUSTRY_HINT = "IoT connected device manufacturer"


def web_search_privacy_policy_urls(company_name: str, max_results: int = 10) -> list[dict[str, Any]]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")

    queries = [
        f"{company_name} {INDUSTRY_HINT} official privacy policy",
        f"{company_name} {INDUSTRY_HINT} privacy policy",
        f"{company_name} privacy notice",
        f"{company_name} privacy statement",
    ]

    results = []

    for q in queries:
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({
            "q": q,
            "count": 8,
            "search_lang": "en",
            "country": "us",
            "safesearch": "moderate",
        })

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
                "User-Agent": DEFAULT_USER_AGENT,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception as exc:
            print(f"[brave-search] query failed: {q} | {exc}")
            continue

        for item in data.get("web", {}).get("results", []):
            result_url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("description", "")

            if not result_url:
                continue

            results.append({
                "url": result_url,
                "title": clean_policy_text(title),
                "snippet": clean_policy_text(snippet),
                "query": q,
                "search_engine": "Brave Search API",
                "search_score": _search_result_score(
                    company_name,
                    result_url,
                    title,
                    snippet,
                ),
            })

        by_url = {}
    for r in results:
        if not r.get("url"):
            continue
        if r["url"] not in by_url or r["search_score"] > by_url[r["url"]]["search_score"]:
            by_url[r["url"]] = r

    # -----------------------------
    # Remove EU/GDPR-specific pages
    # -----------------------------
    EU_HINTS = (
        "/eu/",
        "/eea/",
        "/uk/",
        "/en-gb/",
        "europe",
        "gdpr",
        "privacy.eufy.com/eu",
    )

    filtered = [
        r for r in by_url.values()
        if not any(h in r["url"].lower() for h in EU_HINTS)
    ]

    # Only use the filtered list if we still have results.
    if filtered:
        ranked = sorted(filtered, key=lambda r: r["search_score"], reverse=True)
    else:
        ranked = sorted(by_url.values(), key=lambda r: r["search_score"], reverse=True)

        print("\n=== Ranked Candidate URLs ===")
    for r in ranked:
        print(r["search_score"], r["url"])
    print("=============================\n")

    return [r for r in ranked if r.get("search_score", 0) >= 3][:max_results]

def build_privacy_url_candidates(company_name: str) -> list[str]:
    """
    Build candidate policy URLs from real web-search results only.

    This intentionally replaces the old guessed-path implementation. The app no
    longer depends on /privacy, /privacy-policy, etc. as the main discovery
    method. It searches the web for "<manufacturer> official privacy policy" and
    related queries, then candidate pages are fetched and validated below.
    """
    search_results = web_search_privacy_policy_urls(company_name, max_results=18)
    return list(dict.fromkeys([r["url"] for r in search_results if r.get("url")]))


def _looks_like_privacy_policy(company_name: str, url: str, text: str) -> tuple[bool, int, str]:
    """
    Lightweight source validation before using a URL.
    Returns (valid_enough, score, reason).
    """
    low_text = clean_policy_text(text).lower()
    low_url = (url or "").lower()
    slug = _company_slug(company_name)

    score = 0
    reasons: list[str] = []

    if "privacy policy" in low_text or "privacy notice" in low_text or "privacy statement" in low_text:
        score += 4
        reasons.append("page contains privacy-policy wording")
    if "privacy" in low_url:
        score += 2
        reasons.append("URL contains privacy")
    if slug and slug in urllib.parse.urlparse(low_url).netloc.replace(".", ""):
        score += 2
        reasons.append("domain appears related to company name")
    if len(low_text) >= 1500:
        score += 2
        reasons.append("page has enough readable text")
    if any(bad in low_url for bad in ["blog", "news", "press", "support/article"]):
        score -= 2
        reasons.append("URL may be blog/news/support content")

    return score >= 5, score, "; ".join(reasons)

#5  extracting policy from link
def collect_policy_url_candidates(company_name: str, max_candidates_to_fetch: int = 10) -> list[dict[str, Any]]:
    """
    Real web-search + content validation.

    1. Search the web for the manufacturer privacy policy.
    2. Fetch top candidate pages.
    3. Keep pages that look like full official privacy policies.
    4. Pass those pages to the LLM selector.
    """
    out: list[dict[str, Any]] = []
    search_results = web_search_privacy_policy_urls(company_name, max_results=max_candidates_to_fetch)

    for result in search_results[:max_candidates_to_fetch]:
        url = result.get("url")
        if not url:
            continue
        try:
            fetched = fetch_policy_source(url, timeout=12)
            final_url = fetched.get("url") or url
            text = extract_policy_text(fetched)
            ok, score, reason = _looks_like_privacy_policy(company_name, final_url, text)

            # Blend search score and page validation score.
            combined_score = int(result.get("search_score", 0)) + int(score)

            if ok:
                out.append({
                    "url": final_url,
                    "score": combined_score,
                    "search_score": result.get("search_score", 0),
                    "validation_score": score,
                    "search_title": result.get("title", ""),
                    "search_engine": result.get("search_engine", ""),
                    "query": result.get("query", ""),
                    "reason": f"search: {result.get('title','')}; validation: {reason}",
                    "text_preview": clean_policy_text(text)[:1000],
                    "text_length": len(text),
                })
        except Exception as exc:
            continue

    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    return out[:8]

#6 select which link is best
def llm_select_policy_url(
    company_name: str,
    candidates: list[dict[str, Any]],
    llm_client: Any = None,
    llm_model: str = "llama-3.1-8b-instant",
    llm_callable: Optional[Callable[[str, str], str]] = None,
) -> dict[str, Any]:
    """
    LLM selector for the source-discovery step.
    It chooses the most official privacy-policy URL from candidate pages.
    It does not update the KG.
    """
    if not candidates:
        return {
            "selected_policy_url": None,
            "confidence": "none",
            "llm_used_for_url": False,
            "reason": "No candidate privacy policy pages were found.",
        }

    # In the real-search implementation, the LLM is the source-selection agent.
    # Only skip the LLM when no client/callable is configured.

    system = (
        "You select the official privacy policy URL for a company. "
        "Return JSON only. Reject blogs, news articles, summaries, cookie-setting pages, "
        "and unrelated third-party pages."
    )
    candidate_lines = []
    for i, c in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{i}. URL: {c.get('url')}\n"
            f"   Score: {c.get('score')}\n"
            f"   Reason: {c.get('reason')}\n"
            f"   Preview: {c.get('text_preview')}"
        )

    user = f"""
Company: {company_name}

Candidate privacy-policy pages:
{chr(10).join(candidate_lines)}

Choose the best official privacy policy URL for this company.

Return exactly:
{{
  "selected_policy_url": "URL or null",
  "confidence": "high / medium / low / none",
  "reason": "short explanation"
}}
""".strip()

    if llm_callable is not None:
        raw = llm_callable(system, user)
    elif llm_client is not None:
        resp = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=220,
        )
        raw = resp.choices[0].message.content.strip()
    else:
        # Deterministic fallback: choose highest scored candidate.
        best = candidates[0]
        return {
            "selected_policy_url": best.get("url"),
            "confidence": "medium",
            "llm_used_for_url": False,
            "reason": "No LLM client configured; selected highest-scoring candidate.",
            "candidates": candidates,
        }

    try:
        data = _json_from_llm(raw)
    except Exception:
        best = candidates[0]
        return {
            "selected_policy_url": best.get("url"),
            "confidence": "low",
            "llm_used_for_url": True,
            "reason": "LLM returned invalid JSON; fell back to highest-scoring candidate.",
            "candidates": candidates,
        }

    selected = data.get("selected_policy_url")
    confidence = str(data.get("confidence") or "low").lower()
    if confidence not in {"high", "medium", "low", "none"}:
        confidence = "low"

    valid_urls = {c["url"] for c in candidates}
    if selected not in valid_urls:
        selected = None
        confidence = "none"

        print("\n========== LLM URL SELECTION ==========")
    print(f"Company: {company_name}")
    print(f"Selected URL: {selected}")
    print(f"Confidence: {confidence}")
    print(f"Reason: {data.get('reason')}")
    print("=======================================\n")


    return {
        "selected_policy_url": selected,
        "confidence": confidence,
        "llm_used_for_url": True,
        "reason": str(data.get("reason") or ""),
        "candidates": candidates,
    }

#2
def discover_policy_url_agent(
    company_name: str,
    llm_client: Any = None,
    llm_model: str = "llama-3.1-8b-instant",
    llm_callable: Optional[Callable[[str, str], str]] = None,
) -> dict[str, Any]:
    """
    find the official policy URL from only the manufacturer name.

    Uses real web search first, fetches candidate pages, then uses the LLM as
    the selector/validator agent.
    """
    candidates = collect_policy_url_candidates(company_name)
    result = llm_select_policy_url(
        company_name=company_name,
        candidates=candidates,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_callable=llm_callable,
    )
    result["company"] = company_name
    result["discovery_method"] = "real_web_search_plus_llm_selector"
    result["search_queries"] = [
        f"{company_name} official privacy policy",
        f"{company_name} privacy policy",
        f"{company_name} privacy notice",
    ]
    return result

#4 
def fetch_live_policy_for_company_name(
    company_name: str,
    llm_client: Any = None,
    llm_model: str = "llama-3.1-8b-instant",
    llm_callable: Optional[Callable[[str, str], str]] = None,
) -> dict[str, Any]:
    """
    Used by Classify button.

    Input: only a manufacturer name.
    Output: discovered URL + extracted live policy text + date metadata.
    """
    source = discover_policy_url_agent(
        company_name=company_name,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_callable=llm_callable,
    )
    policy_url = source.get("selected_policy_url")
    if not policy_url:
        return {
            "ok": False,
            "status": "source_not_found",
            "company": company_name,
            "source_discovery": source,
            "error": "Could not find a reliable official privacy policy URL for this manufacturer name.",
        }

    if source.get("confidence") not in {"high", "medium"}:
        return {
            "ok": False,
            "status": "source_low_confidence",
            "company": company_name,
            "policy_url": policy_url,
            "source_discovery": source,
            "error": "Policy URL was found, but confidence was too low for automatic KG insertion.",
        }

    fetched = fetch_policy_source(policy_url)
    live_policy_text = extract_policy_text(fetched)
    if len(live_policy_text) < 500:
        return {
            "ok": False,
            "status": "policy_text_too_short",
            "company": company_name,
            "policy_url": policy_url,
            "source_discovery": source,
            "error": "The discovered page did not contain enough readable policy text.",
        }

    live_date, llm_used_for_date = extract_policy_date_agent(
        live_policy_text,
        last_modified_header=fetched.get("last_modified_header"),
        llm_client=llm_client,
        llm_model=llm_model,
        llm_callable=llm_callable,
    )

    return {
        "ok": True,
        "status": "policy_found",
        "company": company_name,
        "policy_url": policy_url,
        "final_url": fetched.get("url") or policy_url,
        "policy_text": live_policy_text,
        "policy_text_length": len(live_policy_text),
        "live_date": format_date(live_date.parsed),
        "live_date_raw": live_date.raw or None,
        "date_label": live_date.label or None,
        "date_confidence": live_date.confidence,
        "date_source": live_date.source,
        "date_evidence": live_date.evidence or None,
        "llm_used_for_date": llm_used_for_date,
        "source_discovery": source,
    }


def auto_check_and_update_company_policy(
    *,
    company_name: str,
    manufacturer_iri: URIRef,
    stored_policy_text: str,
    g: Graph,
    policy_prop: URIRef,
    onto_path: Optional[str],
    groq_client: Any = None,
    timestamp_module: Any = None,
) -> dict[str, Any]:
    """
    Automatic update pipeline for an existing manufacturer.
    It discovers the policy URL from the manufacturer name, then runs the
    existing date comparison + KG update pipeline.
    """
    source = discover_policy_url_agent(company_name=company_name, llm_client=groq_client)
    policy_url = source.get("selected_policy_url")

    if not policy_url:
        return {
            "target_type": "company_policy",
            "company": company_name,
            "status": "source_not_found",
            "action_taken": "No KG update was made because no official privacy policy URL could be verified.",
            "source_discovery": source,
            "discovered_policy_url": None,
            "llm_used_for_url": source.get("llm_used_for_url", False),
        }

    report = check_and_update_company_policy(
        g=g,
        manufacturer_iri=manufacturer_iri,
        policy_prop=policy_prop,
        policy_url=policy_url,
        company_name=company_name,
        timestamp_module=timestamp_module,
        ontology_path=str(onto_path) if onto_path else None,
        llm_client=groq_client,
    )
    report["source_discovery"] = source
    report["discovered_policy_url"] = policy_url
    report["llm_used_for_url"] = source.get("llm_used_for_url", False)
    return report
