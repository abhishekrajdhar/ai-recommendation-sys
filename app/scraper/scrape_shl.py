from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.models.schemas import Assessment

logger = logging.getLogger(__name__)

# PUBLIC SHL CATALOG URL
CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "app" / "data" / "shl_catalog.json"

REQUEST_TIMEOUT = 20
USER_AGENT = "shl-assessment-recommender/1.0"


# =========================================================
# UTILITIES
# =========================================================

def clean_text(value: str | None) -> str:
    if not value:
        return ""

    value = re.sub(r"\s+", " ", value)
    return value.strip()


def split_values(value: str) -> list[str]:
    if not value:
        return []

    parts = re.split(r",|;|\||\n", value)

    cleaned = []
    for part in parts:
        item = clean_text(part)
        if item:
            cleaned.append(item)

    return cleaned


def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/") + "/",
            "",
            "",
            "",
        )
    )


def with_page(url: str, page: int) -> str:
    parsed = urlparse(url)

    query = parse_qs(parsed.query)
    query["page"] = [str(page)]

    return urlunparse(
        parsed._replace(query=urlencode(query, doseq=True))
    )


# =========================================================
# SESSION
# =========================================================

def create_session() -> requests.Session:
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": USER_AGENT
    })

    return session


# =========================================================
# FETCH
# =========================================================

def fetch(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=REQUEST_TIMEOUT)

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# =========================================================
# LINK DISCOVERY
# =========================================================

def is_individual_solution_link(href: str, text: str) -> bool:
    lowered = f"{href} {text}".lower()

    if "job-solutions" in lowered:
        return False

    if "/products/product-catalog/view/" not in lowered:
        return False

    return True


def discover_product_urls(
    session: requests.Session,
    max_pages: int = 50,
) -> list[str]:

    discovered = []
    seen = set()

    for page in range(1, max_pages + 1):

        url = (
            CATALOG_URL
            if page == 1
            else with_page(CATALOG_URL, page)
        )

        logger.info("Scanning catalog page: %s", url)

        page_links = []

        # Try to fetch raw response so we can handle JSON catalog endpoints
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

        except requests.RequestException as exc:
            logger.warning("Failed page %s: %s", url, exc)
            continue

        # If the catalog endpoint returns JSON (new dataset), extract URLs from JSON
        try:
            # try strict JSON first
            data = resp.json()

            def looks_like_product_path(s: str) -> bool:
                lowered = s.lower()
                return any(
                    token in lowered
                    for token in (
                        "product-catalog",
                        "product-catalog/view",
                        "products/product-catalog",
                        "/products/",
                        "shl.com/products",
                        "product-catalog/view",
                    )
                )

            def extract_urls_from_json(node) -> list[str]:
                urls = []

                if isinstance(node, dict):
                    for k, v in node.items():
                        # common key names that may contain URLs or paths
                        if isinstance(v, str):
                            if looks_like_product_path(v) or k.lower() in ("url", "href", "link", "path", "slug", "permalink", "permalink_path", "canonicalurl"):
                                urls.append(v)
                        elif isinstance(v, (dict, list)):
                            urls.extend(extract_urls_from_json(v))
                elif isinstance(node, list):
                    for item in node:
                        urls.extend(extract_urls_from_json(item))

                return urls

            candidate_urls = extract_urls_from_json(data)

            logger.info("Found %d candidate URLs in JSON on page %s", len(candidate_urls), page)

            for raw in candidate_urls:
                # sanitize and join relative paths
                if not isinstance(raw, str):
                    continue

                raw = raw.strip()

                # Sometimes values are HTML fragments or JSON-escaped strings; try to extract hrefs
                href_match = re.search(r'href\s*=\s*"([^"]+)"', raw)

                if href_match:
                    raw = href_match.group(1)

                full_url = normalize_url(urljoin(CATALOG_URL, raw))

                # keep the same filtering heuristic as HTML discovery
                if is_individual_solution_link(full_url, ""):
                    if full_url not in seen:
                        seen.add(full_url)
                        page_links.append(full_url)

        except ValueError:
            # Try to be tolerant: remove problematic control characters and retry
            try:
                cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", resp.text)
                data = json.loads(cleaned)

                candidate_urls = extract_urls_from_json(data)

                logger.info("Found %d candidate URLs after cleaning JSON on page %s", len(candidate_urls), page)

            except Exception:
                # Last resort: extract "link", "url", "href" values with a regex from the raw text
                candidate_urls = []

                for pattern in (r'"link"\s*:\s*"([^"]+)"', r'"url"\s*:\s*"([^"]+)"', r'"href"\s*:\s*"([^"]+)"'):
                    found = re.findall(pattern, resp.text)
                    if found:
                        candidate_urls.extend(found)

                logger.info("Found %d candidate URLs by regex on page %s", len(candidate_urls), page)

                # Normalize duplicates
                candidate_urls = list(dict.fromkeys(candidate_urls))

                for raw in candidate_urls:
                    if not isinstance(raw, str):
                        continue

                    raw = raw.strip()

                    href_match = re.search(r'href\s*=\s*"([^"]+)"', raw)

                    if href_match:
                        raw = href_match.group(1)

                    full_url = normalize_url(urljoin(CATALOG_URL, raw))

                    if is_individual_solution_link(full_url, ""):
                        if full_url not in seen:
                            seen.add(full_url)
                            page_links.append(full_url)

                # If we got candidates via regex, skip the HTML fallback below
                if page_links:
                    # proceed to next page handling
                    pass
                else:
                    # Not JSON-parsable and regex didn't find product links; fallback to HTML parsing
                    soup = BeautifulSoup(resp.text, "html.parser")

                    for anchor in soup.select("a[href]"):

                        href = anchor.get("href", "")
                        text = clean_text(anchor.get_text(" "))

                        full_url = normalize_url(
                            urljoin(CATALOG_URL, href)
                        )

                        if is_individual_solution_link(full_url, text):

                            if full_url not in seen:
                                seen.add(full_url)
                                page_links.append(full_url)

        if not page_links:
            logger.info("No more product links found.")
            break

        discovered.extend(page_links)

        time.sleep(0.2)

    return discovered


# =========================================================
# FIELD EXTRACTION
# =========================================================

def value_from_labels(
    soup: BeautifulSoup,
    labels: tuple[str, ...],
) -> str:

    pattern = re.compile(
        "|".join(re.escape(label) for label in labels),
        re.IGNORECASE,
    )

    for node in soup.find_all(string=pattern):

        parent = node.parent

        if not parent:
            continue

        row = parent.find_parent(["tr", "li", "div"])

        if not row:
            continue

        text = clean_text(row.get_text(" "))

        text = pattern.sub("", text, count=1)

        text = clean_text(text.strip(" :-"))

        if text:
            return text

    return ""


# =========================================================
# BOOLEAN NORMALIZATION
# =========================================================

YES_PATTERN = re.compile(
    r"\b(yes|supported|available|true)\b",
    re.IGNORECASE,
)

NO_PATTERN = re.compile(
    r"\b(no|unsupported|false|not available)\b",
    re.IGNORECASE,
)


def bool_yes_no(value: str) -> str:
    if not value:
        return ""

    value = value.lower()

    if YES_PATTERN.search(value):
        return "yes"

    if NO_PATTERN.search(value):
        return "no"

    return ""


# =========================================================
# DURATION NORMALIZATION
# =========================================================

def parse_duration_minutes(text: str) -> int | None:
    if not text:
        return None

    lowered = text.lower()

    hour_match = re.search(r"(\d+)\s*hour", lowered)

    if hour_match:
        return int(hour_match.group(1)) * 60

    minute_match = re.search(r"(\d+)\s*(minute|min)", lowered)

    if minute_match:
        return int(minute_match.group(1))

    return None


# =========================================================
# TEST TYPE MAPPING
# =========================================================

def map_test_type(text: str) -> str:
    lowered = text.lower()

    if any(k in lowered for k in ["personality", "opq"]):
        return "Personality"

    if any(k in lowered for k in ["cognitive", "ability", "aptitude", "reasoning", "gsa"]):
        return "Cognitive"

    if any(k in lowered for k in ["technical", "coding", "software", "skill", "knowledge", "java", "python", ".net", "c#"]):
        return "Technical"

    if any(k in lowered for k in ["situational", "judgment", "judgement", "sjq", "situational judgement"]):
        return "Situational Judgment"

    return "General"

    if any(
        word in lowered
        for word in [
            "ability",
            "aptitude",
            "cognitive",
            "reasoning",
        ]
    ):
        return "A"

    if any(
        word in lowered
        for word in [
            "technical",
            "coding",
            "java",
            "python",
            "software",
            "sql",
        ]
    ):
        return "K"

    if any(
        word in lowered
        for word in [
            "situational",
            "judgement",
            "judgment",
        ]
    ):
        return "S"

    return "O"


# =========================================================
# CATEGORY INFERENCE
# =========================================================

def infer_category(
    name: str,
    description: str,
    assessment_type: str,
) -> str:

    text = f"{name} {description} {assessment_type}".lower()

    if "personality" in text or "opq" in text:
        return "Personality"

    if any(
        word in text
        for word in [
            "cognitive",
            "ability",
            "reasoning",
            "aptitude",
        ]
    ):
        return "Cognitive"

    if any(
        word in text
        for word in [
            "technical",
            "coding",
            "java",
            "python",
            "software",
        ]
    ):
        return "Technical"

    if any(
        word in text
        for word in [
            "situational",
            "judgment",
            "judgement",
        ]
    ):
        return "Situational Judgment"

    return "General"


# =========================================================
# PRODUCT PAGE PARSING
# =========================================================

def parse_product_page(
    url: str,
    soup: BeautifulSoup,
) -> dict | None:

    title_node = soup.find("h1") or soup.find("title")

    name = clean_text(
        title_node.get_text(" ")
        if title_node
        else ""
    )

    if not name:
        return None

    name = name.replace("| SHL", "").strip()

    meta_desc = soup.select_one("meta[name='description']")

    description = clean_text(
        meta_desc.get("content")
        if meta_desc
        else ""
    )

    if not description:
        first_para = soup.find("p")

        if first_para:
            description = clean_text(
                first_para.get_text(" ")
            )

    assessment_type = value_from_labels(
        soup,
        ("Assessment type", "Type", "Test type"),
    )

    duration_raw = value_from_labels(
        soup,
        ("Duration", "Completion time", "Time"),
    )

    remote_raw = value_from_labels(
        soup,
        ("Remote testing", "Remote"),
    )

    adaptive_raw = value_from_labels(
        soup,
        ("Adaptive", "IRT"),
    )

    job_levels_raw = value_from_labels(
        soup,
        ("Job levels", "Job level"),
    )

    skills_raw = value_from_labels(
        soup,
        (
            "Skills measured",
            "Measures",
            "Knowledge, skills, abilities",
        ),
    )

    languages_raw = value_from_labels(
        soup,
        ("Languages", "Language support"),
    )

    taxonomy_raw = value_from_labels(
        soup,
        (
            "Categories",
            "Assessment Categories",
            "Solution Family",
        ),
    )

    # NORMALIZED VALUES

    duration = clean_text(duration_raw)

    duration_minutes = parse_duration_minutes(
        duration_raw
    )

    remote = bool_yes_no(remote_raw)

    adaptive = bool_yes_no(adaptive_raw)

    job_levels = split_values(job_levels_raw)

    skills = split_values(skills_raw)

    languages = split_values(languages_raw)

    taxonomy_categories = split_values(
        taxonomy_raw
    )

    test_type = map_test_type(
        f"{assessment_type} {name} {description}"
    )

    category = infer_category(
        name,
        description,
        assessment_type,
    )

    retrieval_text = (
        f"Assessment Name: {name}\n\n"
        f"Description:\n{description}\n\n"
        f"Skills:\n{', '.join(skills)}\n\n"
        f"Job Levels:\n{', '.join(job_levels)}\n\n"
        f"Categories:\n{', '.join(taxonomy_categories)}\n\n"
        f"Test Type:\n{test_type}\n\n"
        f"Remote:\n{remote}"
    )

    entity_id = ""

    for attr in [
        "data-entity-id",
        "data-id",
        "data-product-id",
    ]:

        node = soup.find(attrs={attr: True})

        if node and node.get(attr):
            entity_id = str(node.get(attr)).strip()
            break

    if not entity_id:
        match = re.search(r"/(\d{3,8})/", url)

        if match:
            entity_id = match.group(1)

    record = {
        "entity_id": entity_id,
        "name": name,
        "link": normalize_url(url),
        # keep `url` key for compatibility with the rest of the app
        "url": normalize_url(url),
        "scraped_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "description": description,
        "assessment_type": assessment_type,
        "test_type": test_type,
        "category": category,
        "job_levels": job_levels,
        "job_levels_raw": job_levels_raw,
        "skills_measured": skills,
        "skills_raw": skills_raw,
        "languages": languages,
        "languages_raw": languages_raw,
        "taxonomy_categories": taxonomy_categories,
        "taxonomy_raw": taxonomy_raw,
        "duration": duration,
        "duration_raw": duration_raw,
        "duration_minutes": duration_minutes,
        # both normalized boolean-like flags and the original label are useful
        "remote": remote,
        "adaptive": adaptive,
        "remote_testing": remote_raw,
        "adaptive_support": adaptive_raw,
        "status": "ok",
        "retrieval_text": clean_text(retrieval_text),
    }

    # VALIDATION

    try:

        Assessment(
            name=name,
            url=record["link"],
            description=description,
            assessment_type=assessment_type,
            duration=duration,
            remote_testing=remote,
            adaptive_support=adaptive,
            job_levels=job_levels,
            skills_measured=skills,
            languages=languages,
            category=category,
        )

    except Exception as exc:

        logger.warning(
            "Skipping malformed entry %s: %s",
            url,
            exc,
        )

        return None

    return record


# =========================================================
# MAIN SCRAPER
# =========================================================

def scrape_catalog(
    output_path: Path = DEFAULT_OUTPUT,
    max_pages: int = 50,
    use_json_catalog: bool = False,
) -> list[dict]:

    session = create_session()

    assessments: list[dict] = []

    # Fast path: if the catalog URL returns a JSON array of full product records,
    # normalize and validate them directly (avoids fetching each product page).
    try:
        resp = session.get(CATALOG_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        json_data = None

        try:
            json_data = resp.json()
        except Exception:
            # try to clean control characters and parse again
            json_data = None
            try:
                cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", resp.text)
                json_data = json.loads(cleaned)
            except Exception:
                # As a last resort, extract "link"/"url" values with regex and build minimal records
                candidate_urls = []
                for pattern in (r'"link"\s*:\s*"([^"]+)"', r'"url"\s*:\s*"([^"]+)"', r'"href"\s*:\s*"([^"]+)"'):
                    found = re.findall(pattern, resp.text)
                    if found:
                        candidate_urls.extend(found)

                candidate_urls = list(dict.fromkeys(candidate_urls))

                if candidate_urls:
                    logger.info("Built %d minimal JSON records from regex links", len(candidate_urls))
                    json_data = [{"link": u} for u in candidate_urls]

        if json_data and isinstance(json_data, list) and (use_json_catalog or any(isinstance(i, dict) and (i.get("link") or i.get("entity_id")) for i in json_data[:5])):
            logger.info("Using JSON fast-path: processing %d entries from catalog JSON", len(json_data))

            def normalize_from_item(item: dict) -> dict | None:
                try:
                    url_val = item.get("link") or item.get("url") or ""
                    url_norm = normalize_url(urljoin(CATALOG_URL, str(url_val))) if url_val else ""

                    # If the JSON item doesn't contain name/description, try to fetch the
                    # actual product page and parse it to produce a full record.
                    if url_norm and (not item.get("name") or not item.get("description")):
                        try:
                            soup = fetch(session, url_norm)
                            parsed = parse_product_page(url_norm, soup)
                            if parsed:
                                return parsed
                        except Exception as exc:
                            logger.debug("Failed to enrich JSON entry by fetching page %s: %s", url_norm, exc)

                    name = clean_text(item.get("name") or "")
                    description = clean_text(item.get("description") or "")

                    assessment_type = clean_text(item.get("assessment_type") or item.get("type") or "")

                    duration_raw = item.get("duration_raw") or item.get("duration") or ""
                    duration = clean_text(duration_raw)
                    duration_minutes = parse_duration_minutes(duration_raw)

                    remote_raw = item.get("remote") or item.get("remote_testing") or ""
                    remote = bool_yes_no(str(remote_raw))

                    adaptive_raw = item.get("adaptive") or item.get("adaptive_support") or ""
                    adaptive = bool_yes_no(str(adaptive_raw))

                    job_levels = item.get("job_levels") or split_values(item.get("job_levels_raw") or "")
                    job_levels_raw = item.get("job_levels_raw") or (", ".join(job_levels) if job_levels else "")

                    skills = item.get("skills_measured") or split_values(item.get("skills_raw") or "")
                    skills_raw = item.get("skills_raw") or (", ".join(skills) if skills else "")

                    languages = item.get("languages") or split_values(item.get("languages_raw") or "")
                    languages_raw = item.get("languages_raw") or (", ".join(languages) if languages else "")

                    taxonomy_categories = item.get("taxonomy_categories") or split_values(item.get("taxonomy_raw") or "")
                    taxonomy_raw = item.get("taxonomy_raw") or (", ".join(taxonomy_categories) if taxonomy_categories else "")

                    test_type = map_test_type(f"{assessment_type} {name} {description}")

                    category = infer_category(name, description, assessment_type)

                    entity_id = str(item.get("entity_id") or item.get("id") or "")

                    # Skip records missing required fields (after attempted enrichment)
                    if not name or not url_norm:
                        logger.info("Skipping JSON entry due to missing name or url (entity_id=%s)", entity_id)
                        return None

                    retrieval_text = (
                        f"Assessment Name: {name}\n\n"
                        f"Description:\n{description}\n\n"
                        f"Skills:\n{', '.join(skills)}\n\n"
                        f"Job Levels:\n{', '.join(job_levels)}\n\n"
                        f"Categories:\n{', '.join(taxonomy_categories)}\n\n"
                        f"Test Type:\n{test_type}\n\n"
                        f"Remote:\n{remote}"
                    )

                    record = {
                        "entity_id": entity_id,
                        "name": name,
                        "link": url_norm,
                        "url": url_norm,
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "description": description,
                        "assessment_type": assessment_type,
                        "test_type": test_type,
                        "category": category,
                        "job_levels": job_levels,
                        "job_levels_raw": job_levels_raw,
                        "skills_measured": skills,
                        "skills_raw": skills_raw,
                        "languages": languages,
                        "languages_raw": languages_raw,
                        "taxonomy_categories": taxonomy_categories,
                        "taxonomy_raw": taxonomy_raw,
                        "duration": duration,
                        "duration_raw": duration_raw,
                        "duration_minutes": duration_minutes,
                        "remote": remote,
                        "adaptive": adaptive,
                        "remote_testing": remote_raw,
                        "adaptive_support": adaptive_raw,
                        "status": item.get("status") or "ok",
                        "retrieval_text": clean_text(retrieval_text),
                    }

                    # Validate shape with pydantic
                    Assessment(
                        name=record["name"],
                        url=record["link"],
                        description=record["description"],
                        assessment_type=record["assessment_type"],
                        duration=record["duration"],
                        remote_testing=record["remote"],
                        adaptive_support=record["adaptive"],
                        job_levels=record["job_levels"],
                        skills_measured=record["skills_measured"],
                        languages=record["languages"],
                        category=record["category"],
                    )

                    return record

                except Exception as exc:
                    logger.warning("Skipping malformed JSON entry (entity_id=%s): %s", item.get("entity_id"), exc)
                    return None

            for item in json_data:
                if not isinstance(item, dict):
                    continue

                rec = normalize_from_item(item)
                if rec:
                    assessments.append(rec)

            logger.info("Processed %d valid assessments from JSON", len(assessments))

            # write output and return
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text(json.dumps(assessments, indent=2, ensure_ascii=False), encoding="utf-8")

            logger.info("Saved %d assessments to %s", len(assessments), output_path)

            return assessments

    except requests.RequestException:
        # Fall back to discovery below if the initial fetch fails
        pass

    # Default: perform discovery + per-page parsing
    urls = discover_product_urls(session, max_pages=max_pages)

    logger.info("Discovered %d product URLs", len(urls))

    seen = set()

    for url in urls:

        normalized = normalize_url(url)

        if normalized in seen:
            continue

        seen.add(normalized)

        try:
            soup = fetch(session, normalized)

        except requests.RequestException as exc:

            logger.warning(
                "Failed scraping %s: %s",
                normalized,
                exc,
            )

            continue

        record = parse_product_page(normalized, soup)

        if record:
            assessments.append(record)

        time.sleep(0.15)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            assessments,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved %d assessments to %s",
        len(assessments),
        output_path,
    )

    return assessments


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Scrape SHL Product Catalog"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--use-json-catalog",
        action="store_true",
        help="Use the catalog JSON fast-path (don't fetch individual product pages)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    scrape_catalog(
        output_path=args.output,
        max_pages=args.max_pages,
        use_json_catalog=args.use_json_catalog,
    )


if __name__ == "__main__":
    main()