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
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

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

        try:
            soup = fetch(session, url)

        except requests.RequestException as exc:
            logger.warning("Failed page %s: %s", url, exc)
            continue

        page_links = []

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

    if any(
        word in lowered
        for word in [
            "personality",
            "behavior",
            "behaviour",
            "opq",
        ]
    ):
        return "P"

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

    retrieval_text = f"""
    Assessment Name: {name}

    Description:
    {description}

    Assessment Type:
    {assessment_type}

    Skills:
    {", ".join(skills)}

    Job Levels:
    {", ".join(job_levels)}

    Languages:
    {", ".join(languages)}

    Category:
    {category}

    Taxonomy:
    {", ".join(taxonomy_categories)}
    """

    # ENTITY ID

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
) -> list[dict]:

    session = create_session()

    urls = discover_product_urls(
        session,
        max_pages=max_pages,
    )

    logger.info(
        "Discovered %d product URLs",
        len(urls),
    )

    assessments = []

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

        record = parse_product_page(
            normalized,
            soup,
        )

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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    scrape_catalog(
        output_path=args.output,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()