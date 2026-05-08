from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from app.models.schemas import Assessment

logger = logging.getLogger(__name__)

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "app" / "data" / "shl_catalog.json"
REQUEST_TIMEOUT = 20
USER_AGENT = "shl-assessment-recommender/1.0"


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def split_values(value: str) -> list[str]:
    if not value:
        return []
    pieces = re.split(r",|;|\||\n", value)
    return [clean_text(piece) for piece in pieces if clean_text(piece)]


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + "/", "", "", ""))


def with_page(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def fetch(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def is_individual_solution_link(href: str, text: str) -> bool:
    lowered = f"{href} {text}".lower()
    if "job-solutions" in lowered or "job solution" in lowered:
        return False
    if "/solutions/products/product-catalog/" not in href:
        return False
    return href.rstrip("/") != CATALOG_URL.rstrip("/")


def discover_product_urls(session: requests.Session, max_pages: int = 50) -> list[str]:
    discovered: list[str] = []
    seen_pages: set[str] = set()
    for page in range(1, max_pages + 1):
        url = CATALOG_URL if page == 1 else with_page(CATALOG_URL, page)
        if url in seen_pages:
            break
        seen_pages.add(url)
        try:
            soup = fetch(session, url)
        except requests.RequestException as exc:
            logger.warning("Stopping pagination at %s: %s", url, exc)
            break

        page_links = []
        for anchor in soup.select("a[href]"):
            href = urljoin(CATALOG_URL, anchor.get("href", ""))
            text = clean_text(anchor.get_text(" "))
            if is_individual_solution_link(href, text):
                page_links.append(normalize_url(href))

        before = len(discovered)
        for link in page_links:
            if link not in discovered:
                discovered.append(link)
        if len(discovered) == before and page > 1:
            break
        time.sleep(0.15)
    return discovered


def value_from_labels(soup: BeautifulSoup, labels: tuple[str, ...]) -> str:
    label_pattern = re.compile("|".join(re.escape(label) for label in labels), re.IGNORECASE)
    for node in soup.find_all(string=label_pattern):
        parent = node.parent
        if not parent:
            continue
        row = parent.find_parent(["tr", "li", "div"])
        if row:
            text = clean_text(row.get_text(" "))
            text = label_pattern.sub("", text, count=1)
            text = clean_text(text.strip(" :-"))
            if text:
                return text
    return ""


def infer_category(name: str, description: str, assessment_type: str) -> str:
    text = f"{name} {description} {assessment_type}".lower()
    if any(term in text for term in ("personality", "opq", "behavioral", "behavioural", "motivation")):
        return "Personality"
    if any(term in text for term in ("cognitive", "ability", "reasoning", "aptitude", "numerical", "verbal", "inductive")):
        return "Cognitive"
    if any(term in text for term in ("coding", "programming", "technical", "java", "python", "sql", "software")):
        return "Technical"
    if any(term in text for term in ("simulation", "situational", "judgement", "judgment")):
        return "Situational Judgement"
    return assessment_type or "Individual Test Solution"


def parse_product_page(url: str, soup: BeautifulSoup) -> Assessment | None:
    name = clean_text((soup.find("h1") or soup.find("title") or soup).get_text(" "))
    if not name:
        return None
    for suffix in (" | SHL", " - SHL"):
        name = name.replace(suffix, "")
    description_node = soup.select_one("meta[name='description']")
    description = clean_text(description_node.get("content") if description_node else "")
    if not description:
        first_para = soup.find("p")
        description = clean_text(first_para.get_text(" ") if first_para else "")

    assessment_type = value_from_labels(soup, ("Assessment type", "Test type", "Type"))
    duration = value_from_labels(soup, ("Duration", "Approximate completion time", "Time"))
    remote = value_from_labels(soup, ("Remote testing", "Remote", "Online testing"))
    adaptive = value_from_labels(soup, ("Adaptive", "Adaptive support", "IRT"))
    job_levels = split_values(value_from_labels(soup, ("Job levels", "Job level", "Levels")))
    skills = split_values(value_from_labels(soup, ("Skills measured", "Knowledge, skills, abilities", "Measures")))
    languages = split_values(value_from_labels(soup, ("Languages", "Language support")))
    category = infer_category(name, description, assessment_type)

    try:
        return Assessment(
            name=name,
            url=normalize_url(url),
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
        logger.warning("Malformed product page skipped %s: %s", url, exc)
        return None


def scrape_catalog(output_path: Path = DEFAULT_OUTPUT, max_pages: int = 50) -> list[Assessment]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    urls = discover_product_urls(session, max_pages=max_pages)
    logger.info("Discovered %d candidate Individual Test Solution URLs", len(urls))
    assessments: list[Assessment] = []
    seen: set[str] = set()
    for url in urls:
        normalized = normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            soup = fetch(session, normalized)
        except requests.RequestException as exc:
            logger.warning("Skipping %s: %s", normalized, exc)
            continue
        assessment = parse_product_page(normalized, soup)
        if assessment:
            assessments.append(assessment)
        time.sleep(0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([assessment.model_dump() for assessment in assessments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return assessments


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL Individual Test Solutions catalog")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pages", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assessments = scrape_catalog(args.output, max_pages=args.max_pages)
    logger.info("Wrote %d assessments to %s", len(assessments), args.output)


if __name__ == "__main__":
    main()

