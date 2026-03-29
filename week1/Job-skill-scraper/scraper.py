"""
scraper.py — Fetches job listings from Google Jobs via SerpAPI.

Industry-standard practices included:
  • Exponential back-off on rate-limit / server errors
  • Pagination via `start` offset
  • Typed dataclass for listings
  • Deduplication by listing ID before returning
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import config

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class JobListing:
    listing_id: str
    title: str
    company: str
    location: str
    description: str
    posted_at: Optional[str] = None
    job_link: Optional[str] = None
    salary: Optional[str] = None
    job_type: Optional[str] = None
    highlights: list[str] = field(default_factory=list)


# ── HTTP client with retry ────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Mounts a retry adapter: 3 retries on 429/500/502/503/504."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,           # 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


_session = _build_session()


# ── Core fetch logic ──────────────────────────────────────────────────────────

def _fetch_page(query: str, next_page_token: Optional[str] = None) -> dict:
    """
    Call SerpAPI for a single page of Google Jobs results.

    Parameters
    ----------
    query:           The full search query string (e.g. "Data Engineer fintech London")
    next_page_token: Pagination token from the previous response (None for first page)
    """
    params = {
        "engine": "google_jobs",
        "q": query,
        "api_key": config.serp_api_key,
        "hl": "en",          # language
        "gl": "us",          # country (change as needed)
    }
    if next_page_token:
        params["next_page_token"] = next_page_token

    logger.debug("GET %s  next_page_token=%s", config.serp_base_url, next_page_token)
    response = _session.get(config.serp_base_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _parse_listing(raw: dict) -> JobListing:
    """Convert a raw SerpAPI job result dict into a typed JobListing."""

    # Highlights are bullet-point descriptions SerpAPI sometimes returns
    highlights: list[str] = []
    for block in raw.get("job_highlights", []):
        for item in block.get("items", []):
            highlights.append(item)

    # Salary may live in detected_extensions or extensions
    extensions = raw.get("detected_extensions", {})
    salary = extensions.get("salary") or raw.get("salary")

    # Some job types come from extensions
    job_type = extensions.get("schedule_type") or None

    return JobListing(
        listing_id=raw.get("job_id", raw.get("title", "") + raw.get("company_name", "")),
        title=raw.get("title", "Unknown"),
        company=raw.get("company_name", "Unknown"),
        location=raw.get("location", ""),
        description=raw.get("description", ""),
        posted_at=raw.get("detected_extensions", {}).get("posted_at"),
        job_link=raw.get("share_link") or raw.get("apply_options", [{}])[0].get("link"),
        salary=salary,
        job_type=job_type,
        highlights=highlights,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_jobs(job_title: str, industry: str, location: str = "") -> list[JobListing]:
    """
    Fetch up to config.max_pages * config.results_per_page unique job
    listings for the given job_title within industry (and optional location).

    Returns a deduplicated list of JobListing objects sorted by relevance
    (the order SerpAPI returns them).

    Example
    -------
    listings = fetch_jobs("Data Engineer", "Fintech", location="London")
    """
    query_parts = [job_title, industry]
    if location:
        query_parts.append(location)
    query = " ".join(query_parts)

    logger.info("Fetching Google Jobs for: %r  (up to %d pages)", query, config.max_pages)

    seen_ids: set[str] = set()
    listings: list[JobListing] = []
    next_page_token: Optional[str] = None

    for page in range(config.max_pages):
        try:
            data = _fetch_page(query, next_page_token)
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else "(no body)"
            logger.error("HTTP error on page %d: %s\n  Response body: %s", page, exc, body)
            break

        jobs_on_page = data.get("jobs_results", [])
        if not jobs_on_page:
            logger.info("No more results on page %d, stopping early.", page + 1)
            break

        for raw in jobs_on_page:
            listing = _parse_listing(raw)
            if listing.listing_id not in seen_ids:
                seen_ids.add(listing.listing_id)
                listings.append(listing)

        logger.info("  Page %d → %d listings (total so far: %d)", page + 1, len(jobs_on_page), len(listings))

        # Get the next page token for subsequent requests
        next_page_token = (
            data.get("serpapi_pagination", {}).get("next_page_token")
            or data.get("next_page_token")
        )
        if not next_page_token:
            logger.info("No next_page_token returned — reached last page.")
            break

        # Respect rate limits with a polite delay between pages
        time.sleep(config.request_delay_seconds)

    logger.info("Done. Total unique listings fetched: %d", len(listings))
    return listings