"""
main.py — CLI entry point for the Job Skills Scraper pipeline.

Usage
-----
    python main.py --job "Data Engineer" --industry "Fintech" --location "London"
    python main.py --job "ML Engineer" --industry "Healthcare" --max-pages 3
    python main.py --job "Product Manager" --industry "SaaS"

Environment variables required (set in .env or shell)
-------------------------------------------------------
    SERP_API_KEY  — from https://serpapi.com

LLM
---
    Uses Ollama locally with model: kimi-k2.5:cloud
    Make sure Ollama is running before executing the pipeline.

Output
------
    output/skills_report.md   — human-readable Markdown report
    output/skills_report.json — machine-readable JSON for downstream use
"""

import argparse
import logging
import sys
import time

from config import config
from scraper import fetch_jobs
from extractor import extract_skills
from analyzer import analyse
from report import save_report


# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Google Jobs and generate a skills demand report."
    )
    parser.add_argument("--job",       required=True, help="Job title to search (e.g. 'Data Engineer')")
    parser.add_argument("--industry",  required=True, help="Industry filter (e.g. 'Fintech')")
    parser.add_argument("--location",  default="",    help="Optional location (e.g. 'London')")
    parser.add_argument("--max-pages", type=int, default=None,
                        help=f"Override max pages (default: {config.max_pages})")
    return parser.parse_args()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run(job_title: str, industry: str, location: str = "", max_pages: int | None = None) -> None:
    start_time = time.perf_counter()

    # ── 0. Validate config ────────────────────────────────────────────────────
    try:
        config.validate()
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    if max_pages is not None:
        config.max_pages = max_pages

    logger.info("=" * 60)
    logger.info("Job Skills Scraper — starting pipeline")
    logger.info("  Job title : %s", job_title)
    logger.info("  Industry  : %s", industry)
    logger.info("  Location  : %s", location or "(not specified)")
    logger.info("=" * 60)

    # ── 1. Fetch listings ─────────────────────────────────────────────────────
    logger.info("STEP 1 — Fetching job listings from Google Jobs via SerpAPI…")
    listings = fetch_jobs(job_title, industry, location)

    if not listings:
        logger.error("No listings returned. Check your query or SerpAPI key.")
        sys.exit(1)

    # ── 2. Extract skills ─────────────────────────────────────────────────────
    logger.info("STEP 2 — Extracting skills with Ollama/%s (%d listings)…", config.ollama_model, len(listings))
    extracted = extract_skills(listings)

    # ── 3. Analyse ────────────────────────────────────────────────────────────
    logger.info("STEP 3 — Aggregating and ranking skills…")
    report = analyse(extracted, job_title, industry)

    # ── 4. Save report ────────────────────────────────────────────────────────
    logger.info("STEP 4 — Saving report…")
    md_path, json_path = save_report(report)

    elapsed = time.perf_counter() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("  Markdown : %s", md_path)
    logger.info("  JSON     : %s", json_path)
    logger.info("=" * 60)

    # ── Quick summary to stdout ───────────────────────────────────────────────
    print(f"\n✅  Report ready: {md_path}\n")
    print(f"Top 5 in-demand skills for '{job_title}' in {industry}:")
    for i, skill in enumerate(report.top_hard_skills[:5], 1):
        bar = "█" * round(skill.percentage / 5)
        print(f"  {i}. {skill.name:<28} {skill.percentage:>5}%  {bar}")

    if report.top_tools:
        print(f"\nTop 5 tools/frameworks:")
        for i, skill in enumerate(report.top_tools[:5], 1):
            bar = "█" * round(skill.percentage / 5)
            print(f"  {i}. {skill.name:<28} {skill.percentage:>5}%  {bar}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(
        job_title=args.job,
        industry=args.industry,
        location=args.location,
        max_pages=args.max_pages,
    )