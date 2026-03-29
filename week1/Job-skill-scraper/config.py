"""
config.py — Central configuration for the Job Skills Scraper pipeline.
All API keys and tunable parameters live here.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class Config:
    # ── API Keys ────────────────────────────────────────────────────────────
    serp_api_key: str = field(
        default_factory=lambda: os.environ.get("SERP_API_KEY", "").strip().strip("'\"")
    )

    # ── SerpAPI / Google Jobs settings ──────────────────────────────────────
    serp_base_url: str = "https://serpapi.com/search.json"
    results_per_page: int = 10          # Google Jobs returns up to 10 per page
    max_pages: int = 5                  # Fetch up to 50 listings total
    request_delay_seconds: float = 1.2  # Polite delay between paginated requests

    # ── Ollama / LLM extraction settings ────────────────────────────────────
    ollama_model: str = "kimi-k2.5:cloud"
    extraction_batch_size: int = 5      # Listings to send per LLM call

    # ── Analyser settings ────────────────────────────────────────────────────
    top_skills_count: int = 20          # Skills to surface in the report
    min_skill_frequency: int = 2        # Suppress skills appearing only once

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "output"
    report_filename: str = "skills_report.md"

    def validate(self) -> None:
        if not self.serp_api_key:
            raise EnvironmentError(
                "SERP_API_KEY is not set. "
                "Add it to the .env file or export it as an environment variable. "
                "Get a free key at https://serpapi.com"
            )


# Singleton — import this everywhere
config = Config()