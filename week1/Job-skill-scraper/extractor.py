"""
extractor.py — Uses a local Ollama LLM to extract structured skill data
from raw job descriptions.

Design choices:
  • Batches listings to reduce LLM round-trips
  • Asks the model to return strict JSON (no markdown fences)
  • Categorises skills into hard / soft / tools / certifications
  • Gracefully handles partial failures in a batch
"""

import json
import logging
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config import config
from scraper import JobListing

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ExtractedSkills:
    listing_id: str
    job_title: str
    company: str
    hard_skills: list[str] = field(default_factory=list)      # Technical, domain-specific
    soft_skills: list[str] = field(default_factory=list)      # Interpersonal, communication
    tools: list[str] = field(default_factory=list)            # Software, platforms, frameworks
    certifications: list[str] = field(default_factory=list)   # Required / preferred certs
    experience_years: str = ""                                 # e.g. "3–5 years"
    education: str = ""                                        # e.g. "Bachelor's in CS"
    raw_summary: str = ""                                      # One-line job summary


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise job-market analyst.
Your task is to extract structured skill requirements from job descriptions.

Rules:
- Respond ONLY with a valid JSON array — no markdown, no commentary, no backticks.
- Each element in the array corresponds to one job in the input.
- Normalise skill names: use lowercase, remove trailing punctuation,
  collapse duplicates (e.g. "Python 3", "python" → "python").
- Separate tools/frameworks from conceptual skills:
  e.g. "machine learning" → hard_skill; "scikit-learn" → tool.
- Keep each list concise (max 10 items each). Surface the most prominent skills.
- If a field has no data, use an empty list or empty string."""

USER_TEMPLATE = """Extract skills from the following {n} job listing(s).

{listings_block}

Return a JSON array of {n} objects, one per listing, in the same order.
Each object must have these keys:
  listing_id       (string — copy exactly from input)
  hard_skills      (array of strings)
  soft_skills      (array of strings)
  tools            (array of strings)
  certifications   (array of strings)
  experience_years (string, e.g. "3-5 years", or "")
  education        (string, e.g. "Bachelor's in Computer Science", or "")
  raw_summary      (string — one sentence describing the role)"""


def _format_listings_block(batch: list[JobListing]) -> str:
    parts = []
    for listing in batch:
        description_excerpt = listing.description[:2000]   # Truncate very long descriptions
        highlights_text = "\n".join(f"  • {h}" for h in listing.highlights[:10])
        parts.append(
            f"---\n"
            f"listing_id: {listing.listing_id}\n"
            f"title: {listing.title}\n"
            f"company: {listing.company}\n"
            f"description:\n{description_excerpt}\n"
            + (f"highlights:\n{highlights_text}\n" if highlights_text else "")
        )
    return "\n".join(parts)


# ── Ollama client ─────────────────────────────────────────────────────────────

_llm = ChatOllama(model=config.ollama_model)


def _call_llm(listings_block: str, n: int) -> list[dict]:
    """Send a batch to the Ollama LLM and parse the returned JSON array."""
    user_message = USER_TEMPLATE.format(n=n, listings_block=listings_block)

    response = _llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw_text = response.content.strip()

    # Strip accidental markdown code fences if the model adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    return json.loads(raw_text)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_skills(listings: list[JobListing]) -> list[ExtractedSkills]:
    """
    Extract structured skills from all listings using Claude.
    Processes listings in batches of config.extraction_batch_size.

    Returns one ExtractedSkills per listing (failures produce empty objects).
    """
    results: list[ExtractedSkills] = []
    batch_size = config.extraction_batch_size

    for i in range(0, len(listings), batch_size):
        batch = listings[i : i + batch_size]
        logger.info(
            "Extracting skills: listings %d–%d of %d",
            i + 1, min(i + batch_size, len(listings)), len(listings)
        )

        listings_block = _format_listings_block(batch)
        try:
            parsed = _call_llm(listings_block, len(batch))
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Extraction failed for batch starting at %d: %s", i, exc)
            # Insert empty results so the pipeline can continue
            for listing in batch:
                results.append(ExtractedSkills(
                    listing_id=listing.listing_id,
                    job_title=listing.title,
                    company=listing.company,
                ))
            continue

        # Map Claude's output back to typed objects
        id_to_listing = {l.listing_id: l for l in batch}
        for item in parsed:
            lid = item.get("listing_id", "")
            orig = id_to_listing.get(lid)
            results.append(ExtractedSkills(
                listing_id=lid,
                job_title=orig.title if orig else item.get("title", ""),
                company=orig.company if orig else "",
                hard_skills=item.get("hard_skills", []),
                soft_skills=item.get("soft_skills", []),
                tools=item.get("tools", []),
                certifications=item.get("certifications", []),
                experience_years=item.get("experience_years", ""),
                education=item.get("education", ""),
                raw_summary=item.get("raw_summary", ""),
            ))

    logger.info("Skill extraction complete. %d records processed.", len(results))
    return results