"""
analyzer.py — Aggregates extracted skills across all listings,
ranks them by frequency, groups them into categories, and
surfaces trend signals.
"""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field

from config import config
from extractor import ExtractedSkills

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RankedSkill:
    name: str
    count: int
    percentage: float       # % of listings that mention this skill
    category: str           # hard | soft | tools | certifications


@dataclass
class AnalysisReport:
    job_title: str
    industry: str
    total_listings_analysed: int

    top_hard_skills: list[RankedSkill] = field(default_factory=list)
    top_soft_skills: list[RankedSkill] = field(default_factory=list)
    top_tools: list[RankedSkill] = field(default_factory=list)
    top_certifications: list[RankedSkill] = field(default_factory=list)

    common_experience_ranges: list[tuple[str, int]] = field(default_factory=list)
    common_education: list[tuple[str, int]] = field(default_factory=list)

    # Skills that co-occur most often (pairing signal)
    skill_co_occurrences: list[tuple[str, str, int]] = field(default_factory=list)

    # Top 5 companies hiring and how many listings each had
    top_hiring_companies: list[tuple[str, int]] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(skill: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    skill = skill.lower().strip()
    skill = re.sub(r"[^\w\s\+\#\.]", "", skill)
    skill = re.sub(r"\s+", " ", skill)
    return skill


def _rank(counter: Counter, total: int, category: str, top_n: int) -> list[RankedSkill]:
    skills = []
    for name, count in counter.most_common(top_n):
        if count < config.min_skill_frequency:
            continue
        skills.append(RankedSkill(
            name=name,
            count=count,
            percentage=round(count / total * 100, 1),
            category=category,
        ))
    return skills


def _co_occurrence(extracted: list[ExtractedSkills], top_n: int = 10) -> list[tuple[str, str, int]]:
    """
    Find pairs of hard_skills + tools that appear together across listings.
    Uses a simple O(n * k²) approach — fine for up to ~500 listings.
    """
    pair_counter: Counter = Counter()
    for e in extracted:
        all_skills = [_normalise(s) for s in (e.hard_skills + e.tools)]
        unique = sorted(set(all_skills))
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                pair_counter[(a, b)] += 1
    return [(a, b, c) for (a, b), c in pair_counter.most_common(top_n)]


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(
    extracted: list[ExtractedSkills],
    job_title: str,
    industry: str,
) -> AnalysisReport:
    """
    Aggregate and rank skills across all ExtractedSkills records.

    Parameters
    ----------
    extracted:  Output of extractor.extract_skills()
    job_title:  The original search job title (for labelling)
    industry:   The original search industry (for labelling)
    """
    total = len(extracted)
    logger.info("Analysing %d listings for '%s' in '%s'", total, job_title, industry)

    hard_counter: Counter = Counter()
    soft_counter: Counter = Counter()
    tools_counter: Counter = Counter()
    cert_counter: Counter = Counter()
    exp_counter: Counter = Counter()
    edu_counter: Counter = Counter()
    company_counter: Counter = Counter()

    for e in extracted:
        for s in e.hard_skills:
            hard_counter[_normalise(s)] += 1
        for s in e.soft_skills:
            soft_counter[_normalise(s)] += 1
        for s in e.tools:
            tools_counter[_normalise(s)] += 1
        for s in e.certifications:
            cert_counter[_normalise(s)] += 1
        if e.experience_years:
            exp_counter[e.experience_years.strip()] += 1
        if e.education:
            edu_counter[e.education.strip()] += 1
        if e.company:
            company_counter[e.company.strip()] += 1

    top_n = config.top_skills_count

    report = AnalysisReport(
        job_title=job_title,
        industry=industry,
        total_listings_analysed=total,
        top_hard_skills=_rank(hard_counter, total, "hard", top_n),
        top_soft_skills=_rank(soft_counter, total, "soft", top_n),
        top_tools=_rank(tools_counter, total, "tools", top_n),
        top_certifications=_rank(cert_counter, total, "certifications", top_n),
        common_experience_ranges=exp_counter.most_common(5),
        common_education=edu_counter.most_common(5),
        skill_co_occurrences=_co_occurrence(extracted, top_n=10),
        top_hiring_companies=company_counter.most_common(10),
    )

    logger.info(
        "Analysis complete. Top hard skill: %s (%s%%)",
        report.top_hard_skills[0].name if report.top_hard_skills else "n/a",
        report.top_hard_skills[0].percentage if report.top_hard_skills else 0,
    )
    return report