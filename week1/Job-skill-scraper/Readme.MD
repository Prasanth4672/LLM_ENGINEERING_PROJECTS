# Job Skills Scraper
**Google Jobs via SerpAPI + Claude — skills demand analysis pipeline**

Fetches real job listings for any title + industry, extracts structured skill requirements using Claude, and generates a ranked skills demand report in Markdown and JSON.

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys
```bash
# Get a free SerpAPI key at: https://serpapi.com  (100 free searches/month)
# Get an Anthropic key at: https://console.anthropic.com

export SERP_API_KEY="your-serpapi-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Run the pipeline
```bash
# Basic
python main.py --job "Data Engineer" --industry "Fintech"

# With location
python main.py --job "ML Engineer" --industry "Healthcare" --location "London"

# Limit pages (10 listings per page)
python main.py --job "Product Manager" --industry "SaaS" --max-pages 2
```

---

## Output
```
output/
  skills_report.md     ← human-readable Markdown with tables + bar charts
  skills_report.json   ← machine-readable JSON for dashboards / further analysis
```

**Report sections:**
1. Hard Skills (technical)
2. Tools, Platforms & Frameworks
3. Soft Skills
4. Certifications & Qualifications
5. Experience Requirements
6. Education Requirements
7. Skill Co-occurrence matrix
8. Top Hiring Companies

---

## Project Structure

| File | Responsibility |
|------|---------------|
| `config.py` | All settings and API keys |
| `scraper.py` | SerpAPI paginated fetcher with retry |
| `extractor.py` | Claude-powered structured skill extraction |
| `analyzer.py` | Frequency aggregation, ranking, co-occurrence |
| `report.py` | Markdown + JSON report generator |
| `main.py` | CLI entry point orchestrating the pipeline |

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `max_pages` | 5 | Pages to fetch (×10 listings each) |
| `request_delay_seconds` | 1.2 | Delay between paginated requests |
| `extraction_batch_size` | 5 | Listings per Claude API call |
| `top_skills_count` | 20 | Skills surfaced in report |
| `min_skill_frequency` | 2 | Suppress single-mention skills |

---

## Extending the pipeline

**Add a new output format** — implement a function in `report.py` that accepts `AnalysisReport` and call it from `main.py`.

**Schedule recurring reports** — wrap `main.run()` in a cron job or GitHub Action. Pass `--max-pages 1` for a lightweight daily diff.

**Add a sector comparison** — call `run()` multiple times with different industries, collect the resulting JSON files, and diff them to see skill gaps between sectors.

**Export to a spreadsheet** — read `output/skills_report.json` and use `openpyxl` or `pandas` to produce an Excel workbook.

---

## Cost estimates

| Listings | SerpAPI calls | Claude tokens (est.) | Approximate cost |
|----------|--------------|----------------------|-----------------|
| 10 | 1 | ~15k | ~$0.02 |
| 50 | 5 | ~75k | ~$0.10 |
| 100 | 10 | ~150k | ~$0.20 |

SerpAPI free tier: 100 searches/month.
Anthropic Sonnet pricing: ~$3/MTok input, ~$15/MTok output.