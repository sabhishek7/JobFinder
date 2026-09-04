"""
Filtering, deduplication, ranking, and salary-tagging logic.

Design decisions (so this stays maintainable):
- Everything works off the `raw_content` field (job description text) plus title/location.
- Salary detection is best-effort text matching. Many Indian job posts don't
  disclose salary at all -- those aren't dropped, just marked "Not disclosed"
  so you can judge for yourself instead of losing real openings.
- "Freshness" cutoff is configurable (default 7 days per your preference).
"""

import re
import datetime
from difflib import SequenceMatcher

# ---- Role matching ----
# Deliberately narrow: require a Java/Spring/backend or web-stack signal, not
# just "software engineer" by itself.
ROLE_KEYWORDS = [
    "java developer", "java engineer", "java software engineer",
    "backend developer", "backend engineer", "software developer",
    "spring", "spring boot", "springboot", "spring framework",
    "hibernate", "jpa", "microservices", "rest api", "restful api",
    "mysql", "sql", "database",
    "html", "html5", "javascript", "js",
    "full stack", "fullstack", "full-stack",
]

# If a posting is generic "software engineer" but ALSO contains one of these,
# it's very likely not a web role -- exclude even though ROLE_KEYWORDS might
# not fire (belt-and-suspenders against titles like "Software Engineer,
# Android").
NON_WEB_EXCLUDE_KEYWORDS = [
    "android", "ios", "kotlin", "swift", "objective-c",
    "embedded", "firmware", "site reliability", "devops",
    "data engineer", "machine learning engineer", "ml engineer",
    "golang", "scala", "rust ", " c++", "distributed systems",
    "infrastructure engineer", "network engineer",
]

REMOTE_KEYWORDS = ["remote", "work from home", "wfh", "distributed"]
HYBRID_KEYWORDS = ["hybrid"]

# ---- Location matching ----
INDIA_KEYWORDS = [
    "india", "bangalore", "bengaluru", "hyderabad", "pune", "mumbai",
    "delhi", "gurgaon", "gurugram", "noida", "chennai", "kolkata",
]
GLOBAL_OPEN_KEYWORDS = [
    "worldwide", "remote - global", "remote (global)", "work from anywhere",
    "remote - anywhere", "anywhere in the world", "remote, global",
]
US_ONLY_SIGNALS = [
    "remote - usa", "remote - united states", "remote (us)", "remote, us",
    "us only", "us-based", "us based", "must be based in the united states",
    "authorized to work in the united states", "us work authorization",
    "menlo park", "new york, ny", "san francisco", "seattle, wa",
    "austin, tx", "remote - us", "remote-us",
]

# ---- Experience matching ----
# Extract explicit "N+ years" style requirements and threshold numerically
# instead of matching fixed phrases (which missed "2+ years", "3+ years",
# "8+ years" entirely in v1).
YEARS_PATTERN = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?years?[^.]{0,40}?experience", re.IGNORECASE
)
MAX_YEARS_ALLOWED = 1  # target fresher / entry-level / 0-1 year roles

JUNIOR_KEYWORDS = [
    "fresher", "fresh graduate", "graduate", "new grad", "new graduate",
    "entry level", "entry-level", "junior", "trainee", "internship",
    "0-1 year", "0 to 1 year", "1 year", "1+ year",
]
SENIOR_TITLE_KEYWORDS = [
    "senior", "staff", "principal", "lead ", "director", " vp ", "head of",
    "engineering manager", "manager,",
]

SALARY_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)?\s?([\d,.]+)\s?(lpa|lakh|lakhs)", re.IGNORECASE
)


def matches_role(job: dict) -> bool:
    text = f"{job.get('title','')} {job.get('raw_content','')}".lower()

    if not any(kw in text for kw in ROLE_KEYWORDS):
        return False

    # Even with a role keyword hit, drop it if strong unrelated signals dominate
    # and there's no direct Java/web-stack mention in the title itself.
    title = job.get("title", "").lower()
    if any(kw in text for kw in NON_WEB_EXCLUDE_KEYWORDS) and not any(
        kw in title for kw in ["java", "spring", "backend", "full stack", "fullstack", "software developer"]
    ):
        return False

    return True


def extract_min_years_required(job: dict) -> int | None:
    """Returns the highest explicit 'N years experience' figure found, or None
    if the posting doesn't state a number at all."""
    text = f"{job.get('title','')} {job.get('raw_content','')}"
    matches = YEARS_PATTERN.findall(text)
    if not matches:
        return None
    return max(int(m) for m in matches)


def matches_experience(job: dict) -> bool:
    """
    Primary signal: explicit numeric year requirements (e.g. "8+ years",
    "3+ years of professional experience"). If the highest such number found
    exceeds MAX_YEARS_ALLOWED, reject -- this catches cases plain keyword
    matching missed (v1 only checked for a few hardcoded phrasings and let
    "2+ years", "3+ years", "8+ years" straight through).

    Fallback when no number is stated: use title-level seniority wording.
    """
    min_years = extract_min_years_required(job)
    if min_years is not None:
        return min_years <= MAX_YEARS_ALLOWED

    text = f"{job.get('title','')} {job.get('raw_content','')}".lower()
    if any(kw in text for kw in JUNIOR_KEYWORDS):
        return True

    title = job.get("title", "").lower()
    if any(kw in title for kw in SENIOR_TITLE_KEYWORDS):
        return False
    return True


def classify_work_mode(job: dict) -> str:
    text = f"{job.get('title','')} {job.get('location','')} {job.get('raw_content','')}".lower()
    if any(kw in text for kw in REMOTE_KEYWORDS):
        return "Remote"
    if any(kw in text for kw in HYBRID_KEYWORDS):
        return "Hybrid"
    return "On-site"


def passes_location(job: dict) -> bool:
    """
    Hard filter -- drops jobs that clearly aren't reachable from India:
      - Explicitly US/EU-only remote postings (e.g. "Remote - USA")
      - On-site postings at a non-India office (e.g. "Menlo Park, CA")
    Everything else (India-based, explicitly worldwide-remote, or remote
    with no stated geo restriction) is kept -- see location_tier() for how
    the unspecified-remote case gets ranked below the clearer matches.
    """
    text = f"{job.get('location','')} {job.get('raw_content','')}".lower()
    is_india = any(kw in text for kw in INDIA_KEYWORDS)

    if any(kw in text for kw in US_ONLY_SIGNALS) and not is_india:
        return False

    if job.get("work_mode") == "On-site" and not is_india:
        return False

    return True


def location_tier(job: dict) -> int:
    """Lower = ranked higher. Used by rank() as a sort key."""
    text = f"{job.get('location','')} {job.get('raw_content','')}".lower()
    if any(kw in text for kw in INDIA_KEYWORDS):
        return 0
    if any(kw in text for kw in GLOBAL_OPEN_KEYWORDS):
        return 1
    return 2  # remote with no stated geo restriction either way


def extract_salary(job: dict) -> str:
    if job.get("salary_min") and job.get("salary_max"):
        return f"${job['salary_min']:,} - ${job['salary_max']:,} (USD, RemoteOK)"
    text = job.get("raw_content", "") or ""
    match = SALARY_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return "Not disclosed"

def meets_salary_floor(job: dict, min_lpa: float = 10.0) -> bool:
    """
    Best-effort filter: only DROPS a job if a salary is explicitly stated
    AND it's clearly below the floor. If salary isn't disclosed, we keep
    the job (can't verify, but shouldn't lose real openings over it).
    """
    salary_str = job.get("salary_tag", "")
    if salary_str == "Not disclosed":
        return True
    numbers = re.findall(r"[\d.]+", salary_str.replace(",", ""))
    if not numbers:
        return True
    try:
        value = float(numbers[0])
    except ValueError:
        return True
    if "lakh" in salary_str.lower() or "lpa" in salary_str.lower():
        return value >= min_lpa
    return True  # unit unclear (e.g. USD RemoteOK) -- don't drop, just show as-is


def is_fresh(job: dict, max_age_days: int = 7) -> bool:
    posted = job.get("posted")
    if not posted:
        return True  # unknown date -- keep rather than drop
    try:
        posted_date = datetime.datetime.strptime(posted, "%Y-%m-%d").date()
    except ValueError:
        return True
    age = (datetime.date.today() - posted_date).days
    return age <= max_age_days


def dedup(jobs: list[dict]) -> list[dict]:
    """Remove near-duplicate postings (same company + very similar title)."""
    unique = []
    for job in jobs:
        is_dup = False
        for seen in unique:
            if seen["company"].lower() == job["company"].lower():
                similarity = SequenceMatcher(
                    None, seen["title"].lower(), job["title"].lower()
                ).ratio()
                if similarity > 0.85:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(job)
    return unique


def rank(jobs: list[dict], curated_companies: set) -> list[dict]:
    """
    Sort priority (earlier = ranked higher):
      1. Curated 'good' companies first
      2. India-based / explicitly-worldwide-remote before ambiguous remote
      3. Remote before hybrid before on-site
      4. Most recently posted first
    """
    mode_rank = {"Remote": 0, "Hybrid": 1, "On-site": 2}

    return sorted(
        jobs,
        key=lambda j: (
            0 if j["company"].lower() in curated_companies else 1,
            location_tier(j),
            mode_rank.get(j.get("work_mode"), 2),
            j.get("posted") or "0000-00-00",
        ),
        reverse=False,
    )
