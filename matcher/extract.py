"""Skill lexicon and requirement extraction from job descriptions.

The lexicon maps canonical skill names to the surface forms JDs actually use
("Postgres" -> sql, "GenAI" -> generative ai). Every alias is matched as a
whole word/phrase via \\b word boundaries, so "Java" never matches "JavaScript".
"""
from __future__ import annotations

import re

SKILL_ALIASES: dict[str, list[str]] = {
    "python": ["python", "python3"],
    "sql": ["sql", "postgres", "postgresql", "mysql", "sqlite", "sql server"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "tensorflow": ["tensorflow", "keras"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "power bi": ["power bi", "powerbi", "power-bi"],
    "tableau": ["tableau"],
    "excel": ["excel", "spreadsheets"],
    "generative ai": ["generative ai", "genai", "gen-ai", "generative artificial intelligence"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "prompt engineering": ["prompt engineering", "prompt engineer", "prompting"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation", "retrieval-augmented"],
    "ai agents": ["ai agent", "ai agents", "agentic", "multi-agent", "autonomous agent",
                  "crewai", "langchain", "langgraph"],
    "machine learning": ["machine learning", "ml engineer", "ml model", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "nlp": ["nlp", "natural language processing", "text classification"],
    "computer vision": ["computer vision", "opencv", "image classification"],
    "data analysis": ["data analysis", "data analyst", "data analytics",
                      "exploratory analysis", "eda"],
    "data visualization": ["data visualization", "dashboard", "dashboards", "charting"],
    "etl": ["etl", "elt", "data pipeline", "data pipelines"],
    "api": ["rest api", "restful", "api", "apis", "api integration", "graphql"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "docker": ["docker", "containeriz", "container", "containers", "containerized", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "vertex ai"],
    "azure": ["azure"],
    "git": ["git", "github", "gitlab", "version control"],
    "java": ["java"],
    "c++": ["c++"],
    "javascript": ["javascript", "typescript", "node.js", "nodejs", "react", "vue", "svelte"],
    "linux": ["linux", "unix", "command line", "bash", "shell"],
    "spark": ["spark", "pyspark"],
    "communication": ["communication", "written and verbal", "stakeholder", "interpersonal"],
    "testing": ["unit test", "unit tests", "pytest", "test coverage", "automated testing", "tdd"],
}

# JD sentences that mark a skill as required rather than nice-to-have.
_REQUIRE_CONTEXT = re.compile(
    r"(required|must have|must-have|minimum of|at least|you will need|"
    r"solid|strong|proficien|excellent|expert|hands-on|skilled|fluen|"
    r"experience (in|with)|\b\d\+? years)", re.I)
_NICE_CONTEXT = re.compile(r"(nice to have|bonus|plus|preferred|desirable|familiar|optional)", re.I)

# Section heading cues in JDs (e.g. 'Requirements:', 'What you will bring:')
_SECTION_REQUIRE = re.compile(
    r"^[#*\s-]*(requirements?|qualifications?|what you(?:'ll| will)? (?:need|bring)|"
    r"what we(?:'re| are) looking for|must[- ]haves?|minimum (?:qualifications|requirements)|"
    r"who you are|you are|core skills?)\b",
    re.I,
)
_SECTION_NICE = re.compile(
    r"^[#*\s-]*(nice[- ]to[- ]haves?|bonus points?|preferred (?:qualifications|skills)|"
    r"good[- ]to[- ]have|plus points?|desirable|optional)\b",
    re.I,
)
_SECTION_RESET = re.compile(
    r"^[#*\s-]*(about (?:us|the (?:company|role|team))|responsibilities|what you(?:'ll| will)? do|"
    r"benefits|perks|compensation|overview|culture)\b",
    re.I,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def _compile_alias_pattern(alias: str) -> re.Pattern:
    r"""Build a regex pattern that matches the alias with boundary safety.

    Uses lookarounds (?<!\w) and (?!\w) which correctly handle non-word
    characters like '+' in 'c++' without requiring a trailing \w.
    """
    prefix = r"(?<!\w)" if re.match(r"^\w", alias) else r"(?<!\S)"
    suffix = r"(?!\w)"
    return re.compile(rf"{prefix}{re.escape(alias)}{suffix}", re.I)


_ALIAS_PATTERNS: dict[str, list[re.Pattern]] = {
    skill: [_compile_alias_pattern(alias) for alias in aliases]
    for skill, aliases in SKILL_ALIASES.items()
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def find_skills(text: str) -> dict[str, int]:
    """Return {skill: mention_count} found anywhere in the text."""
    low = normalize(text).lower()
    found: dict[str, int] = {}
    for skill, patterns in _ALIAS_PATTERNS.items():
        n = sum(len(p.findall(low)) for p in patterns)
        if n:
            found[skill] = n
        # special check for c++ in mixed casing / text if needed
    return found


def extract_requirements(jd: str) -> dict[str, dict[str, int]]:
    """Split the JD into sentences and classify each skill mention as required/nice.

    Context-aware: leverages section headings (e.g. 'Requirements:', 'Bonus:')
    as well as inline linguistic cues ('must have', 'nice to have').
    """
    required: dict[str, int] = {}
    nice: dict[str, int] = {}
    current_section = "unknown"

    for line in (jd or "").splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue

        # Check section headings
        if _SECTION_REQUIRE.search(trimmed):
            current_section = "required"
        elif _SECTION_NICE.search(trimmed):
            current_section = "nice"
        elif _SECTION_RESET.search(trimmed):
            current_section = "reset"

        for sentence in _SENTENCE_SPLIT.split(trimmed):
            skills = find_skills(sentence)
            if not skills:
                continue

            if _NICE_CONTEXT.search(sentence):
                bucket = nice
            elif _REQUIRE_CONTEXT.search(sentence):
                bucket = required
            elif current_section == "required":
                bucket = required
            elif current_section == "nice":
                bucket = nice
            else:
                bucket = nice

            for skill, n in skills.items():
                bucket[skill] = bucket.get(skill, 0) + n

    return {"required": required, "nice": nice}
