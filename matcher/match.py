"""Resume-vs-JD scoring: TF-IDF cosine similarity + weighted skill coverage.

The two signals answer different questions:
  - cosine similarity: "does the resume *speak the JD's language* overall?"
  - skill coverage:    "are the JD's required skills actually present?"
The final score blends both so neither keyword lists nor generic prose alone
can carry the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .extract import extract_requirements, find_skills

# skill-specific weight bonus: core technical skills matter more than generic ones
_WEIGHTS: dict[str, float] = {
    "python": 3.0, "sql": 2.5, "generative ai": 3.0, "llm": 3.0, "rag": 2.5,
    "ai agents": 3.0, "machine learning": 3.0, "power bi": 2.5, "pandas": 2.0,
    "tensorflow": 2.5, "pytorch": 2.5, "data analysis": 2.0, "api": 1.5,
    "data visualization": 1.5, "etl": 1.5, "deep learning": 2.0, "nlp": 2.0,
}
_DEFAULT_WEIGHT = 1.0


@dataclass
class MatchReport:
    score: float                     # 0-100 blended match score
    cosine_similarity: float         # 0-1 TF-IDF cosine between resume and JD
    skill_coverage: float            # 0-1 weighted coverage of JD skills
    required_missing: list[str] = field(default_factory=list)
    present_skills: list[str] = field(default_factory=list)
    coverage_detail: dict = field(default_factory=dict)


def _cosine(resume: str, jd: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [resume or "", jd or ""]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        tfidf = vec.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return 0.0
    sim = (tfidf[0] @ tfidf[1].T).toarray()[0][0]
    return float(max(0.0, min(1.0, sim)))


def _skill_coverage(resume: str, jd: str) -> tuple[float, list[str], list[str], dict]:
    resume_skills = find_skills(resume)
    reqs = extract_requirements(jd)
    jd_skills = set(reqs["required"]) | set(reqs["nice"])

    weighted_have = weighted_total = 0.0
    present, missing = [], []
    detail = {}
    for skill in sorted(jd_skills):
        weight = _WEIGHTS.get(skill, _DEFAULT_WEIGHT)
        weighted_total += weight
        count = resume_skills.get(skill, 0)
        detail[skill] = {
            "in_resume": bool(count),
            "required": skill in reqs["required"],
            "jd_mentions": reqs["required"].get(skill, 0) + reqs["nice"].get(skill, 0),
            "weight": weight,
        }
        if count:
            weighted_have += weight
            present.append(skill)
        elif skill in reqs["required"]:
            missing.append(skill)
    coverage = weighted_have / weighted_total if weighted_total else 0.0
    return coverage, present, missing, detail


def score_match(resume: str, jd: str,
                cosine_weight: float = 0.4, coverage_weight: float = 0.6) -> MatchReport:
    cosine = _cosine(resume, jd)
    coverage, present, missing, detail = _skill_coverage(resume, jd)
    blended = 100.0 * (cosine_weight * cosine + coverage_weight * coverage)
    return MatchReport(
        score=round(blended, 1),
        cosine_similarity=round(cosine, 3),
        skill_coverage=round(coverage, 3),
        required_missing=missing,
        present_skills=present,
        coverage_detail=detail,
    )
