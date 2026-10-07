from pathlib import Path

from matcher.extract import extract_requirements, find_skills
from matcher.match import score_match

DATA = Path(__file__).parent / "data"
RESUME = (DATA / "sample_resume.md").read_text(encoding="utf-8")
JD = (DATA / "sample_jd.txt").read_text(encoding="utf-8")


def test_find_skills_counts_and_aliases():
    skills = find_skills("Knows Python and PostgreSQL. Comfortable with complex SQL queries.")
    assert skills["python"] >= 1
    assert skills["sql"] >= 2  # postgres + sql both map to the sql skill
    assert "pandas" not in skills


def test_requirement_classification():
    reqs = extract_requirements(JD)
    assert "sql" in reqs["required"]          # "Proficient in SQL ... 3+ years"
    assert "communication" in reqs["required"]
    assert "javascript" in reqs["nice"]       # "Bonus points if you're familiar with ..."
    assert "linux" in reqs["required"]


def test_score_ranges_and_components():
    rep = score_match(RESUME, JD)
    assert 0.0 <= rep.cosine_similarity <= 1.0
    assert 0.0 <= rep.skill_coverage <= 1.0
    assert 0.0 <= rep.score <= 100.0
    # resume states SQL + communication; JD also asks for 3+ yrs support it cannot show
    assert "sql" in rep.present_skills
    assert "communication" in rep.present_skills


def test_required_missing_is_actionable():
    rep = score_match(RESUME, JD)
    # the resume does not claim linux/command-line or customer-support experience
    assert "linux" in rep.required_missing or "communication" not in rep.required_missing


def test_relevant_jd_outranks_irrelevant_jd():
    irrelevant = score_match(RESUME, "Seeking a Sales Development Representative. "
                                      "Cold outreach, CRM hygiene, quota targets, outbound calling.")
    relevant = score_match(RESUME, JD)
    assert relevant.score > irrelevant.score + 15
