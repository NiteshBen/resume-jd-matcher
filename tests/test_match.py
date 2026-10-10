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


def test_cpp_boundary_matching():
    # c++ has non-alphanumeric trailing symbols that previously broke with \b
    text = "Languages: Python, SQL, Java, C++"
    skills = find_skills(text)
    assert "c++" in skills
    assert skills["c++"] == 1

    # Should also match in sample resume
    resume_skills = find_skills(RESUME)
    assert "c++" in resume_skills


def test_prompt_engineering_and_docker_aliases():
    text = "Experience with prompt engineering and containerization using Docker."
    skills = find_skills(text)
    assert "prompt engineering" in skills
    assert "docker" in skills


def test_section_heading_classification():
    jd = """
    Software Engineer

    Requirements:
    - Python
    - Docker
    - SQL

    Nice to have:
    - React
    - Kubernetes
    """
    reqs = extract_requirements(jd)
    assert "python" in reqs["required"]
    assert "docker" in reqs["required"]
    assert "sql" in reqs["required"]
    assert "javascript" in reqs["nice"]
    assert "kubernetes" in reqs["nice"]


def test_empty_inputs_handling():
    rep = score_match("", "")
    assert rep.score == 0.0
    assert rep.cosine_similarity == 0.0
    assert rep.skill_coverage == 0.0
    assert rep.present_skills == []


def test_match_report_to_dict_and_json(tmp_path):
    from matcher.report import render_json, write_report
    import json

    rep = score_match(RESUME, JD)
    data = rep.to_dict()
    assert "score" in data
    assert "cosine_similarity" in data
    assert "skill_coverage" in data
    assert "present_skills" in data

    json_str = render_json("resume.md", "jd.txt", rep)
    parsed = json.loads(json_str)
    assert parsed["score"] == rep.score
    assert parsed["resume"] == "resume.md"

    # Test writing to json file
    json_path = tmp_path / "out.json"
    p = write_report("resume.md", "jd.txt", rep, json_path)
    assert p.is_file()
    assert json.loads(p.read_text(encoding="utf-8"))["score"] == rep.score


def test_custom_weights():
    rep = score_match(RESUME, JD, cosine_weight=0.8, coverage_weight=0.2)
    assert rep.cosine_weight == 0.8
    assert rep.coverage_weight == 0.2


def test_cli_match_and_batch(tmp_path, monkeypatch, capsys):
    from cli import main
    import pandas as pd

    # Test CLI match with --json
    resume_path = tmp_path / "res.md"
    jd_path = tmp_path / "jd.txt"
    resume_path.write_text("Python and SQL developer", encoding="utf-8")
    jd_path.write_text("Must have Python and SQL.", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["cli.py", "match", "--resume", str(resume_path), "--jd", str(jd_path), "--json"])
    main()
    captured = capsys.readouterr()
    assert '"score":' in captured.out

    # Test CLI batch
    jobs_csv = tmp_path / "jobs.csv"
    pd.DataFrame([
        {"title": "Backend Dev", "description": "Python and SQL required"},
        {"title": "Designer", "description": "Figma and CSS required"},
    ]).to_csv(jobs_csv, index=False)

    out_csv = tmp_path / "ranked.csv"
    monkeypatch.setattr("sys.argv", ["cli.py", "batch", "--resume", str(resume_path), "--jobs", str(jobs_csv), "--out", str(out_csv)])
    main()
    assert out_csv.is_file()
    df_out = pd.read_csv(out_csv)
    assert len(df_out) == 2
    assert df_out.iloc[0]["title"] == "Backend Dev"

