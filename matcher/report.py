"""Markdown report generation for a resume-vs-JD match."""
from __future__ import annotations

import json
from pathlib import Path

from .match import MatchReport


def render_report(resume_name: str, jd_name: str, report: MatchReport) -> str:
    lines = [
        f"# Match report: {resume_name} vs {jd_name}",
        "",
        f"**Score: {report.score}/100** "
        f"(cosine {report.cosine_similarity:.2f} x {report.cosine_weight:.2f} + skill coverage "
        f"{report.skill_coverage:.2f} x {report.coverage_weight:.2f})",
        "",
        "## Present skills",
        ", ".join(report.present_skills) if report.present_skills else "_none detected_",
        "",
    ]
    if report.required_missing:
        lines += [
            "## Required skills missing from the resume",
            "",
        ]
        lines += [f"- **{s}**" for s in report.required_missing]
        lines += [
            "",
            "_If you have this skill but worded it differently in your resume, say it the "
            "JD's way; if you don't have it, that's the gap to close._",
            "",
        ]
    lines += ["## Per-skill detail", ""]
    if report.coverage_detail:
        lines += ["| skill | in resume | required by JD | JD mentions |", "|---|---|---|---|"]
        for skill, d in report.coverage_detail.items():
            lines.append(f"| {skill} | {'yes' if d['in_resume'] else 'no'} "
                         f"| {'yes' if d['required'] else 'nice-to-have'} | {d['jd_mentions']} |")
    else:
        lines.append("_No specific skills detected to compare._")
    lines.append("")
    return "\n".join(lines)


def render_json(resume_name: str, jd_name: str, report: MatchReport, indent: int = 2) -> str:
    data = {
        "resume": resume_name,
        "jd": jd_name,
        **report.to_dict(),
    }
    return json.dumps(data, indent=indent)


def write_report(resume_name: str, jd_name: str, report: MatchReport, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(render_json(resume_name, jd_name, report), encoding="utf-8")
    else:
        out.write_text(render_report(resume_name, jd_name, report), encoding="utf-8")
    return out
