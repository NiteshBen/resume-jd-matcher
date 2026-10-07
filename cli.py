"""Command-line interface.

Single match:
    python cli.py match --resume sample_resume.md --jd job_description.txt --out report.md

Batch: score a CSV of job postings (columns: title, description):
    python cli.py batch --resume sample_resume.md --jobs postings.csv --out ranked.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from matcher.match import score_match
from matcher.report import render_report, write_report


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def cmd_match(args: argparse.Namespace) -> None:
    resume, jd = _read(args.resume), _read(args.jd)
    report = score_match(resume, jd)
    print(render_report(Path(args.resume).name, Path(args.jd).name, report))
    if args.out:
        p = write_report(Path(args.resume).name, Path(args.jd).name, report, args.out)
        print(f"[saved] {p}")


def cmd_batch(args: argparse.Namespace) -> None:
    import pandas as pd

    resume = _read(args.resume)
    df = pd.read_csv(args.jobs)
    rows = []
    for _, r in df.iterrows():
        jd = f"{r.get('title', '')}\n{r.get('description', '')}"
        rep = score_match(resume, jd)
        rows.append({"title": r.get("title", ""), "score": rep.score,
                     "cosine": rep.cosine_similarity, "coverage": rep.skill_coverage,
                     "missing_required": ", ".join(rep.required_missing[:6]),
                     "url": r.get("url", "")})
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(out.head(15).to_string(index=False))
    print(f"\n[saved] {args.out} ({len(out)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="resume-jd-matcher",
                                 description="Score how well a resume answers a job description.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("match", help="score one resume against one JD")
    sp.add_argument("--resume", required=True)
    sp.add_argument("--jd", required=True)
    sp.add_argument("--out", default="", help="optional .md output path")

    bp = sub.add_parser("batch", help="score a resume against a CSV of postings")
    bp.add_argument("--resume", required=True)
    bp.add_argument("--jobs", required=True)
    bp.add_argument("--out", default="ranked.csv")

    args = ap.parse_args()
    {"match": cmd_match, "batch": cmd_batch}[args.cmd](args)


if __name__ == "__main__":
    main()
