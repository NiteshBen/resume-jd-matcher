"""Command-line interface.

Single match:
    python cli.py match --resume sample_resume.md --jd job_description.txt --out report.md

Batch: score a CSV of job postings (columns: title, description):
    python cli.py batch --resume sample_resume.md --jobs postings.csv --out ranked.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from matcher.match import score_match
from matcher.report import render_json, render_report, write_report


def _read(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        sys.exit(f"Error: file not found: {path}")
    return p.read_text(encoding="utf-8", errors="ignore")


def cmd_match(args: argparse.Namespace) -> None:
    resume, jd = _read(args.resume), _read(args.jd)
    report = score_match(
        resume,
        jd,
        cosine_weight=args.cosine_weight,
        coverage_weight=args.coverage_weight,
    )

    resume_name = Path(args.resume).name
    jd_name = Path(args.jd).name

    if args.json:
        print(render_json(resume_name, jd_name, report))
    else:
        print(render_report(resume_name, jd_name, report))

    if args.out:
        p = write_report(resume_name, jd_name, report, args.out)
        print(f"[saved] {p}")

    if args.min_score is not None and report.score < args.min_score:
        sys.exit(f"Score {report.score} is below required minimum of {args.min_score}")


def cmd_batch(args: argparse.Namespace) -> None:
    import pandas as pd

    resume = _read(args.resume)
    jobs_path = Path(args.jobs)
    if not jobs_path.is_file():
        sys.exit(f"Error: file not found: {args.jobs}")

    try:
        df = pd.read_csv(jobs_path)
    except Exception as exc:
        sys.exit(f"Error reading {args.jobs}: {exc}")

    if df.empty:
        print("Warning: postings CSV is empty.")
        if args.out:
            Path(args.out).touch()
        return

    # Flexible column resolution
    title_col = next((c for c in ["title", "job_title", "role", "position"] if c in df.columns), None)
    desc_col = next((c for c in ["description", "job_description", "details", "text"] if c in df.columns), None)
    url_col = next((c for c in ["url", "link", "apply_url"] if c in df.columns), None)

    rows = []
    for _, r in df.iterrows():
        title = str(r[title_col]) if title_col and pd.notna(r[title_col]) else ""
        desc = str(r[desc_col]) if desc_col and pd.notna(r[desc_col]) else ""
        url = str(r[url_col]) if url_col and pd.notna(r[url_col]) else ""

        jd = f"{title}\n{desc}".strip()
        rep = score_match(resume, jd)

        if args.min_score is not None and rep.score < args.min_score:
            continue

        rows.append({
            "title": title,
            "score": rep.score,
            "cosine": rep.cosine_similarity,
            "coverage": rep.skill_coverage,
            "missing_required": ", ".join(rep.required_missing[:6]),
            "url": url,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json or out_path.suffix.lower() == ".json":
        data = out.to_dict(orient="records")
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if not args.quiet:
            print(json.dumps(data[:15], indent=2))
    else:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        if not args.quiet:
            print(out.head(15).to_string(index=False))

    print(f"\n[saved] {out_path} ({len(out)} rows matched)")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="resume-jd-matcher",
        description="Score how well a resume answers a job description.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("match", help="score one resume against one JD")
    sp.add_argument("--resume", required=True, help="path to resume file (.md, .txt)")
    sp.add_argument("--jd", required=True, help="path to job description file")
    sp.add_argument("--out", default="", help="optional output path (.md or .json)")
    sp.add_argument("--json", action="store_true", help="output report as JSON")
    sp.add_argument("--min-score", type=float, default=None, help="minimum score threshold (exits 1 if below)")
    sp.add_argument("--cosine-weight", type=float, default=0.4, help="weight for TF-IDF cosine similarity")
    sp.add_argument("--coverage-weight", type=float, default=0.6, help="weight for skill coverage")

    bp = sub.add_parser("batch", help="score a resume against a CSV of postings")
    bp.add_argument("--resume", required=True, help="path to resume file")
    bp.add_argument("--jobs", required=True, help="CSV file of job postings")
    bp.add_argument("--out", default="ranked.csv", help="output path (.csv or .json)")
    bp.add_argument("--min-score", type=float, default=None, help="filter out postings below this score")
    bp.add_argument("--json", action="store_true", help="save / display results in JSON format")
    bp.add_argument("--quiet", "-q", action="store_true", help="do not print table to stdout")

    args = ap.parse_args()
    {"match": cmd_match, "batch": cmd_batch}[args.cmd](args)


if __name__ == "__main__":
    main()
