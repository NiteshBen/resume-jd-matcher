# resume-jd-matcher

Score **how well a resume actually answers a job description** — before anyone
wastes an application. Built by [Nitesh Beniwal](https://github.com/NiteshBen01)
during a real worldwide-remote job hunt, after watching too many postings pass a
keyword filter while hiding real gaps.

The matcher combines two signals that answer different questions:

| signal | question it answers | weight |
|---|---|---|
| TF-IDF cosine similarity (scikit-learn) | does the resume *speak the JD's language* overall? | 0.4 |
| weighted skill coverage | are the JD's **required** skills actually present? | 0.6 |

Skill coverage is weighted, not binary: core skills (python, sql, LLM/agent
stack) count more than generic ones, and skills are classified as
**required vs nice-to-have** by reading the sentence they appear in
("must have", "3+ years", "proficient in" → required; "bonus points",
"familiar with" → nice).

## Quick start

```bash
pip install -r requirements.txt

# score one resume against one JD, print + save a markdown report
python cli.py match --resume tests/data/sample_resume.md --jd tests/data/sample_jd.txt --out report.md

# batch: rank a CSV of postings (columns: title, description[, url])
python cli.py batch --resume resume.md --jobs postings.csv --out ranked.csv
```

## Sample output (real run)

`python cli.py match --resume sample_resume.md --jd sample_jd.txt`
(the JD is Supabase's Support Engineer (APAC) posting):

```
Score: 37.7/100 (cosine 0.04 x 0.4 + skill coverage 0.60 x 0.6)

Present skills: communication, git, sql

Required skills missing from the resume:
- javascript
- linux
- testing
```

That 37.7 is honest: the resume's SQL and communication are real, and so are
the Linux/JavaScript/testing gaps for that particular role — which is exactly
what makes the score trustworthy. Full report in
[`sample_output/report_supabase_support_engineer.md`](sample_output/report_supabase_support_engineer.md).

## Why not just cosine similarity?

A resume and a JD can share vocabulary without sharing requirements. In the
sample above, cosine similarity alone (0.04) would rank almost everything near
zero; coverage alone would be fooled by JDs that mention a skill once in a
"bonus" sentence. Blending them (0.4 / 0.6, both configurable in
`matcher/match.py:score_match`) keeps the ranking stable and the missing-skill
list actionable — every missing skill names the next thing to learn or reword.

## Extending the lexicon

`matcher/extract.py:SKILL_ALIASES` maps canonical skills to surface forms
("postgres" → `sql`, "genai" → `generative ai`). Aliases are matched as whole
words, so `java` never matches `javascript`. Add a row when a JD uses a new
term; `tests/test_match.py` shows the expected behavior.

## Project layout

```
matcher/
  extract.py    # skill lexicon, word-boundary matching, required/nice classification
  match.py      # TF-IDF cosine + weighted coverage -> 0-100 score
  report.py     # markdown report renderer
cli.py          # match / batch commands
tests/          # pytest: alias matching, requirement classification, ranking sanity
```

## License

MIT — see [LICENSE](LICENSE).
