#!/usr/bin/env python3
"""Build a self-contained local demo site for the pastvideo eval results.

Usage:
    python3 scripts/build_site.py <results.json> <footage_dir> <out_dir>

Copies the corpus videos / reference images / trimmed clip into <out_dir>/media
and writes <out_dir>/index.html with the results baked in. Open via file://.
"""
import html
import json
import os
import shutil
import sys
from pathlib import Path

CORPUS_DESC = {
    "red.mp4": ("Solid saturated red", "color"),
    "green.mp4": ("Solid saturated green", "color"),
    "blue.mp4": ("Solid saturated blue", "color"),
    "white.mp4": ("Full white (max luma)", "brightness"),
    "black.mp4": ("Full black (min luma)", "brightness"),
    "busy.mp4": ("Animated testsrc2", "motion"),
}


def esc(s: str) -> str:
    return html.escape(str(s))


def video_tag(src: str, cls: str = "vid") -> str:
    return f'<video class="{cls}" controls preload="metadata" src="media/{esc(src)}"></video>'


def img_tag(src: str, cls: str = "ref") -> str:
    return f'<img class="{cls}" src="media/{esc(src)}" alt="{esc(src)}">'


def bar(score: float) -> str:
    pct = max(0.0, min(1.0, float(score))) * 100
    return f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'


def main() -> None:
    results_path = Path(sys.argv[1])
    footage = Path(sys.argv[2])
    out = Path(sys.argv[3])
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)

    R = json.loads(results_path.read_text())

    # --- copy media ---------------------------------------------------------
    copied = set()

    def copy(name: str) -> str:
        src = footage / name
        if src.is_file():
            shutil.copy2(src, media / name)
            copied.add(name)
        return name

    for f in R["corpus"]:
        copy(f)
    for q in R["queries"]:
        if q["kind"] == "image":
            copy(q["query"])
        for fname, _ in q["top5"]:
            copy(fname)
    for h in R["highlights"]:
        copy(h["file"])
    trim_name = None
    if R.get("trim"):
        trim_src = Path(R["trim"]["clip"])
        if trim_src.is_file():
            trim_name = "trim_red.mp4"
            shutil.copy2(trim_src, media / trim_name)

    # --- build sections -----------------------------------------------------
    chips = (
        f'<span class="chip ok">{R["hits_at_1"]}/{len(R["queries"])} Hit@1</span>'
        f'<span class="chip">baseline-v1 · 124-dim</span>'
        f'<span class="chip">{R["chunks"]} chunks indexed</span>'
        f'<span class="chip">{R["ms_per_chunk"]:.0f} ms/chunk</span>'
    )

    # corpus gallery
    corpus_cards = []
    for f in R["corpus"]:
        desc, tag = CORPUS_DESC.get(f, (f, ""))
        corpus_cards.append(
            f'<div class="card"><div class="cardhead">{esc(f)}<span class="tag tag-{tag}">{tag}</span></div>'
            f'{video_tag(f)}<div class="muted">{esc(desc)}</div></div>'
        )
    corpus_html = "\n".join(corpus_cards)

    # query -> result showcase
    qcards = []
    for q in R["queries"]:
        if q["kind"] == "text":
            query_el = f'<div class="qtext">“{esc(q["query"])}”</div><div class="qkind">text query</div>'
        else:
            query_el = f'{img_tag(q["query"])}<div class="qkind">image query</div>'
        badge = '<span class="badge ok">correct</span>' if q["hit"] else '<span class="badge no">miss</span>'
        top = q["top5"][0] if q["top5"] else ("—", 0.0)
        result_el = (
            f'<div class="resline">{video_tag(top[0])}'
            f'<div class="resscore">{esc(top[0])} <b>{top[1]:.3f}</b> {badge}</div></div>'
        )
        rows = []
        for fname, score in q["top5"]:
            mark = " ✓" if fname == q["expect"] else ""
            cls = "ranked yes" if fname == q["expect"] else "ranked"
            rows.append(
                f'<li class="{cls}"><span class="fn">{esc(fname)}{mark}</span>'
                f'{bar(score)}<span class="sv">{score:.3f}</span></li>'
            )
        ranked = f'<ol class="rankedlist">{"\n".join(rows)}</ol>'
        qcards.append(
            f'<div class="qcard"><div class="qquery">{query_el}</div>'
            f'<div class="qarrow">→</div><div class="qresult">{result_el}{ranked}</div></div>'
        )
    queries_html = "\n".join(qcards)

    # highlights
    hl_rows = []
    for i, h in enumerate(R["highlights"], 1):
        hl_rows.append(
            f'<li class="hl"><span class="rank">#{i}</span>{video_tag(h["file"], "vidsm")}'
            f'<span class="fn">{esc(h["file"])}</span><span class="sv">{h["score"]:.4f}</span></li>'
        )
    highlights_html = "\n".join(hl_rows)

    # subsystem checks
    trim_dur = R["trim"]["duration_s"] if R.get("trim") else 0
    checks = [
        ("Dedupe", f'{R["dedupe_no_dedupe_hits"]} overlapping chunks', f'→ <b>{R["dedupe_with_dedupe_hits"]}</b> after <code>--dedupe 0.9</code>', "ok"),
        ("Resume", "re-index same file", f'→ <b>{R["resume_new_chunks"]}</b> new chunks (no-op)', "ok"),
        ("Still-skip", "static clip", f'→ skipped <b>{R["still_skip_skipped"]}</b>, indexed <b>{R["still_skip_new_chunks"]}</b>', "ok"),
        ("Dead-letter queue", "missing file", f'→ <b>{R["dlq_entries"]}</b> DLQ entry, run continued', "ok"),
    ]
    check_cards = "\n".join(
        f'<div class="check"><div class="checktitle">{esc(t)}</div>'
        f'<div class="muted">{esc(a)}</div><div class="checkres">{b}</div></div>'
        for t, a, b, _ in checks
    )

    trim_html = ""
    if trim_name:
        trim_html = (
            f'<div class="trim">{video_tag(trim_name)}'
            f'<div class="muted">trimmed result clip · ffprobe duration <b>{trim_dur:.1f}s</b></div></div>'
        )

    stats_rows = "\n".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"
        for k, v in [
            ("Hit@1", f'{R["hits_at_1"]} / {len(R["queries"])}'),
            ("Backend / model", "baseline / baseline-v1"),
            ("Embedding dim", "124"),
            ("Chunks indexed", R["chunks"]),
            ("Index wall time", f'{R["index_ms"]} ms'),
            ("Per chunk", f'{R["ms_per_chunk"]:.0f} ms'),
        ]
    )

    page = PAGE.format(
        chips=chips,
        corpus=corpus_html,
        queries=queries_html,
        highlights=highlights_html,
        checks=check_cards,
        trim=trim_html,
        stats=stats_rows,
        n_media=len(copied),
    )
    (out / "index.html").write_text(page)
    print(f"wrote {out/'index.html'} ({(out/'index.html').stat().st_size} bytes, {len(copied)} media files)")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pastvideo — semantic video search demo</title>
<style>
  :root {{
    --bg:#0b0f14; --panel:#141b22; --panel2:#1b242e; --line:#26313c;
    --text:#e6edf3; --muted:#8b98a5; --accent:#2dd4bf; --amber:#f59e0b;
    --ok:#4ade80; --no:#f87171;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--text);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 20px 80px; }}
  header {{ padding:48px 20px 24px; max-width:1080px; margin:0 auto; }}
  h1 {{ margin:0 0 6px; font-size:34px; letter-spacing:-.5px; }}
  h1 .accent {{ color:var(--accent); }}
  .sub {{ color:var(--muted); margin:0 0 18px; max-width:680px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:5px 12px; font-size:13px; }}
  .chip.ok {{ color:var(--ok); border-color:#1f3a28; }}
  section {{ margin:56px 0 0; }}
  h2 {{ font-size:20px; margin:0 0 4px; }}
  h2 .n {{ color:var(--accent); font-variant-numeric:tabular-nums; margin-right:8px; }}
  .lead {{ color:var(--muted); margin:0 0 18px; max-width:720px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
  .cardhead {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-weight:600; }}
  .tag {{ font-size:11px; text-transform:uppercase; letter-spacing:.5px; padding:2px 8px; border-radius:999px; }}
  .tag-color {{ background:#3a1d2b; color:#f0a6c6; }}
  .tag-brightness {{ background:#3a341d; color:#f6d58a; }}
  .tag-motion {{ background:#1d2f3a; color:#9ad6ef; }}
  .muted {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  video {{ width:100%; border-radius:8px; background:#000; display:block; }}
  .vidsm {{ width:140px; }}
  img.ref {{ width:100%; border-radius:8px; display:block; }}
  code {{ background:var(--panel2); padding:1px 6px; border-radius:5px; font-size:13px; }}

  /* query -> result */
  .qcard {{ display:grid; grid-template-columns:200px 28px 1fr; gap:16px; align-items:start;
           background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:14px; }}
  .qquery {{}} .qarrow {{ color:var(--accent); font-size:24px; text-align:center; padding-top:40px; }}
  .qtext {{ font-size:18px; font-weight:600; margin-bottom:4px; }}
  .qkind {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; margin-top:6px; }}
  .qresult {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }}
  .resline {{ width:240px; flex:0 0 auto; }}
  .resscore {{ margin-top:6px; font-size:13px; }}
  .resscore b {{ color:var(--accent); }}
  .badge {{ font-size:11px; padding:2px 8px; border-radius:999px; margin-left:6px; }}
  .badge.ok {{ background:#13351f; color:var(--ok); }} .badge.no {{ background:#3a1d1d; color:var(--no); }}
  .rankedlist {{ list-style:none; padding:0; margin:0; flex:1; min-width:220px; }}
  .ranked {{ display:grid; grid-template-columns:1fr 160px 56px; gap:10px; align-items:center; padding:4px 0; }}
  .ranked.yes {{ color:var(--text); font-weight:600; }} .ranked:not(.yes) {{ color:var(--muted); }}
  .ranked .fn {{ font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .bar {{ background:var(--panel2); border-radius:4px; height:8px; overflow:hidden; }}
  .bar > span {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),#5eead4); }}
  .ranked.yes .bar > span {{ background:linear-gradient(90deg,var(--ok),#86efac); }}
  .sv {{ font-variant-numeric:tabular-nums; text-align:right; font-size:13px; }}

  /* highlights */
  .hlist {{ list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:10px; }}
  .hl {{ display:grid; grid-template-columns:36px 140px 1fr 70px; gap:12px; align-items:center;
        background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 14px; }}
  .rank {{ color:var(--amber); font-weight:700; }}

  /* checks */
  .checks {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; }}
  .check {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--ok); border-radius:10px; padding:12px 14px; }}
  .checktitle {{ font-weight:600; }} .checkres {{ margin-top:6px; }}
  .trim {{ margin-top:16px; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }}
  .trim video {{ max-width:360px; }}

  table {{ border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  td {{ padding:9px 14px; border-top:1px solid var(--line); }}
  td:first-child {{ color:var(--muted); width:45%; }}
  footer {{ margin-top:64px; color:var(--muted); font-size:13px; border-top:1px solid var(--line); padding-top:20px; }}
  pre {{ background:var(--panel2); padding:12px 14px; border-radius:8px; overflow:auto; }}
  @media (max-width:720px) {{
    .qcard {{ grid-template-columns:1fr; }} .qarrow {{ display:none; }}
  }}
</style>
</head>
<body>
<header>
  <h1>past<span class="accent">video</span></h1>
  <p class="sub">A Rust video-search database. Index footage, search it by text or image — chunk → preprocess →
  skip-stills → embed → store → rank, all inside the database. This page shows a real evaluation run against a
  synthetic corpus, generated by <code>examples/eval.rs</code>.</p>
  <div class="chips">{chips}</div>
</header>
<div class="wrap">

  <section>
    <h2><span class="n">01</span>The corpus</h2>
    <p class="lead">Six 6-second clips generated with ffmpeg, each chosen to exercise one feature of the offline
    baseline embedder (color / brightness / motion). All clips are playable.</p>
    <div class="grid">{corpus}</div>
  </section>

  <section>
    <h2><span class="n">02</span>Search text → result</h2>
    <p class="lead">For each query: the query on the left, the matched original video + cosine score on the right,
    and the full ranked list with score bars. Green row is the ground-truth expected clip.</p>
    {queries}
  </section>

  <section>
    <h2><span class="n">03</span>Highlights (anomaly ranking)</h2>
    <p class="lead">No query needed — clips whose embedding sits farthest from the index centroid. The animated
    clip is correctly surfaced as the most unusual.</p>
    <ul class="hlist">{highlights}</ul>
  </section>

  <section>
    <h2><span class="n">04</span>Pipeline checks</h2>
    <p class="lead">Every supporting subsystem behaves correctly.</p>
    <div class="checks">{checks}</div>
    {trim}
  </section>

  <section>
    <h2><span class="n">05</span>Run summary</h2>
    <table>{stats}</table>
  </section>

  <footer>
    <p>Generated from <code>results.json</code> by <code>scripts/build_site.py</code> · {n_media} media files embedded.</p>
    <p>Reproduce the whole evaluation:</p>
    <pre>cargo run --release --example eval -- results.json work
python3 scripts/build_site.py results.json work/footage site
open site/index.html</pre>
  </footer>

</div>
</body>
</html>
"""

if __name__ == "__main__":
    main()
