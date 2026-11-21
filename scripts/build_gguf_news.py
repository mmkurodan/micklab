#!/usr/bin/env python3
"""GGUF news builder.

Reads public/gguf-news/YYYY-MM-DD.json, then regenerates:
  - public/gguf-news/index.json
  - the <!-- GGUF-NEWS:START/END --> block in public/index.html / index-en.html
  - the <!-- GGUF-NEWS-STATIC:START/END --> block in public/gguf-news.html / gguf-news-en.html
Usage: python3 scripts/build_gguf_news.py
"""
import glob
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(ROOT, "public")
NEWS = os.path.join(PUB, "gguf-news")
STATIC_DAYS = 5

e = html.escape


def load_days():
    days = []
    for p in glob.glob(os.path.join(NEWS, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")):
        with open(p, encoding="utf-8") as f:
            days.append(json.load(f))
    days.sort(key=lambda d: d["date"], reverse=True)
    return days


def title(it, lang):
    return it.get("title_en") or it["title"] if lang == "en" else it["title"]


def item_html(it, lang, indent):
    summ = it["summary_en"] if lang == "en" else it["summary_ja"]
    src = "Source" if lang == "en" else "出典"
    tags = "".join(f'<span class="badge">{e(t)}</span> ' for t in it.get("tags", []))
    return (
        f'{indent}<div class="card news-item">{tags}<h3>{e(title(it, lang))}</h3>'
        f'<p>{e(summ)}</p>'
        f'<p class="news-src"><a href="{e(it["url"])}" target="_blank" rel="noopener">{src}: {e(it["url"])}</a></p></div>\n'
    )


def empty_html(lang, indent):
    msg = "No notable new GGUF releases today." if lang == "en" else "目立った新着なし"
    return f'{indent}<div class="card news-item"><p>{msg}</p></div>\n'


def day_items(day, lang, indent):
    if not day["items"]:
        return empty_html(lang, indent)
    return "".join(item_html(it, lang, indent) for it in day["items"])


def replace_block(path, marker, body):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(src):
        raise SystemExit(f"marker {marker} not found in {path}")
    src = pat.sub(lambda m: start + "\n" + body + "    " + end, src, count=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def build_index(days):
    idx = {
        "updated": days[0]["date"] if days else None,
        "days": [
            {
                "date": d["date"],
                "count": len(d["items"]),
                "headlines": [it["title"] for it in d["items"][:3]],
                "headlines_en": [title(it, "en") for it in d["items"][:3]],
            }
            for d in days
        ],
    }
    with open(os.path.join(NEWS, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_top(days, lang):
    path = os.path.join(PUB, "index-en.html" if lang == "en" else "index.html")
    d = days[0]
    if lang == "en":
        head, sub, more, link = "Latest GGUF Model News", f"Updated {d['date']} (JST) · GGUF models up to 12B and llama.cpp updates", "See past news →", "/gguf-news-en.html"
    else:
        head, sub, more, link = "GGUF モデル最新情報", f"更新日: {d['date']}（日本時間）· 12B 以下の GGUF モデルと llama.cpp の更新", "過去の情報を見る →", "/gguf-news.html"
    body = (
        '    <section class="cat" id="gguf-news">\n'
        f'      <div class="cat-head"><h2>{head}</h2><span class="count">{len(d["items"])}{" items" if lang == "en" else " 件"}</span></div>\n'
        f'      <p class="cat-sub">{e(sub)}</p>\n'
        '      <div class="news-scroll" tabindex="0">\n'
        '      <div class="apps news-list">\n'
        + day_items(d, lang, "        ")
        + '      </div>\n'
        '      </div>\n'
        f'      <p class="cat-sub news-more" style="margin-top:14px"><a href="{link}">{more}</a></p>\n'
        '    </section>\n'
    )
    replace_block(path, "GGUF-NEWS", body)


def build_backlog(days, lang):
    path = os.path.join(PUB, "gguf-news-en.html" if lang == "en" else "gguf-news.html")
    body = ""
    for d in days[:STATIC_DAYS]:
        body += (
            f'    <section class="cat day" id="d-{d["date"]}">\n'
            f'      <div class="cat-head"><h2>{d["date"]}</h2><span class="count">{len(d["items"])}{" items" if lang == "en" else " 件"}</span></div>\n'
            '      <div class="apps news-list">\n'
            + day_items(d, lang, "        ")
            + '      </div>\n    </section>\n'
        )
    replace_block(path, "GGUF-NEWS-STATIC", body)


def main():
    days = load_days()
    if not days:
        raise SystemExit("no data")
    build_index(days)
    for lang in ("ja", "en"):
        build_top(days, lang)
        build_backlog(days, lang)
    print(f"built {len(days)} day(s); latest {days[0]['date']}")


if __name__ == "__main__":
    main()
