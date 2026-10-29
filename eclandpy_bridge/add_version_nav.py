#!/usr/bin/env python3
"""Add the clickable run navigation to an eclandpy LIAISE dashboard page.

  add_version_nav.py SRC DST [CURRENT]

CURRENT is "" for the landing page (links are v1/ ...) or e.g. "v3" for a version page
(links become ../v1/ ... and v3 is marked as the current page). Idempotent: a nav added by
an earlier run is stripped first, so re-running never stacks two rows.

The dashboards are published to https://sites.ecmwf.int/pad/liaise/eclandpy/ with sitesctl;
this runs over a local copy of each index.html before upload.
"""
import re, sys

VERSIONS = [
    ("v1", ""),
    ("v2", ""),
    ("v3", ""),
    ("v4", "urban tile on, 10 tiles"),
    ("v5", "rebased on upstream main"),
]

CSS = """/* version-nav start */
.versions{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:20px 0 4px}
.versions .vlabel{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.versions a{display:inline-flex;align-items:baseline;gap:6px;padding:6px 13px;border:1px solid var(--line);
border-radius:999px;background:var(--surface);color:var(--accent);text-decoration:none;font-size:14px}
.versions a:hover,.versions a:focus-visible{border-color:var(--accent);outline:none}
.versions a .vnote{color:var(--muted);font-size:12px}
.versions a[aria-current="page"]{background:var(--accent);border-color:var(--accent);color:#fff}
.versions a[aria-current="page"] .vnote{color:rgba(255,255,255,.75)}
/* version-nav end */"""

def build_nav(current: str) -> str:
    prefix = "../" if current else ""
    out = ['<nav class="versions" aria-label="Dashboard versions">', '<span class="vlabel">Runs</span>']
    for v, note in VERSIONS:
        cur = ' aria-current="page"' if v == current else ""
        label = f'{v} <span class="vnote">{note}</span>' if note else v
        out.append(f'<a href="{prefix}{v}/"{cur}>{label}</a>')
    out.append("</nav>")
    return "\n".join(out)

def patch(html: str, current: str = "") -> str:
    html = re.sub(r'<nav class="versions".*?</nav>\n?', "", html, flags=re.S)
    html = re.sub(r'/\* version-nav start \*/.*?/\* version-nav end \*/', "", html, flags=re.S)
    if "</style>" not in html:
        raise SystemExit("no <style> block - refusing to guess where the CSS goes")
    html = html.replace("</style>", CSS + "</style>", 1)
    nav = build_nav(current)
    if "</header>" in html:
        return html.replace("</header>", "</header>\n" + nav, 1)
    m = re.search(r'<div class="wrap">', html)
    if not m:
        raise SystemExit("no </header> and no .wrap - refusing to guess where the nav goes")
    return html[:m.end()] + "\n" + nav + html[m.end():]

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    cur = sys.argv[3] if len(sys.argv) > 3 else ""
    h = open(src, encoding="utf-8").read()
    open(dst, "w", encoding="utf-8").write(patch(h, cur))
    print(f"  {dst.split('/')[-2]}/{dst.split('/')[-1]}  current={cur or '(landing)'}")
